import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function
from typing import Any
import gc

try:
    from flash_attn import flash_attn_varlen_func
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False

from profiler.utils import ProfileMethod
from profiler.utils.logger import *
from profiler.utils.record_function_tracer import RecordFunctionTracer
from profiler.common.timer import Timer
from profiler.common.timer_stats_store import TimerStatsStore
from .attention_input import AttentionInput


# -------------------------------------------------------
# Build dummy q,k,v tensors from sampled lengths
# -------------------------------------------------------

def _build_varlen_qkv(
    attention_input: AttentionInput,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype=torch.float16,
    device="cuda",
):
    """
    Build Q, K, V and the corresponding cu_seqlens tensors for flash_attn_varlen_func.
    This function matches FA2's required input format strictly.

    - Q shape: [total_q, num_heads, head_dim]
    - K shape: [total_k, num_kv_heads, head_dim]
    - V shape: [total_k, num_kv_heads, head_dim]
    - cu_seqlens_q: prefix-sum of query lengths
    - cu_seqlens_k: prefix-sum of key lengths
    """
    q_len = attention_input.prefill_chunk_size if attention_input.is_prefill else attention_input.batch_size
    kv_len = attention_input.kv_cache_size
    batch_size = attention_input.batch_size

    # Allocate Q/K/V
    Q = torch.randn(q_len, num_heads, head_dim, device=device, dtype=dtype)
    K = torch.randn(kv_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    V = torch.randn(kv_len, num_kv_heads, head_dim, device=device, dtype=dtype)

    # Build cu_seqlens
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]

    Lq_list = [q_len // batch_size] * batch_size
    for Lq in Lq_list:
        cu_seqlens_q.append(cu_seqlens_q[-1] + Lq)
    Lk_list = [kv_len // batch_size] * batch_size
    for Lk in Lk_list:
        cu_seqlens_k.append(cu_seqlens_k[-1] + Lk)

    cu_seqlens_q = torch.tensor(cu_seqlens_q, device=device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor(cu_seqlens_k, device=device, dtype=torch.int32)

    max_seqlen_q = max(Lq_list)
    max_seqlen_k = max(Lk_list)

    return Q, K, V, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k


def _sdpa_attention(Q, K, V, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    dropout_p=0.0, causal=True):
    """PyTorch SDPA fallback for flash_attn_varlen_func. Used when flash_attn is unavailable."""
    batch = cu_seqlens_q.shape[0] - 1
    num_heads = Q.shape[1]
    num_kv_heads = K.shape[1]
    head_dim = Q.shape[2]
    # Allocate properly-shaped [batch, heads, seq, head_dim] tensors for SDPA
    q = torch.randn(batch, num_heads, max_seqlen_q, head_dim, device=Q.device, dtype=Q.dtype)
    k = torch.randn(batch, num_kv_heads, max_seqlen_k, head_dim, device=K.device, dtype=K.dtype)
    v = torch.randn(batch, num_kv_heads, max_seqlen_k, head_dim, device=V.device, dtype=V.dtype)
    # GQA: expand kv heads to match q heads if needed
    if num_kv_heads < num_heads:
        repeat = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal)


# -------------------------------------------------------
# Profile FA kernel
# -------------------------------------------------------

def profile_flash_attention(
    hardware: str,
    model_name: str,
    model_config: Any,
    tp_size: int,
    attention_input: AttentionInput,
    warmup: int = 10,
    repeat: int = 30,
    profile_method: str = "record_function",
    device: str = "cuda",
):
    """
    Profile flash_attn_varlen_func latency for a given batch (Lq_list, Lk_list)
    and generate metadata through utils.build_metadata().

    This function ONLY:
    1) builds Q/K/V + cu_seqlens
    2) profiles FA varlen kernel latency
    3) calls build_metadata()
    """

    attn_fn = flash_attn_varlen_func if _FLASH_ATTN_AVAILABLE else _sdpa_attention
    attn_backend = "FLASH_ATTENTION" if _FLASH_ATTN_AVAILABLE else "SDPA"

    timer_stats_store = TimerStatsStore(profile_method=profile_method)

    num_heads_per_shard = getattr(model_config, "num_attention_heads", 32) // tp_size
    num_kv_heads_per_shard = getattr(model_config, "num_key_value_heads", getattr(model_config, "num_attention_heads", 32)) // tp_size
    head_dim = getattr(model_config, "hidden_size", 4096) // getattr(model_config, "num_attention_heads", 32)

    # -------------------------------------------------------
    # Build varlen Q/K/V inputs
    # -------------------------------------------------------
    (
        Q,
        K,
        V,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    ) = _build_varlen_qkv(
        attention_input=attention_input,
        num_heads=num_heads_per_shard,
        num_kv_heads=num_kv_heads_per_shard,
        head_dim=head_dim,
        device=device,
    )

    # -------------------------------------------------------
    # Warmup
    # -------------------------------------------------------
    try:
        for _ in range(warmup):
            with torch.no_grad():
                _ = attn_fn(
                    Q,
                    K,
                    V,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    0.0,   # dropout_p
                    True,  # causal
                )
        torch.cuda.synchronize()
        timer_stats_store.clear_stats()

        # -------------------------------------------------------
        # Profiling loop
        # -------------------------------------------------------

        prefill_timer = Timer("attn_prefill")
        decode_timer = Timer("attn_decode")

        if profile_method == ProfileMethod.RECORD_FUNCTION.value:
            trace_output_dir = f"perf_models/{hardware}/{model_name}/tp{tp_size}"
            record_function_tracer = RecordFunctionTracer(trace_output_dir)

            with record_function_tracer:
                for _ in range(repeat):
                    if attention_input.is_prefill:
                        with torch.no_grad():
                            with prefill_timer:
                                _ = attn_fn(
                                    Q,
                                    K,
                                    V,
                                    cu_seqlens_q,
                                    cu_seqlens_k,
                                    max_seqlen_q,
                                    max_seqlen_k,
                                    0.0,
                                    True,
                                )
                    else:
                        with torch.no_grad():
                            with decode_timer:
                                _ = attn_fn(
                                    Q,
                                    K,
                                    V,
                                    cu_seqlens_q,
                                    cu_seqlens_k,
                                    max_seqlen_q,
                                    max_seqlen_k,
                                    0.0,
                                    True,
                                )
            torch.cuda.synchronize()
            time_stats = record_function_tracer.get_operation_time_stats()
            record_function_tracer.clean_up()
        
        else:
            for _ in range(repeat):
                if attention_input.is_prefill:
                    with torch.no_grad():
                        with prefill_timer:
                            _ = attn_fn(
                                Q,
                                K,
                                V,
                                cu_seqlens_q,
                                cu_seqlens_k,
                                max_seqlen_q,
                                max_seqlen_k,
                                0.0,
                                True,
                            )
                else:
                    with torch.no_grad():
                        with decode_timer:
                            _ = attn_fn(
                                Q,
                                K,
                                V,
                                cu_seqlens_q,
                                cu_seqlens_k,
                                max_seqlen_q,
                                max_seqlen_k,
                                0.0,
                                True,
                            )
            torch.cuda.synchronize()
            time_stats = timer_stats_store.get_stats()

        # -------------------------------------------------------
        return {
            "time_stats": time_stats,
            "n_embd": getattr(model_config, "hidden_size", 4096),
            "n_q_head": getattr(model_config, "num_attention_heads", 32),
            "n_kv_head": getattr(model_config, "num_key_value_heads", getattr(model_config, "num_attention_heads", 32)),
            "num_tensor_parallel_workers": tp_size,
            "max_model_len": getattr(model_config, "max_position_embeddings", 131072),
            "batch_size": attention_input.batch_size,
            "prefill_chunk_size": attention_input.prefill_chunk_size,
            "kv_cache_size": attention_input.kv_cache_size,
            "is_prefill": attention_input.is_prefill,
            "attention_backend": attn_backend,
        }

    except torch.cuda.OutOfMemoryError:
        log_error(
            f"Out of Memory! AttentionInput: {attention_input}"
        )
        torch.cuda.empty_cache()
        gc.collect()
        return None
