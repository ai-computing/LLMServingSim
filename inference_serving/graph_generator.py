import os
import subprocess
from time import time
from .request import *
from .utils import PID_TAG
from .logger import get_logger

logger = get_logger("GraphGenerator")

def generate_graph(batch, hardware, npu_num, node_id=0, instance_id=0, npu_offset=0, enable_local_offloading=False, event=False):

    cwd = os.getcwd()
    chakra = os.path.join(cwd, "extern/graph_frontend/chakra")
    os.chdir(chakra)

    # PID_TAG must match the path that trace_generator.py wrote to and
    # utils.get_workload() returns; otherwise Chakra fails to find the
    # trace and the main.py loop hangs waiting for the workload.
    if event:
        file_name = f'{PID_TAG}event_handler'
    else:
        file_name = f'{hardware}/{batch.model}/{PID_TAG}instance{instance_id}_batch{batch.batch_id}'

    workload_dir = f'../../../inputs/workload/{file_name}'
    os.makedirs(workload_dir, exist_ok=True)

    cmd = f'python -m chakra.src.converter.converter LLM ' \
            f'--input ../../../inputs/trace/{file_name}.txt ' \
            f'--output ../../../inputs/workload/{file_name}/llm ' \
            f'--num-npus {npu_num} ' \
            f'--npu-offset {npu_offset}'

    if enable_local_offloading:
        cmd += ' --local-offloading'

    logger.debug("Generating graph with command: %s", cmd, extra={"node_id": node_id, "instance_id": instance_id})

    cmd = cmd.split()
    subprocess.run(cmd, text=True)    
    os.chdir(cwd)
    return