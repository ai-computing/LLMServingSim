#!/bin/bash

rbln-stat --query-npu=timestamp,index,utilization.npu,power.draw --format=csv,noheader,nounits -lms 1000 -i 0 > rngd_power_log.txt
