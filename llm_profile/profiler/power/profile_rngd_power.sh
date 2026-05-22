#!/bin/bash

while true; do
    TIMESTAMP=$(date +"%Y/%m/%d %T.000")
    furiosa-smi info --format json | \
        jq -r ".[] | \"$TIMESTAMP, \(.dev_name), \(.power | split(\" \") | .[0])\"" >> rngd_power_log.txt
    sleep 1
done
