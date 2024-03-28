#!/bin/bash

# Activate the Python environment
source /home/Austin/2024_ICME_Challenge/vhr/bin/activate

# Change the directory
cd /home/Austin/2024_ICME_Challenge/Animal-Kingdom/Animal_Kingdom/pose_estimation/code/hrnet/

echo start up server
nohup python tools/train.py --cfg experiments/mpii/vhrbirdpose/w32_256x256_adam_lr1e-3_ak_vhr_s.yaml > /home/Austin/2024_ICME_Challenge/output.txt &