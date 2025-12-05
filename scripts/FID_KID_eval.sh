#!/bin/bash

python ./evaluation/FID_KID/fid_kid.py \
    './datasets/YYX-1340/test' '/root/HVTC/results/hvtc-SSP_yyx_vgg-0.5/test_120/images/fake_B' \
    --gpu 0 \
    --batch-size 1