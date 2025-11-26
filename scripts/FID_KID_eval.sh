#!/bin/bash

python ./evaluation/FID_KID/fid_kid.py \
    './datasets/YYX-1340/test' '/root/autodl-tmp/results/FCN-yyx/test_latest/images/fake_B' \
    --gpu 0 \
    --batch-size 1