#!/bin/bash

python ./evaluation/seg_eval.py \
    --result_path ./results/FCN-yyx/test_latest/images \
    --dataset yyx \
    --label_path ./datasets/YYX-1340/lbl \
