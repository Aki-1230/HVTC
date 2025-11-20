#!/bin/bash

python test.py  \
 --dataroot  /root/autodl-tmp/ \
 --checkpoints_dir /root/autodl-tmp/ckpt \
 --model hvtc_gan_FCN \
 --name FCN-yyx \
 --eval \
 --num_classes 7 \
 --n_downsampling 4 \
 --epoch latest
