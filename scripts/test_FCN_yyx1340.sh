#!/bin/bash

python test.py  \
 --dataroot  ./datasets/YYX-1340 \
 --model hvtc_gan_FCN \
 --name FCN-yyx \
 --eval \
 --num_classes 7 \
 --n_downsampling 4 \
 --epoch latest
