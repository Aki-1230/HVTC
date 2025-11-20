#!/bin/bash

python test.py  \
 --dataroot  ./datasets/whu-900 \
 --num_classes 8 \
 --model hvtc_gan_FCN \
 --name FCN-whu \
 --epoch latest \
 --eval
