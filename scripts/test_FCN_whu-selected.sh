#!/bin/bash

python test.py  \
 --dataroot  ./datasets/whu \
 --num_classes 8 \
 --model hvtc_gan_FCN \
 --name FCN-whu \
 --epoch latest \
 --eval
