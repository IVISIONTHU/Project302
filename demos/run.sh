#!/bin/sh

export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH="/data/wangq/code/caffe-ms/python":$PYTHONPATH
#export LD_PRELOAD="/usr/lib64/libtcmalloc_minimal.so.4"
# source venv/bin/activate
export GLOG_minloglevel=1
python demo.py
