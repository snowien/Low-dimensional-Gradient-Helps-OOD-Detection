#!/bin/bash
k=$1
sample_num=$2

nproc_per_node=4
batch_size=500
model='resnet50' 
data='imagenet'
kernel='NFK' 
port='13254'

for ((i=start_k; i<k; i=i+interval_k))
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python github_Average_gradient.py --nproc_per_node $nproc_per_node --batch_size $batch_size --model $model --data $data --kernel $kernel --k $k --sample_num $sample_num --port $port
done