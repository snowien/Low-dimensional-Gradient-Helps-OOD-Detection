#!/bin/bash
start_k=$1
interval_k=$2
k=$3
sample_num=$4

nproc_per_node=1
batch_size=100
model='resnet18' 
data='cifar10'
kernel='NFK' 
port='13246'

for ((i=start_k; i<k; i=i+interval_k))
do
    CUDA_VISIBLE_DEVICES=0 python github_PCA.py --nproc_per_node $nproc_per_node --batch_size $batch_size --model $model --data $data --kernel $kernel --k $k --start_k $i --interval_k $interval_k --sample_num $sample_num --port $port
done