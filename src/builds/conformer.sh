#!/bin/bash

#SBATCH --job-name=cnnViT
#SBATCH --account=ai2es
#SBATCH --qos=long
#SBATCH --time=96:00:00
#SBATCH --partition=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gpus-per-node=6
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Number of processes
#SBATCH --error=/home/aevans/slurm_error_logs/ai2es_error_%j.err
#SBATCH --output=/home/aevans/slurm_output_logs/ai2es_log_%j.out

export NGPU=6
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b")
echo Node IP: $head_node_ip
apptainer run --nv /home/aevans/apptainer/pytorch2.sif /home/aevans/miniconda3/bin/python -m torch.distributed.launch \
--use-env \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $NGPU \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint "127.0.0.1:64425" \
/home/aevans/conformer_ml/src/eval_model.py