#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=486G
#SBATCH --job-name=train
#SBATCH --output=train.out
#SBATCH --gres=gpu:4
#SBATCH --account=xxx
#SBATCH --mail-user=xxx
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

export NPROC_PER_NODE=4
export NCCL_DEBUG=INFO
export HDF5_USE_FILE_LOCKING='FALSE'
export PARENT=`/bin/hostname -s`
export MPORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))
export CHILDREN=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $PARENT`
export HOSTLIST="$PARENT $CHILDREN"
export WORLD_SIZE=$SLURM_NTASKS
echo $HOSTLIST

echo "Python $PYTHONPATH"

srun ./train-runner.sh
