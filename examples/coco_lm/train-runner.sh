#!/bin/bash

export NCCL_BLOCKING_WAIT=1

echo "Num of node, $SLURM_JOB_NUM_NODES"
echo "Num of GPU per node, $NPROC_PER_NODE"
echo "PROCID: $SLURM_PROCID"
echo "LOCALID: $SLURM_LOCALID"
echo $PARENT

DATA_DIR=xxxx
TOTAL_UPDATES=100000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=8        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x
SEED=100
TOTAL_GPUS=`expr ${SLURM_JOB_NUM_NODES} \* ${NPROC_PER_NODE}`

python -m torch.distributed.launch --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${SLURM_JOB_NUM_NODES} --node_rank=$SLURM_PROCID --master_addr="$PARENT" \
    --master_port="$MPORT" \
    $(which fairseq-train) --data-path ${DATA_DIR} --fp16 --log-interval 200 --num-workers 4 \
    --reset-optimizer --reset-meters --reset-lr-scheduler \
    --arch cocolm --task cocolm --rel-pos-bins 64 \
    --restore-file /path/to/coco_lm/model_edited.pt \
    --criterion cocolm --batch-size 16 --ignore-unused-valid-subsets \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 2.0 \
    --lr-scheduler polynomial_decay --lr ${PEAK_LR} --warmup-updates ${WARMUP_UPDATES} --total-num-update ${TOTAL_UPDATES} \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --update-freq ${UPDATE_FREQ} --seed ${SEED} \
    --sample-break-mode complete --tokens-per-sample ${TOKENS_PER_SAMPLE} \
    --seq-contrast --span 0.9 --add-span-cls --seq-contrast-dim 768 --temperature 1.0 \
    --scl-loss-weight 1.0 --binary-loss-weight 50 --clm
