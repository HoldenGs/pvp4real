#!/bin/bash

DEVICES=1
EXP_NAME="pvp4real"
BC_LOSS_WEIGHT=1.0
BATCH_SIZE=1024
WANDB=true
WANDB_PROJECT="pvp4real MetaUrban"
WANDB_TEAM="rathulucla"

CMD="python pvp/experiments/metadrive/train_pvp_metadrive_fakehuman.py \
--exp_name=\"${EXP_NAME}\" \
--bc_loss_weight=${BC_LOSS_WEIGHT} \
--batch_size=${BATCH_SIZE}"

if [ ${WANDB} = true ]; then
    CMD="${CMD} \
--wandb \
--wandb_project=\"${WANDB_PROJECT}\" \
--wandb_team=\"${WANDB_TEAM}\""
fi

EXP_NAME=${EXP_NAME}_bc${BC_LOSS_WEIGHT}_bs${BATCH_SIZE}

echo ${CMD}
eval "CUDA_VISIBLE_DEVICES=${DEVICES} ${CMD}"
