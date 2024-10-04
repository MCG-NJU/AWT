#!/bin/bash

# custom config
DATA=/data3/zhuyuhan/CoOp_dataset
TRAINER=AWT

DATASET=$1
CFG=$2
SHOTS=$3

# change to 25 if 8/16 shots
EPOCH=20

for SEED in 1 2 3
do
    DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
    if [ -d "$DIR" ]; then
        rm -rf ${DIR}
    fi
        python -u train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
        --load-epoch ${EPOCH} \
        --eval-only
done