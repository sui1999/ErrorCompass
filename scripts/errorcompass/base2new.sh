#!/bin/bash
set -e

# Run from the repository root.
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 DATA_DIR DATASET BiomedCLIP" >&2
    exit 2
fi
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Error: DATA_DIR, DATASET, and MODEL must not be empty." >&2
    exit 2
fi
if [ "$3" != "BiomedCLIP" ]; then
    echo "Error: this release supports MODEL=BiomedCLIP only." >&2
    exit 2
fi

DATA=$1
DATASET=$2
MODEL=$3

METHOD=ErrorCompass
TRAINER=ErrorCompass_${MODEL}

SHOTS=16
LOADEP=100
CTP=end
CSC=False
NCTX=4
N_PROMPTS="${N_PROMPTS:-50}"
SUB_base=base
SUB_novel=new
OUTPUT_ROOT="${OUTPUT_ROOT:-output/errorcompass_paper_v3}/base2new_nprompts_${N_PROMPTS}"

for SEED in 1 2 3
do
DIR=${OUTPUT_ROOT}/train_${SUB_base}/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    echo "Training with N_PROMPTS=${N_PROMPTS}"
    python train.py \
    --root "${DATA}" \
    --seed "${SEED}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/${METHOD}/base/${DATASET}.yaml" \
    --output-dir "${DIR}" \
    DATASET.NUM_SHOTS "${SHOTS}" \
    TRAINER.ERRORCOMPASS.N_PROMPTS "${N_PROMPTS}" \
    DATASET.SUBSAMPLE_CLASSES "${SUB_base}"
fi
COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
MODEL_DIR=${OUTPUT_ROOT}/train_${SUB_base}/${COMMON_DIR}
DIR=${OUTPUT_ROOT}/test_${SUB_novel}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    echo "Testing with N_PROMPTS=${N_PROMPTS}"
    python train.py \
    --root "${DATA}" \
    --seed "${SEED}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/${METHOD}/base/${DATASET}.yaml" \
    --output-dir "${DIR}" \
    --model-dir "${MODEL_DIR}" \
    --load-epoch "${LOADEP}" \
    --eval-only \
    DATASET.NUM_SHOTS "${SHOTS}" \
    TRAINER.ERRORCOMPASS.N_PROMPTS "${N_PROMPTS}" \
    DATASET.SUBSAMPLE_CLASSES "${SUB_novel}"
fi
done
