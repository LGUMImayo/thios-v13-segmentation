#!/bin/bash
#SBATCH --job-name=thios_v13
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/thios_v13_%j.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/thios_v13_%j.err
#SBATCH --partition=gpu-n24-170g-4x-a100-40g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --time=15-00:00:00

# ============================================================================
# ThioS V13 Training — Conservative Boundary Segmentation
# ============================================================================
#
# GOAL: Combine V5/V6's precise boundary detection with V11's corrected labels.
#
# KEY INSIGHT: V12's high IoU (0.917) came from aggressive over-segmentation
# that flood-fills regions. V5/V6 had lower IoU (0.36) but MUCH better boundary
# precision because they were conservative — only predicting where confident.
#
# V12 → V13 changes:
#
# 1. BALANCED TVERSKY α=0.5/β=0.5 for ALL classes (was α=0.6/β=0.4)
#    Equal penalty for false positives and false negatives.
#    No bias toward over- or under-segmentation.
#
# 2. REMOVED BOUNDARY LOSS entirely (was weight=0.2)
#    Boundary loss trained on SPARSE ANNOTATION edges, not real tissue edges.
#    This taught the model to match arbitrary label cutoffs, not biology.
#
# 3. REMOVED BOUNDARY WEIGHT MAPS (was boost=5x within 3px)
#    Same problem: boosted weight at annotation edges, not tissue edges.
#
# 4. HALVED CLASS WEIGHTS
#    V12: bg=1.0, diffuse=3.0, plaque=2.0, tangle=4.0
#    V13: bg=1.0, diffuse=1.5, plaque=1.5, tangle=2.0
#    Lower weights = less incentive to aggressively predict rare classes.
#
# 5. REDUCED OVERSAMPLING
#    V12: tangle 3x, diffuse 2x
#    V13: tangle 2x, diffuse 1x (no extra), plaque 1x
#    Less exposure to rare-class patches = more conservative predictions.
#
# 6. EQUAL FOCAL/TVERSKY weighting (0.5/0.5, was 0.3/0.5)
#    Higher focal weight makes model more discriminative (not just overlap).
#
# Data: Same patches_v11 with Erica's corrected labels (22,397 patches)
# Architecture: Same EfficientNet-B4 + UNet decoder
# ============================================================================

set -e

echo "============================================================"
echo "ThioS V13 Training — Conservative Boundaries — $(date)"
echo "============================================================"

source /home/ext_chen_yuheng_mayo_edu/miniconda3/etc/profile.d/conda.sh
conda activate rhizonet

nvidia-smi

BASE_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/ThioS_classification"
SCRIPTS_DIR="${BASE_DIR}/scripts"
CONFIG_DIR="${BASE_DIR}/configs"

PATCHES_DIR="${BASE_DIR}/patches_v11"
LOGS_DIR="${BASE_DIR}/logs/efficientnet_b4_v13_kfold"

mkdir -p "${LOGS_DIR}"

echo ""
echo "============================================================"
echo "V13 Training (5-Fold Stratified Cross-Validation)"
echo "============================================================"
echo "Using patches from: ${PATCHES_DIR}"
echo "Log dir: ${LOGS_DIR}"
echo ""

TRAIN_CONFIG="${CONFIG_DIR}/setup-train_thios_efficientnet_v13.json"

echo "Training config: ${TRAIN_CONFIG}"
echo ""
echo "V13 Key changes from V12:"
echo "  Tversky: α=0.5/β=0.5 BALANCED (was α=0.6/β=0.4 precision-biased)"
echo "  Boundary loss: REMOVED (was 0.2)"
echo "  Boundary weight maps: REMOVED (was 5x boost)"
echo "  Class weights: bg=1.0, diff=1.5, plaq=1.5, tang=2.0 (halved)"
echo "  Oversampling: tangle 2x (was 3x), diffuse 1x (was 2x)"
echo "  Loss: 0.5×Focal + 0.5×Tversky (was 0.3+0.5+0.2)"
echo ""

# Run k-fold training
python "${SCRIPTS_DIR}/train_thios_kfold.py" \
    --config_file "${TRAIN_CONFIG}" \
    --n_folds 5 \
    --gpus 1 \
    --accelerator gpu

echo ""
echo "============================================================"
echo "Analyze K-Fold Results"
echo "============================================================"

python "${SCRIPTS_DIR}/analyze_kfold_results.py" \
    --log_dir "${LOGS_DIR}" \
    --n_folds 5

echo ""
echo "============================================================"
echo "ThioS V13 Training Complete — $(date)"
echo "============================================================"
echo ""
echo "Results:"
echo "  - Checkpoints: ${LOGS_DIR}/fold_*/checkpoints/"
echo "  - Summary: ${LOGS_DIR}/kfold_summary.json"
echo "============================================================"
