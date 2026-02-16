# ThioS V13 Segmentation Pipeline

## Overview

Multiclass semantic segmentation of ThioS (Thioflavin-S) fluorescence microscopy
images for Alzheimer's disease neuropathology. Segments four biomarker classes:

| Class ID | Name       | Description                              |
|----------|------------|------------------------------------------|
| 0        | Unlabeled  | No annotation (ignored in loss)          |
| 1        | Background | Labeled background tissue                |
| 2        | Diffuse    | Diffuse amyloid plaques                  |
| 3        | Plaque     | Neuritic/cored plaques                   |
| 4        | Tangle     | Neurofibrillary tangles                  |

## Model Architecture

- **Encoder:** EfficientNet-B4 (ImageNet pretrained, via `timm`)
- **Decoder:** U-Net with attention gates (via MONAI `FlexibleUNet`)
- **Input:** 128×128 RGB patches (pseudo-RGB: green channel = AF488 fluorescence)
- **Output:** 5-class pixel-wise segmentation

## Directory Structure

```
v13_package/
├── README.md                          # This file
├── batch_scripts/
│   └── run_full_pipeline_v13.sh       # SLURM batch script (end-to-end pipeline)
├── configs/
│   ├── setup-train_thios_efficientnet_v13.json  # Training config (model, loss, hyperparams)
│   └── patch_extraction_v11.json      # Patch extraction config
├── scripts/
│   ├── prepare_thios_data_v2.py       # Step 1: CZI → preprocessed TIFF + merged labels
│   ├── prepare_thios_patches.py       # Step 2: Full images → 128×128 patches
│   ├── generate_thios_weight_maps.py  # Step 3: Per-pixel spatial weight maps
│   ├── train_thios_kfold.py           # Step 4: 5-fold stratified CV training
│   ├── unet2D_thios.py               # Core module: model, dataset, losses, augmentations
│   └── analyze_kfold_results.py       # Step 5: Aggregate fold metrics
├── data/                              # Full dataset (excluded from git, ~6.6GB)
│   ├── raw_czi/                       # Original CZI fluorescence images (10 slides)
│   ├── raw_labels/                    # Erica's corrected Arivis label exports (.ome.tiff)
│   ├── processed/
│   │   ├── images/                    # Preprocessed pseudo-RGB TIFFs
│   │   └── labels/                    # Merged multiclass label masks
│   └── patches/
│       ├── images/                    # 27,997 training patches (128×128 TIFF)
│       ├── labels/                    # Corresponding label masks (128×128 PNG)
│       └── weights/                   # Spatial attention weight maps
├── sample_data/                       # 50 sample patches (included in git)
│   └── patches/
│       ├── images/                    # Sample image patches
│       ├── labels/                    # Sample label masks
│       └── weights/                   # Sample weight maps
├── checkpoints/                       # (Populated after training completes)
└── environment/
    └── rhizonet_env.yml               # Conda environment specification
```

## Pipeline Steps

### Step 1: Data Preparation (`prepare_thios_data_v2.py`)

Converts raw CZI fluorescence images + per-class label masks → training-ready format.

- **Image processing:**
  - Reads single-channel AF488 CZI via `pylibCZIrw`
  - Adaptive percentile intensity windowing (P1/P99.5, floor=50, ceiling=20000)
  - Converts to pseudo-RGB: R=0, G=intensity, B=0 (uint8)

- **Label processing:**
  - Reads per-class `.ome.tiff` masks from Arivis export
  - Merges with priority: tangle > plaque > diffuse > background > unlabeled
  - Outlier ROI filtering (removes abnormally large annotations)
  - Shape alignment (±20px tolerance)

### Step 2: Patch Extraction (`prepare_thios_patches.py`)

- 128×128 patches with 64×64 stride (50% overlap)
- Minimum biomarker ratio: 0.1% for inclusion
- Balanced sampling + max 50 pure-background patches per source image
- Output: 27,997 image/label patch pairs

### Step 3: Weight Map Generation (`generate_thios_weight_maps.py`)

- Per-pixel weight maps for training loss
- Inverse class frequency weighting (median frequency balancing)
- Boundary detection (2px dilation) with 2× weight boost at class edges
- Unlabeled pixels (class 0) get weight 0 (ignored)

### Step 4: Training (`train_thios_kfold.py` + `unet2D_thios.py`)

5-fold stratified cross-validation with:

| Hyperparameter       | V13 Value                          |
|----------------------|------------------------------------|
| Optimizer            | Adam, lr=5e-5                      |
| Loss                 | 0.5×Focal + 0.5×Tversky           |
| Tversky α/β          | 0.5/0.5 (balanced FP/FN penalty)  |
| Class weights        | bg=1.0, diff=1.5, plaq=1.5, tang=2.0 |
| Boundary loss        | 0.0 (disabled)                     |
| Dropout              | 0.3                                |
| Weight decay         | 0.01                               |
| Batch size           | 64                                 |
| Max epochs           | 500                                |
| Early stopping       | 50 epochs patience on val_iou      |
| Precision            | 16-mixed (AMP)                     |
| Oversampling         | tangle 2×, diffuse 1× (none)      |

**Data augmentations** (on-the-fly):
- Random horizontal/vertical flips
- Random 90° rotations
- Random affine (scale, rotation, translation)
- Color jitter (brightness, contrast)
- Mixup and CutMix (stochastic)
- Gaussian noise and blur

### Step 5: Results Analysis (`analyze_kfold_results.py`)

Aggregates per-fold metrics (IoU, accuracy, precision, recall) and generates
summary plots and JSON output.

## Training Data

10 ThioS-stained brain sections, fluorescence microscopy (AF488 channel):

| Slide ID | Patches | Notes                  |
|----------|---------|------------------------|
| 019_A    | 4,154   | Erica-corrected labels |
| 019_B    | 5,051   | Erica-corrected labels |
| 019_C    | 2,930   | Erica-corrected labels |
| 033_B    | 2,257   | Erica-corrected labels |
| 033_G    | 2,571   | Original labels        |
| 112_A    | 3,632   | Erica-corrected labels |
| 112_B    | 2,907   | Erica-corrected labels |
| 161_F    | 1,981   | Original labels        |
| 161_G    | 1,008   | Original labels        |
| high_exp | 1,506   | Original labels        |
| **Total**| **27,997** |                     |

**Label corrections (V11 data):** 6/10 slides had labels corrected by
neuropathologist Erica — expanded annotations for diffuse, plaque, and tangle
classes. Two new label files were added (019_A Tangles, 112_B Diffuse/Tangles).

**Annotation sparsity:** Only ~19% of pixels have labels. The majority of each
image is class 0 (unlabeled), which is ignored in the loss function.

## V13 Design Rationale

V13 was designed to address an **over-segmentation problem** found in V10–V12:

- V10/V11/V12 used aggressive class weights and oversampling, which caused the
  model to flood-fill large regions with biomarker predictions
- IoU on sparse labels (only labeled pixels evaluated) **rewards** over-segmentation:
  a model that predicts everything as the correct class scores higher IoU than one
  that precisely segments only actual pathology
- V12's boundary loss made it worse — it trained against annotation edges (wherever
  the annotator stopped labeling), not actual tissue boundaries

V13 reverts to conservative settings matching V5/V6's precise prediction style:
- Balanced Tversky (equal FP/FN penalty)
- Halved class weights (less over-prediction incentive)
- Reduced oversampling (model sees more background, learns restraint)
- No boundary loss (removed annotation artifact training)

## Sample Data

The `sample_data/` directory contains 50 representative patches (5 per source
slide, 4 with biomarker content + 1 background-only) for quick inspection and
testing. Each patch includes the image, label mask, and weight map.

The full dataset (27,997 patches, ~6.6GB) is excluded from this repository due
to size. Contact the authors for access to the complete training data.

## Environment Setup

```bash
conda env create -f environment/rhizonet_env.yml
conda activate rhizonet
```

Key dependencies: PyTorch, MONAI, PyTorch Lightning, timm, scikit-learn,
scikit-image, tifffile, pylibCZIrw.

## Running the Pipeline

To run end-to-end on a SLURM cluster with GPU:

```bash
# Edit paths in batch_scripts/run_full_pipeline_v13.sh and configs/*.json
# Then submit:
sbatch batch_scripts/run_full_pipeline_v13.sh
```

Or run steps individually:

```bash
# Step 1: Preprocess CZI → TIFF
python scripts/prepare_thios_data_v2.py \
    --input_czi_dir data/raw_czi \
    --input_label_dir data/raw_labels \
    --output_dir data/processed \
    --lower_percentile 1.0 --upper_percentile 99.5 \
    --min_floor 50 --max_ceiling 20000 --gamma 1.0 --pure_green --verbose

# Step 2: Extract patches
python scripts/prepare_thios_patches.py --config_file configs/patch_extraction_v11.json

# Step 3: Generate weight maps
python scripts/generate_thios_weight_maps.py --data_dir data/patches

# Step 4: Train (5-fold CV)
python scripts/train_thios_kfold.py \
    --config_file configs/setup-train_thios_efficientnet_v13.json \
    --n_folds 5 --gpus 1 --accelerator gpu

# Step 5: Analyze results
python scripts/analyze_kfold_results.py --log_dir <log_dir> --n_folds 5
```

## Code Architecture

The core module `unet2D_thios.py` contains all model/data definitions:

| Class                       | Purpose                                          |
|-----------------------------|--------------------------------------------------|
| `FlexibleUNetWithAttention` | EfficientNet-B4 encoder + attention-gated U-Net  |
| `AttentionGate`             | Attention mechanism for skip connections         |
| `ThioSDataset`              | Training/val dataset with augmentations           |
| `PredDataset2D`             | Inference-time dataset                           |
| `ThioSUnet2D`               | Lightning module: loss, optimizer, metrics        |
| `ThioSTiffReader`           | MONAI transform for TIFF loading                 |
