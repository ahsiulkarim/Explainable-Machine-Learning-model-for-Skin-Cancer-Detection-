# CSE475 — Explainable ML Model for Skin-Cancer Detection
**Group-2 | Section 06 | Summer 2026 | East West University**

Submitted to: **Dr. Raihan Ul Islam**, Associate Professor, CSE, EWU



---

## Project Overview

Multi-class skin-lesion classification on **HAM10000** (10,015 dermoscopic images, 7 classes) using:
- 4 deep-learning baselines (ResNet50, EfficientNet-B2, ViT-B/16, Swin-Tiny)
- 1 GNN pipeline (ResNet50 spatial features → GATv2 graph classifier)
- XAI: attention rollout (ViT), Grad-CAM (Swin), SHAP + LIME (GNN)

**Key design choice:** Lesion-level data splitting throughout — no same-lesion images across train/val/test splits.

---

## Dataset

| Class | Label | Images |
|---|---|---|
| Actinic keratoses | `akiec` | 327 |
| Basal cell carcinoma | `bcc` | 514 |
| Benign keratosis | `bkl` | 1,099 |
| Dermatofibroma | `df` | 115 |
| Melanoma | `mel` | 1,113 |
| Melanocytic nevi | `nv` | 6,705 |
| Vascular lesions | `vasc` | 142 |

**Highly imbalanced** — `nv` dominates (~67%). Macro F1 used as primary metric.

Source: [Skin Cancer MNIST: HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

---

## Repository Structure

```
.
├── README.md
├── Group02_HAM10000_task1_EDA.ipynb              # Task 1: Exploratory Data Analysis
├── Group02_HAM10000_task2_baselines_resnet50.ipynb       # Baseline 1: ResNet50
├── Group02_HAM10000_task2_baselines_efficientnet-b2.ipynb # Baseline 2: EfficientNet-B2
├── Group02_HAM10000_task2_baselines_vit-b16-5-fold.ipynb  # Baseline 3: ViT-B/16 (5-fold)
├── Group02_HAM10000_task2_baselines_swin-tiny-5-fold.ipynb # Baseline 4: Swin-Tiny (5-fold)
├── group2-ham10000-task2-gnn-pipeline.ipynb      # Task 2: GNN pipeline (reference/ablation start)
├── group2-ham10000-task3-gnn-pipeline.ipynb      # Task 3: GNN ablation + XAI (final GNN)
├── Group02_HAM10000_Report.pdf                   # Full project report
├── Group-2_HAM10000_Task_1_report.pdf            # Task 1 EDA report
└── Group-2_HAM10000_Related_Work_table.pdf       # Literature review table
```

---

## Notebooks

### Task 1 — EDA (`task1_EDA.ipynb`)
- Load metadata CSV, check missing values (Age column has some)
- Class distribution, gender split, age histogram + boxplot
- Lesion localization (back most common)
- Diagnosis type breakdown (histopathology dominant)
- Sample image visualization, image size check

### Task 2 — Baselines

#### ResNet50 (`task2_baselines_resnet50.ipynb`)
- `timm` ResNet50, ImageNet pretrained, 23.5M params
- Input: 224×224 | Batch: 16 | Epochs: 50
- 5-epoch head warmup → full fine-tune (backbone LR 1e-5, head LR 1e-4)
- Square-root inverse class weights + label smoothing 0.10
- Split: seed-42 lesion holdout — 7,002 train / 1,519 val / **1,494 test**
- Best checkpoint by val Macro F1

#### EfficientNet-B2 (`task2_baselines_efficientnet-b2.ipynb`)
- `timm` EfficientNet-B2, 7.7M params
- Input: 260×260 | same split/training recipe as ResNet50
- Split: seed-42 — **1,494 test** (same as ResNet50, directly comparable)

#### ViT-B/16 (`task2_baselines_vit-b16-5-fold.ipynb`)
- `timm` `vit_base_patch16_224.augreg_in21k_ft_in1k`, 85.8M params
- Input: 224×224 | 5-fold StratifiedGroupKFold (lesion-safe)
- Early stopping: patience 8, starts epoch 15
- Split: seed-171 — 8,509 dev / **1,506 final test** (untouched)
- Final prediction: **mean probability ensemble** of 5 fold models

#### Swin-Tiny (`task2_baselines_swin-tiny-5-fold.ipynb`)
- `timm` `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k`, 27.5M params
- Same 5-fold / seed-171 / ensemble protocol as ViT-B/16
- Split: seed-171 — **1,506 final test** (same as ViT, directly comparable)

### Task 3 — GNN Pipeline (`task3-gnn-pipeline.ipynb`)
- Fixed ResNet50 encoder: extracts 512×7×7 spatial feature maps from `layer4[-1].bn2`
- 49 spatial nodes per image; 514-dim node features (512 deep + 2 coords, L2-normalized)
- Grid4 graph: up/down/left/right + self-loops (217 directed edges)
- Sequential ablation: layers, hidden size, dropout, BatchNorm, LR, scheduler, optimizer, graph type, edge weights, GNN variant, imbalance loss, patience
- **Winner: GATv2 — 2 layers, hidden 128, 4 heads, dropout 0.40, cosine LR, Adam**
- Split: seed-171 — 7,007 train / 1,502 val / **1,506 test** (same final test as ViT + Swin)
- XAI: node-masking SHAP + LIME on 49 spatial nodes → mapped to 7×7 importance grid

---

## Results

### Same test set (seed-171, n=1,506) — direct comparison

| Model | Accuracy | Macro F1 | Macro AUC |
|---|---|---|---|
| **ViT-B/16** (5-fold ensemble) | **86.52%** | **0.7777** | 0.9772 |
| Swin-Tiny (5-fold ensemble) | 85.46% | 0.7289 | 0.9729 |
| Final GNN (GATv2) | 75.43% | 0.5453 | 0.8806 |

### Different test set (seed-42, n=1,494) — reference only

| Model | Accuracy | Macro F1 |
|---|---|---|
| EfficientNet-B2 | 80.52% | 0.6152 |
| ResNet50 | 78.51% | 0.5887 |

> ⚠️ Cross-group comparison invalid — different test splits.

### Class-wise F1 (seed-171 models only)

| Class | ViT F1 | Swin F1 | GNN F1 |
|---|---|---|---|
| akiec | 0.6742 | 0.4857 | 0.3478 |
| bcc | 0.8047 | 0.7927 | 0.5233 |
| bkl | 0.7864 | 0.7713 | 0.5681 |
| df | 0.6452 | 0.6061 | 0.3684 |
| mel | 0.6829 | 0.6850 | 0.5366 |
| nv | 0.9277 | 0.9281 | 0.8780 |
| vasc | 0.9231 | 0.8333 | 0.5946 |

GNN 95% bootstrap CI (lesion-level): accuracy 0.7293–0.7790, Macro F1 0.4846–0.5966

---

## Explainability

| Model | Method | Output |
|---|---|---|
| ViT-B/16 | Attention rollout | Token-level attention heatmap |
| Swin-Tiny | Grad-CAM | Class-specific activation maps |
| GNN | Node-masking SHAP + LIME | 7×7 spatial node importance grid |

Note: XAI outputs are interpretation aids, not clinically validated segmentations.

---

## Setup & Dependencies

All notebooks run on **Kaggle** (Python 3, GPU enabled).

```bash
# Core (pre-installed on Kaggle)
numpy pandas matplotlib seaborn PIL scikit-learn torch torchvision

# Model hub
pip install timm

# GNN notebooks only
pip install torch-geometric shap lime
```

**Data path** (Kaggle dataset: `kmader/skin-cancer-mnist-ham10000`):
```
/kaggle/input/skin-cancer-mnist-ham10000/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

---

## Key Design Decisions

1. **Lesion-level splits** — `lesion_id` used as group key in all splits. Prevents same-lesion train/test leakage.
2. **Macro F1 as primary metric** — dataset heavily imbalanced toward `nv`. Accuracy alone misleads.
3. **Two-stage fine-tuning** — head warmup (5 epochs) then full backbone unfreeze, with separate LRs.
4. **Early stopping** — ViT + Swin use patience-8 early stop from epoch 15. ResNet50 + EfficientNet-B2 complete full 50 epochs.
5. **Ensemble** — ViT + Swin use 5-fold mean probability ensemble. GNN is single model.
6. **GNN design** — 7×7 = 49 nodes from ResNet spatial map. Fixed CNN encoder (not end-to-end). Grid4 connectivity.
7. **ROC missing for ResNet50/EfficientNet-B2** — probability format error in notebooks; AUC stored as `None`. Not fabricated.

---

## Literature Baseline Comparison

| Paper | Model | Dataset | Accuracy |
|---|---|---|---|
| Tahir et al. 2023 | Custom CNN | HAM10000+ISIC2020 | 94.17% |
| Manole et al. 2024 | EfficientNet-B3 | ISIC2019 | ~96% (4-class) |
| Dagnaw et al. 2024 | ResNet50 | ISIC subset | 88.8% |
| Dogga et al. 2026 | ViT+GNN+RAA | HAM10000 | 88.7% |
| **This work** | **ViT-B/16 (ens.)** | **HAM10000** | **86.52%** |

---

## Limitations

- ResNet50 + EfficientNet-B2 on different test split → not directly comparable to ViT/Swin/GNN
- GNN = single model vs. ViT/Swin = 5-model ensembles
- GNN CNN encoder fixed (not end-to-end trained with graph objective)
- XAI not validated against dermatologist-drawn lesion masks
- Bangladesh-specific clinical dataset (SkinDisNet) not used — all experiments on HAM10000

---

## Future Work

- End-to-end CNN-GNN joint training
- Semantic/adaptive graph construction beyond fixed grid
- Attention-based graph pooling
- Cross-validated GNN evaluation (5-fold)
- External clinical validation
- Bangladesh-relevant dataset integration (SkinDisNet)

---

## Team

| No. | Name               | Student ID     |
|-----|--------------------|----------------|
| 1   | Md. Rahat Khan     | 2021-1-60-090  |
| 2   | Jannatul Alam Shifa| 2022-2-60-147  |
| 3   | Md Ahsiul Karim    | 2022-3-60-074  |
| 4   | Fahim Hossain      | 2022-2-60-142  |
