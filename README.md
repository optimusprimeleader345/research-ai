# I-JEPA — Image-based Joint-Embedding Predictive Architecture

> Official PyTorch codebase published @ **CVPR 2023**
> [\[arXiv\]](https://arxiv.org/pdf/2301.08243.pdf) [\[Blog\]](https://ai.facebook.com/blog/yann-lecun-ai-model-i-jepa/) [\[JEPAs\]](https://ai.facebook.com/blog/yann-lecun-advances-in-ai-research/)

---

## 🧠 What is I-JEPA?

**I-JEPA** is a self-supervised learning method that learns rich semantic image representations by **predicting representations of image regions from other regions of the same image** — entirely in latent space, with no hand-crafted augmentations or pixel-level reconstruction.

Key advantages:
- ✅ No hand-crafted data augmentation
- ✅ No pixel-level reconstruction (less low-level bias)
- ✅ Computationally efficient (single image view)
- ✅ Strong off-the-shelf semantic features

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        I-JEPA Training Loop                         │
│                                                                     │
│   Input Image                                                       │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐    Context Patches        ┌──────────────────┐    │
│  │  Patch      │ ──────────────────────▶   │  Context Encoder │    │
│  │  Tokenizer  │                           │    (ViT)         │    │
│  │  (ViT)      │    Target Patches         └────────┬─────────┘    │
│  └─────────────┘ ──────────────────────▶            │              │
│                               │            Context Representation   │
│                               │                     │              │
│                    ┌──────────▼──────────┐          ▼              │
│                    │   Target Encoder    │  ┌────────────────┐     │
│                    │   (ViT, EMA copy)   │  │   Predictor    │     │
│                    └──────────┬──────────┘  │   (Narrow ViT) │     │
│                               │             └───────┬────────┘     │
│                     Target Representations          │              │
│                               │                     │              │
│                               └──────────┐  Predicted              │
│                                          │  Representations        │
│                                          ▼          │              │
│                                   ┌──────────────┐  │              │
│                                   │   L2 Loss    │◀─┘              │
│                                   │  (Smooth L1) │                 │
│                                   └──────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Training Flow Diagram

```
                        ┌──────────────────┐
                        │   Input Image    │
                        └────────┬─────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
   ┌──────────────────┐                  ┌──────────────────┐
   │  Context Blocks  │                  │  Target Blocks   │
   │  (visible patches│                  │  (masked patches │
   │   to encoder)    │                  │   to predict)    │
   └────────┬─────────┘                  └────────┬─────────┘
            │                                     │
            ▼                                     ▼
   ┌──────────────────┐                  ┌──────────────────┐
   │ Context Encoder  │                  │  Target Encoder  │
   │   (ViT — full    │                  │  (ViT — EMA/     │
   │    gradient)     │                  │   no gradient)   │
   └────────┬─────────┘                  └────────┬─────────┘
            │                                     │
            │   Context                Target     │
            │   Representation         Repr.      │
            └──────────┬──────────────────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │    Predictor     │◀─── Target Block Positional Mask
              │  (Narrow ViT)    │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │   Predicted      │
              │  Representations │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │    L2 / Smooth   │
              │   L1 Loss        │
              └──────────────────┘
                   (Backprop only through Context Encoder + Predictor)
                   (Target Encoder updated via EMA)
```

---

## 📦 Component Breakdown

| Component | Role | Architecture |
|---|---|---|
| **Context Encoder** | Encodes visible (context) patches | Vision Transformer (ViT) — full gradient |
| **Target Encoder** | Encodes target patches (ground truth) | ViT — updated via **EMA** (no gradient) |
| **Predictor** | Predicts target representations from context | Narrow ViT with positional tokens |
| **Mask Collator** | Generates context/target block masks | Multi-block random masking strategy |

---

## 🗂️ Masking Strategy

```
 ┌─────────────────────────────────────┐
 │            Input Image              │
 │  ┌───┬───┬───┬───┬───┬───┬───┬───┐ │
 │  │ C │ C │ T │ T │ C │ T │ T │ C │ │  C = Context patch
 │  ├───┼───┼───┼───┼───┼───┼───┼───┤ │  T = Target patch
 │  │ C │ T │ T │ T │ T │ T │ C │ C │ │
 │  ├───┼───┼───┼───┼───┼───┼───┼───┤ │
 │  │ T │ T │ C │ C │ C │ T │ T │ T │ │
 │  ├───┼───┼───┼───┼───┼───┼───┼───┤ │
 │  │ C │ C │ C │ T │ T │ T │ C │ C │ │
 │  └───┴───┴───┴───┴───┴───┴───┴───┘ │
 └─────────────────────────────────────┘
   Context patches → fed to Context Encoder
   Target patches  → fed to Target Encoder (EMA)
                   → Predictor must predict their representations
```

---

## 🔁 EMA (Exponential Moving Average) Update

The **target encoder** is never directly trained. It's updated as a moving average of the context encoder:

```
θ_target ← m × θ_target + (1 - m) × θ_context

where m (momentum) starts near 1.0 (e.g., 0.996) and gradually increases to 1.0
```

This ensures stable target representations throughout training.

---

## 📁 Code Structure

```
.
├── configs/                        # Experiment YAML config files
│   ├── in1k_vith14_ep300.yaml      #   ViT-H/14, ImageNet-1K, 300 epochs
│   ├── in1k_vith16-448_ep300.yaml  #   ViT-H/16, 448px, ImageNet-1K
│   ├── in22k_vith14_ep66.yaml      #   ViT-H/14, ImageNet-22K
│   └── in22k_vitg16_ep44.yaml      #   ViT-g/16, ImageNet-22K
│
├── src/                            # Core source package
│   ├── train.py                    #   Main I-JEPA training loop
│   ├── helper.py                   #   Model init, optimizer, checkpoint helpers
│   ├── transforms.py               #   Pre-training data transforms
│   ├── datasets/
│   │   └── imagenet1k.py           #   ImageNet-1K data loader
│   ├── models/
│   │   └── vision_transformer.py   #   ViT encoder & predictor definitions
│   ├── masks/
│   │   ├── multiblock.py           #   Multi-block masking collator
│   │   ├── random.py               #   Random masking
│   │   ├── default.py              #   Default mask strategy
│   │   └── utils.py                #   Masking utilities
│   └── utils/
│       ├── distributed.py          #   Distributed training helpers
│       ├── logging.py              #   Logging utilities
│       ├── schedulers.py           #   LR and WD schedulers
│       └── tensors.py              #   Tensor utilities
│
├── main.py                         # Entry point: local single/multi-GPU
└── main_distributed.py             # Entry point: SLURM distributed training
```

---

## 🚀 Launching I-JEPA Pretraining

### Single / Multi-GPU (Local)

```bash
python main.py \
  --fname configs/in1k_vith14_ep300.yaml \
  --devices cuda:0 cuda:1 cuda:2
```

### Distributed (SLURM Cluster)

```bash
python main_distributed.py \
  --fname configs/in1k_vith14_ep300.yaml \
  --folder $path_to_save_submitit_logs \
  --partition $slurm_partition \
  --nodes 2 --tasks-per-node 8 \
  --time 1000
```

> **Note:** ViT-H/14 config requires 16× A100 80G GPUs with an effective batch size of 2048 to reproduce published results.

---

## 🤖 Pretrained Models

| Architecture | Patch Size | Resolution | Epochs | Dataset | Download |
|---|---|---|---|---|---|
| ViT-H | 14×14 | 224×224 | 300 | ImageNet-1K | [checkpoint](https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar) · [logs](https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-logs-rank.0.csv) · [config](configs/in1k_vith14_ep300.yaml) |
| ViT-H | 16×16 | 448×448 | 300 | ImageNet-1K | [checkpoint](https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar) · [logs](https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16.448-logs-rank.0.csv) · [config](configs/in1k_vith16-448_ep300.yaml) |
| ViT-H | 14×14 | 224×224 | 66 | ImageNet-22K | [checkpoint](https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar) · [logs](https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-logs-rank.0.csv) · [config](configs/in22k_vith14_ep66.yaml) |
| ViT-g | 16×16 | 224×224 | 44 | ImageNet-22K | [checkpoint](https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar) · [logs](https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-logs-rank.0.csv) · [config](configs/in22k_vitg16_ep44.yaml) |

---

## 📊 Evaluation Results

I-JEPA achieves strong performance on linear evaluation and few-shot learning benchmarks, outperforming many pixel-reconstruction-based SSL methods while being significantly more compute-efficient.

| Method | 1% Labels (Top-1) | Linear Eval (Top-1) | Type |
|---|---|---|---|
| MAE | 51.0 | 68.0 | Pixel Reconstruction |
| SimCLR | 48.3 | 69.3 | Contrastive |
| DINO | 57.5 | 77.3 | Self-Distillation |
| **I-JEPA (ViT-H/14)** | **72.9** | **81.9** | **Latent Prediction** |

---

## ⚙️ Requirements

```
Python     >= 3.8
PyTorch    >= 2.0
torchvision
pyyaml
numpy
opencv-python
submitit       # for SLURM distributed training
```

Install dependencies:
```bash
pip install torch torchvision pyyaml numpy opencv-python submitit
```

---

## 🔬 How It Compares to Other SSL Methods

```
 Pixel Space Methods          Latent Space Methods
 (MAE, BEiT, etc.)           (I-JEPA, etc.)
 ┌───────────────┐            ┌───────────────┐
 │  Reconstruct  │            │  Predict in   │
 │  raw pixels   │            │  feature/     │
 │  of masked    │            │  latent space │
 │  patches      │            │  of masked    │
 └───────┬───────┘            │  patches      │
         │                    └───────┬───────┘
         ▼                            ▼
  Learns low-level           Learns high-level
  textures & edges           semantics & structure
                             (better downstream!)
```

---

## 📄 License

See [LICENSE](./LICENSE) for details.

---

## 📖 Citation

```bibtex
@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023}
}
```
