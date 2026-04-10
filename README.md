# 🌊 Query-Guided Multi-Resolution Transformer for Underwater Marine Litter Instance Segmentation

<p align="center">
  <img src="https://img.shields.io/badge/Task-Instance%20Segmentation-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Framework-Detectron2-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-Mask2Former-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

> Automatically detecting and segmenting underwater marine debris using deep learning — comparing CNN-based and Transformer-based instance segmentation architectures on a custom annotated dataset of 8,000 images.

---

## 📌 Problem Statement

Marine pollution is one of the most critical environmental challenges of our time. Manual identification of underwater litter is slow, expensive, and impractical at scale. This project builds an automated system to **detect, classify, and segment marine debris** at the pixel level using state-of-the-art deep learning models.

---

## 🏆 Results

| Model | Architecture | Segmentation mAP |
|-------|-------------|-----------------|
| Mask R-CNN | CNN Baseline | ~44.5 |
| **Mask2Former** | **Transformer (Proposed)** | **~45.99** |

> Mask2Former outperforms the CNN baseline, demonstrating the advantage of transformer-based global context modeling for complex underwater scenes.

---

## 🗂️ Dataset
-**Raw sea litter images collected from diverse sources, including SEACLEAR and TRASHCAN.**
- **~8,000 annotated underwater images**
- **Annotation format:** COCO segmentation (polygon masks)
- **Image resolution:** 640 × 640
- **Split:** Train / Validation / Test

### 10 Marine Litter Categories

| Class | Description |
|-------|-------------|
| `plastic_bag` | Plastic shopping bags |
| `plastic_bottle` | PET bottles |
| `glass_bottle` | Glass containers |
| `metal_scrap` | Metal debris |
| `fishing_net` | Fishing nets and gear |
| `rope` | Ropes and cords |
| `tyre` | Rubber tyres |
| `can` | Metal cans |
| `plastic_box` | Plastic containers/boxes |
| `other_trash` | Miscellaneous debris |

> **Note:** The dataset is not included in this repository due to size constraints (~8GB). Download instructions below.

---

## 🏗️ Architecture

```
Annotated Dataset (COCO format)
          ↓
   Detectron2 Framework
          ↓
  ┌───────────────────┐
  │   Mask R-CNN      │  ← CNN Baseline (ResNet backbone + FPN)
  └───────────────────┘
          ↓
  ┌───────────────────┐
  │   Mask2Former     │  ← Proposed Model (Transformer-based)
  └───────────────────┘
          ↓
   COCO Evaluation (mAP)
```

**Mask2Former** uses a transformer decoder with masked attention to predict instance masks, making it significantly better at capturing global context in complex underwater scenes compared to traditional CNN approaches.

---

## ⚙️ Setup

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended)
- PyTorch 1.10+

### Installation

```bash
# Clone the repository
git clone https://github.com/Lalitaditya-tickoo/marine-litter-detection.git
cd marine-litter-detection

# Install Detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# Install other dependencies
pip install -r requirements.txt
```

### Download Dataset

The dataset was annotated using Roboflow in COCO segmentation format.

```bash
# Download dataset using the provided script
python download_dataset.py
```

Or manually place your dataset in this structure:
```
dataset/
├── train/
│   ├── images/
│   └── _annotations.coco.json
├── valid/
│   ├── images/
│   └── _annotations.coco.json
└── test/
    ├── images/
    └── _annotations.coco.json
```

### Download Model Weights

Pre-trained model weights (~516 MB) are available via Google Drive:

> 🔗 **[Download mask2former_final.pth](#)** ← *(add your Google Drive link here)*

Place the downloaded file in the root `workspace/` directory.

---

## 🚀 Usage

### Train Mask R-CNN (Baseline)

```bash
python train_maskrcnn.py
```

### Train Mask2Former (Proposed)

```bash
cd Mask2Former
python train_net.py --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml
```

### Evaluate Model

```bash
python evaluate_model.py
```

### Run Inference on New Images

```bash
python test_model.py --input path/to/your/image.jpg
```

---

## 📊 Evaluation

Models were evaluated using standard **COCO evaluation metrics**:

- **mAP** (Mean Average Precision) @ IoU 0.50:0.95
- Evaluated on **801 unseen test images**

Each prediction outputs:
- Object category label
- Bounding box coordinates
- Pixel-level segmentation mask

---

## 🛠️ Tech Stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Detectron2-FF6F00?style=flat"/>
  <img src="https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/RunPod-GPU%20Cloud-purple?style=flat"/>
  <img src="https://img.shields.io/badge/Roboflow-Annotation-blue?style=flat"/>
</p>

---

## 📁 Repository Structure

```
workspace/
├── train_maskrcnn.py       # Mask R-CNN training script
├── evaluate_model.py       # COCO evaluation script
├── test_model.py           # Inference on new images
├── download_dataset.py     # Dataset download script
├── dataset/                # Dataset (not included, see above)
│   ├── train/
│   ├── valid/
│   └── test/
├── Mask2Former/            # Mask2Former framework
├── output/                 # Training outputs
├── predictions/            # Sample prediction images
└── README.md
```

---

## 👥 Team

| Name | University |
|------|-----------|
| Lalitaditya Tickoo | SRM University |
| Rian K Sinu | SRM University |
| Md Samar Aazmi | SRM University |
| Abhinav Singh Chauhan | SRM University |

---

## 🌍 Applications

This system can be used for:
- 🤖 Underwater robotics and AUVs
- 📡 Marine pollution monitoring systems
- 🌊 Automated ocean cleanup analysis
- 🏛️ Environmental protection agencies

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Mask2Former](https://github.com/facebookresearch/Mask2Former) by Facebook Research
- [Detectron2](https://github.com/facebookresearch/detectron2) by Facebook Research
- [Roboflow](https://roboflow.com) for dataset annotation tools
- [RunPod](https://runpod.io) for GPU cloud computing

---

<p align="center">
  <i>Built with ❤️ for marine conservation 🌊</i>
</p>
