# 📐 GridPE: Grid Cell-Inspired Positional Encoding for Vision and Point Cloud Transformers

This is the official implementation of **GridPE**, a biologically inspired positional encoding method based on periodic spatial codes observed in grid cells. GridPE approximates shift-invariant attention kernels using structured Fourier embeddings, and is applicable to both 2D vision transformers and 3D point cloud transformers.

---
## GridPE-Attention
The core implementation of GridPE Attention is available in [`gridAttn.py`](/PCT/gridAttn.py), making it easy to integrate into other repositories.

## 📁 Project Structure

```
GridPE/
├── PCT/               # Point cloud transformer (ModelNet40)
├── VIT/               # Vision transformer (ImageNet100)
├── Simulation/        # Grid cell simulation notebooks
├── requirements.txt   # Environment setup
└── README.md
```

---

## 📦 Installation

Install required packages via pip:
```bash
pip install -r requirements.txt

# Install ops for point cloud processing
pip install PCT/pointnet2_ops_lib/.
```

---

## 📥 Dataset Preparation

- **ImageNet100**  
  Handled via `VIT/deit/huggingface-image.py` using HuggingFace datasets.

- **ModelNet40**  
  Automatically downloaded using `PCT/data.py` via the `download()` function.

---

## 🏋️ Training

- **2D Vision Transformer**
```bash
bash VIT/deit/run_experiment.sh
```

- **3D Point Cloud Transformer**
```bash
bash PCT/run_train.sh
```

---

## 📊 Evaluation

- **2D Evaluation (ViT)**
```bash
bash VIT/deit/run_evaluation.sh
```

- **3D Evaluation (PCT)**
```bash
bash PCT/run_evaluation.sh
```

---

## 🧠 Pretrained Models

- **PCT (ModelNet40)**  
  Pretrained weights:
  ```
  PCT/checkpoints/train_pct*/models/model.t7
  ```

- **ViT (ImageNet100)**  
  Download from:  
  [Google Drive](https://drive.google.com/drive/folders/1amSClZKBXm1ewoNcDWHak2CXSbp9SAKe?usp=sharing)

---

## 📈 Results & Analysis

### 🖼️ 2D Image Classification (ViT)
- `VIT/accuracy_1&5.ipynb`: Main results
- `VIT/accuracy_1&5_abligation.ipynb`: Ablation study
- `VIT/2d_attn/` and `attn_dist.ipynb`: Attention distance/entropy visualization
- `VIT/grid_distance.ipynb`: Spatial correlation analysis

### 🧊 3D Point Cloud Classification (PCT)
- `PCT/plot.py`: Accuracy plotting
- `PCT/attn_distance.ipynb`: Attention statistics
- `PCT/3d_attn/`: 3D spatial attention maps