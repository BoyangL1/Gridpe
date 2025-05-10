# GridPE: Grid Cell-Inspired Positional Encoding for Vision and Point Cloud Transformers

This repository implements **GridPE**, a biologically inspired positional embedding method based on periodic spatial codes from grid cells. GridPE approximates shift-invariant attention kernels using structured Fourier embeddings and is applicable to both 2D vision and 3D point cloud transformers.

## 📂 Project Structure

Gridpe/
├── VIT/               # Vision Transformer (ViT) experiments and evaluation
├── PCT/               # Point Cloud Transformer models and evaluation
├── Simulation/        # Grid cell simulation and frequency analysis
├── README.md
├── requirements.txt

### 🔹 `VIT/`
- Contains ViT model variants with different positional embeddings: Learnable, RoPE, and GridPE.
- Includes attention distance/entropy analysis notebooks and visualization scripts.

### 🔹 `PCT/`
- Implements GridPE-based Point Cloud Transformer variants.
- Includes training/testing scripts, model definitions, and evaluation tools.

### 🔹 `Simulation/`
- Jupyter notebooks for simulating periodic grid cell responses and studying multi-scale frequency effects.

### 🔹 `deit/` *(inside `VIT/`)*
- Pretrained DeiT backbones and positional embedding modifications.

### 🔹 `GridAttn.py`
- Core implementation of the GridPE attention mechanism.

## 📊 Evaluation

Evaluation scripts and figures are available in:
- `VIT/2d_attn`: Visualization of attention statistics across image sizes.
- `PCT/3d_attn`: Similar analyses on point cloud models.

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```


📬 Contact

For questions or collaborations, feel free to reach out via [liboyang0209@gmail.com].