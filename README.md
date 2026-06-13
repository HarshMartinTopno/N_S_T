# Neural Style Transfer with PyTorch

Implementation of Neural Style Transfer (NST) based on the paper [*Image Style Transfer Using Convolutional Neural Networks*](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) by Leon A. Gatys et al. (CVPR 2016).

---

## Overview

This project transfers the artistic style of one image onto the content of another by optimizing a noise image using a pretrained VGG network as a fixed feature extractor. The optimization minimizes a weighted combination of content loss, style loss (via Gram matrices), and total variation loss.

---

## Results

| Content | Style | Output |
|---------|-------|--------|
| golden_gate.jpg | city.jpeg | *(add your output image here)* |

> Add your before/after images to the `data/output-images/` folder and link them above.

---

## How It Works

### Feature Extraction
Three VGG architectures are supported, each exposing different layer combinations for style and content representation:

- **VGG16** — uses `relu1_2`, `relu2_2`, `relu3_3`, `relu4_3` for style; `relu2_2` for content
- **VGG16 Experimental** — exposes all intermediate layers for flexible experimentation
- **VGG19** — follows the original Gatys paper: `conv1_1` through `conv5_1` for style, `conv4_2` for content

All models are loaded with pretrained ImageNet weights and frozen (`requires_grad=False`) — only the output pixel values are optimized.

### Loss Function

```
Total Loss = content_weight × content_loss
           + style_weight   × style_loss
           + tv_weight      × tv_loss
```

- **Content Loss** — MSE between feature maps of the generated and content image at the content layer
- **Style Loss** — MSE between Gram matrices of feature maps across style layers; Gram matrix captures texture/style by computing feature correlations
- **TV Loss** — Total variation regularization to reduce high-frequency noise in the output

### Optimization

Two optimizers are supported:

| Optimizer | Iterations | Notes |
|-----------|-----------|-------|
| L-BFGS | 1000 | Second-order; converges faster, matches original paper |
| Adam | 3000 | First-order; more stable on CPU |

L-BFGS with `strong_wolfe` line search is used by default, matching the original paper's approach.

### Initialization Methods

| Method | Description |
|--------|-------------|
| `content` | Start from content image (faster convergence) |
| `random` | Start from Gaussian noise (more stylistic freedom) |
| `style` | Start from resized style image |

---

## Configuration

Parameters used in this implementation:

```python
content_img_name  = 'golden_gate.jpg'
style_img_name    = 'city.jpeg'
height            = 400          # Output image height (width scaled proportionally)

content_weight    = 1e5          # Higher → preserve more content structure
style_weight      = 3e4          # Higher → stronger style transfer
tv_weight         = 1e0          # Smoothness regularization

optimizer         = 'lbfgs'
model             = 'vgg19'
init_method       = 'content'
```

**Why these weights?** `content_weight` is set ~3x higher than `style_weight` to preserve recognizable content structure while still applying strong stylization. `tv_weight` at `1e0` adds just enough smoothing without blurring detail.

---

## Project Structure

```
neural-style-transfer/
│
├── NST.ipynb                  # Main notebook (Google Colab compatible)
│
└── data/
    ├── content-images/        # Input content images
    ├── style-images/          # Input style images
    └── output-images/         # Generated outputs (created automatically)
```

---

## Setup & Usage

### Run on Google Colab (Recommended)
Open `NST.ipynb` in Colab. GPU runtime recommended — go to `Runtime > Change runtime type > GPU`.

### Run Locally

```bash
git clone https://github.com/HarshMartinTopno/N_S_T.git
cd N_S_T
pip install torch torchvision opencv-python matplotlib numpy
```

Add your images:
```
data/content-images/your_content.jpg
data/style-images/your_style.jpg
```

Update the filenames in the config section of the notebook and run all cells.

---

## Dependencies

```
torch
torchvision
opencv-python (cv2)
numpy
matplotlib
```

---

## References

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf). CVPR 2016.
- Implementation insights from [Aleksa Gordić's NST tutorial series](https://www.youtube.com/playlist?list=PLBoQnSflObcmbfshq9oNs41vODgXG-608)
