# 🖌 Neural Style Transfer

Turn your images into artwork! 🎨 This PyTorch-based **Neural Style Transfer (NST)** project is inspired by the research paper:

**Image Style Transfer Using Convolutional Neural Networks**  
by *Leon A. Gatys*.

## 🚀 Getting Started
### 1️⃣ Install Dependencies
```bash
conda create --name nst_env python=3.8 -y
conda activate nst_env
pip install numpy matplotlib torch torchvision opencv-python
```

### 2️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/neural-style-transfer.git
cd neural-style-transfer
```

### 3️⃣ Run the Notebook
```bash
jupyter notebook
```
Or execute the Python script:
```bash
python neural_style_transfer.py
```

## 📁 Project Structure
1️⃣ **Libraries** – Importing essential packages.
2️⃣ **Models** – VGG16 & VGG19 (VGG19 performs better!).
3️⃣ **Data** – Loading & preprocessing images.
4️⃣ **Reconstruction** – Image recovery from features.
5️⃣ **NST** – The magic happens here! ✨

## 💻 Running Locally
Manually set up your NST parameters in `neural_style_transfer.py`:
```python
content_img = 'golden_gate.jpg'
style_img = 'city.jpeg'
model = 'vgg19'  # Options: 'vgg16', 'vgg19'
optimizer = 'lbfgs'  # Options: 'adam', 'lbfgs'
height = 400
content_weight = 1e5
style_weight = 3e4
tv_weight = 1e0
```

## 🌐 Google Colab Support
1️⃣ Upload the notebook to Colab.
2️⃣ Enable GPU: **Runtime → Change runtime type → GPU**.
3️⃣ Install dependencies:
4️⃣ Run the cells & enjoy your stylized images!

✨ *Unleash your creativity and make stunning artworks!* 🎭

