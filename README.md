# Dog Breed Prediction (ResNet50 + PyTorch + Streamlit)

A deep learning project that classifies **120 dog breeds** using **transfer learning with ResNet50**.  
Trained on the [Kaggle Dog Breed Identification dataset](https://www.kaggle.com/competitions/dog-breed-identification).  
Built and deployed with **PyTorch**, **JupyterLab**, and **Streamlit**.

---

## Features
- **Transfer Learning (ResNet50)** pre-trained on ImageNet.
- **Custom PyTorch Dataset** built from Kaggle’s `labels.csv`.
- **Train / Validation Split** with accuracy and loss tracking.
- **Feature Visualization** using **PCA** and **t-SNE**.
- **Streamlit Web App** for real-time image predictions.
- **Timestamped model saving** for experiment tracking (`models/run_YYYYMMDD_HHMMSS/`).

---

## Project Structure

Dog Breed/

├─ app/

│ ├─ app.py # Streamlit inference app

│ ├─ assets/

│ │ └─ class_names.json # (optional) breed labels

│

├─ data/

│ ├─ raw/ # Original Kaggle dataset

│ │ ├─ train/

│ │ ├─ test/

│ │ └─ labels.csv

│ ├─ processed/ # (optional) organized data

│

├─ models/

│ ├─ run_20251109_153000/ # auto-created per training run

│ │ └─ resnet50_dogbreed.pth

│

├─ notebooks/

│ └─ 01_dog_breed_prediction.ipynb

│

├─ requirements.txt

└─ README.md


---

## Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/Trickerqty/dog-breed-prediction.git
```
```bash
cd dog-breed-prediction
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate     # (Windows)
```
# or
```bash
source .venv/bin/activate  # (Mac/Linux)
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download from Kaggle: (https://www.kaggle.com/competitions/dog-breed-identification)

Extract into:

data/raw/

  ├─ train/
  
  ├─ test/
  
  └─ labels.csv


---
## Model Training

```bash
jupyter lab
```

---
## Streamlit App

```bash
streamlit run app/app.py
```
