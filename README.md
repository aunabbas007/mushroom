# 🍄 Random Forest — Mushroom Classification

> Predict whether a mushroom is **Edible** or **Poisonous** using a Random Forest classifier.  
> Part of my step-by-step ML learning journey → Linear Regression → Logistic Regression → KNN → Decision Trees → **Random Forest** ✅

---

## 📁 Project Structure

```
random-forest-mushroom/
│
├── mushrooms.csv              ← Dataset (download from Kaggle)
├── train.py                   ← Train model + save as .pkl
├── model.py                   ← Load .pkl + predict new samples
├── mushroom_rf_model.pkl      ← Saved model (generated after running train.py)
├── confusion_matrix.png       ← Generated plot
├── feature_importance.png     ← Generated plot
└── README.md
```

---

## 📦 Dataset

| Property    | Detail |
|-------------|--------|
| Source      | [Kaggle — UCI Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification) |
| Rows        | 8,124 mushrooms |
| Features    | 22 categorical features (cap shape, color, odor, etc.) |
| Target      | `class` → `e` = edible, `p` = poisonous |
| Missing     | `?` in `stalk-root` column (handled in code) |

### Download
1. Go to the Kaggle link above
2. Download `mushrooms.csv`
3. Place it in the same folder as `train.py`

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Clone or create your project folder

```bash
mkdir random-forest-mushroom
cd random-forest-mushroom
# Place mushrooms.csv, train.py, model.py here
```

---

## 🚀 How to Run

### Step 1 — Train the model

```bash
python train.py
```

This will:
- Load and clean `mushrooms.csv`
- Label encode all 22 features
- Train a `RandomForestClassifier` with 100 trees
- Print accuracy + classification report
- Save plots: `confusion_matrix.png`, `feature_importance.png`
- Save the trained model to `mushroom_rf_model.pkl`

### Step 2 — Predict new mushrooms

```bash
python model.py
```

This will:
- Load `mushroom_rf_model.pkl`
- Run prediction on a sample mushroom
- Print: label (Edible/Poisonous) + confidence score
- Run a batch prediction on 2 mushrooms

---

## 🧠 How It Works

### What is Random Forest?

Random Forest is an **ensemble learning** method that builds many Decision Trees and combines their predictions.

```
Training Data
     │
     ├── Tree 1 (random rows + features) → Poisonous
     ├── Tree 2 (random rows + features) → Poisonous  
     ├── Tree 3 (random rows + features) → Edible
     ├── ...
     └── Tree 100                        → Poisonous
                                              ↓
                                   Majority Vote → POISONOUS ✅
```

### Two Sources of Randomness
| Technique | What it does |
|-----------|-------------|
| **Bootstrap Sampling** | Each tree trains on a random subset of rows (with replacement) |
| **Feature Sampling** | Each split considers only a random subset of features |

This forces diversity among trees → errors cancel out → better accuracy.

---

## 📊 Results

| Metric   | Score |
|----------|-------|
| Accuracy | ~100% |
| Precision| ~100% |
| Recall   | ~100% |
| F1-Score | ~100% |

> Random Forest achieves near-perfect accuracy on this dataset because `odor` is an extremely strong predictor of toxicity.

---

## 🔍 Key Concepts Learned

- **Ensemble Learning** — combining many weak models into one strong model
- **Bagging** — bootstrap aggregating (random row sampling per tree)
- **Feature Importance** — which features matter most across all trees
- **Label Encoding** — converting categorical text to integers
- **Pickle (.pkl)** — saving and loading trained ML models
- **predict_proba()** — getting confidence scores, not just labels

---

## 💾 About the .pkl File

The saved `mushroom_rf_model.pkl` contains **3 things**:

```python
{
    "model"   : RandomForestClassifier,   # Trained model
    "encoders": { col: LabelEncoder },    # One encoder per column
    "features": [ list of column names ]  # Feature order
}
```

> ⚠️ Always save encoders with the model! Without them, you can't correctly encode new raw inputs for prediction.

---

## 🗺️ ML Journey So Far

- [x] Linear Regression
- [x] Logistic Regression
- [x] K-Nearest Neighbors (KNN)
- [x] Decision Trees
- [x] **Random Forest** ← You are here
- [ ] Support Vector Machines (SVM)
- [ ] XGBoost / Gradient Boosting
- [ ] Neural Networks

---

## 👨‍💻 Author

Learning ML step by step to become an AI Engineer 🚀  
All projects pushed to GitHub as part of a structured learning path.