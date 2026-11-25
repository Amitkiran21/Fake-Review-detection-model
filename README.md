# Fake-Review-detection-model


An end-to-end Natural Language Processing (NLP) and Machine Learning pipeline that detects fraudulent product reviews (`0 = Real`, `1 = Fake`). The project compares a **from-scratch NumPy implementation of Multinomial Naive Bayes** against Scikit-Learn's **Logistic Regression** and **Linear Support Vector Machine (`LinearSVC`)**, complete with an interactive **Streamlit** web dashboard for live testing.

---

## Project Overview

Online marketplaces rely heavily on customer reviews, making automated detection of deceptive or computer-generated reviews essential. This repository implements a reproducible text-classification workflow:

1. **Data Cleaning & NLP Preprocessing** (Pandas & NLTK)
2. **Feature Engineering** via TF-IDF Vectorization (unigrams & bigrams)
3. **Model Training & Comparison** across three linear/probabilistic classifiers
4. **Visual Evaluation** (Confusion Matrices & F1-Score comparison charts)
5. **Interactive Inference** via a local Streamlit web application (`appw.py`)

---

## Dataset & Preprocessing Pipeline

The pipeline processes labeled product review data (binary classification: `0` for Real, `1` for Fake) through the following stages:

- **Structural Cleaning:** Removed exact duplicate entries and filtered out micro-reviews containing fewer than 4 words.
- **Text Normalization:** Lowercasing, punctuation and non-alphabetic character stripping, English stopword removal, and word lemmatization via NLTK.
- **Data Splitting:** 80/20 stratified train-test split performed **before** vectorization to prevent data leakage.
- **Vectorization:** `TfidfVectorizer(max_features=10000, ngram_range=(1, 2))` fit strictly on the training split.

---

## Models Compared

| Model | Implementation | Key Characteristics |
| :--- | :--- | :--- |
| **Multinomial Naive Bayes** | Custom from scratch (`NumPy`) | Uses log-probabilities to prevent numerical underflow and Laplace smoothing (`alpha=1.0`) directly on sparse TF-IDF matrices. |
| **Logistic Regression** | `sklearn.linear_model.LogisticRegression` | Discriminative linear classifier (`max_iter=1000`) modeling class probabilities. |
| **Linear SVM** | `sklearn.svm.LinearSVC` | Maximum-margin linear classifier well-suited for high-dimensional sparse text features. |

All models are evaluated on the unseen 20% test set using **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrices** (exported to `comparison_results.csv`, `f1_comparison.png`, and `confusion_matrices.png`).

---

## Project Structure

```text
fake_review/
├── appw.py                     # Interactive Streamlit dashboard
├── requirements.txt            # Python dependencies
├── tfidf.joblib                # Fitted TF-IDF vectorizer
├── models.joblib               # Trained classifiers (NB, LogReg, LinearSVC)
├── comparison_results.csv      # Test-set evaluation metrics
├── f1_comparison.png           # F1-Score comparison bar chart
├── confusion_matrices.png      # Side-by-side confusion matrix heatmaps
└── .gitignore                  # Excludes virtual environments and caches
```

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/Amitkiran21/Fake-Review-detection-model.git](https://github.com/Amitkiran21/Fake-Review-detection-model.git)
cd Fake-Review-detection-model
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**Linux / macOS / WSL:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Streamlit App

Make sure your virtual environment is active and that `tfidf.joblib`, `models.joblib`, and `comparison_results.csv` are present in the root directory, then run:

```bash
streamlit run appw.py
```

The dashboard will open automatically in your browser (default: `http://localhost:8501`), allowing you to:
- View the side-by-side **Model Performance Metrics** table.
- Inspect the **F1-Score** and **Confusion Matrix** visualizations.
- Paste any custom product review to see live predictions from all three models simultaneously.
