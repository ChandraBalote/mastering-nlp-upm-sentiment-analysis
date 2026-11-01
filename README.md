# 🏆 #1 Place Solution — Mastering NLP: From Fundamentals to Advanced Techniques (UPM)

[![Kaggle Competition](https://img.shields.io/badge/Kaggle_Leaderboard-%231_Place_(0.949_Private_%7C_0.945_Public)-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/mastering-nlp-from-fundamentals-to-advanced-tech/leaderboard)
[![10-Fold CV](https://img.shields.io/badge/10--Fold_Stratified_CV-95.19%25_(0.95193)-success?style=for-the-badge)](#)
[![Institution](https://img.shields.io/badge/Course-Universidad_Polit%C3%A9cnica_de_Madrid_(UPM)-00509E?style=for-the-badge)](https://moodle.upm.es/en-abierto/)

**Author:** Luis Chandrasekhar Ballesteros López-Tello ([@ChandraBalote](https://github.com/ChandraBalote) | [Kaggle @chandrasaekhar](https://www.kaggle.com/chandrasaekhar))

---

## 📌 Overview

This repository contains the **#1 winning end-to-end solution** for the **Mastering NLP: From Fundamentals to Advanced Techniques (edition 3)** final competition hosted by the **Universidad Politécnica de Madrid (UPM)** on Kaggle.

The challenge consists of binary sentiment classification on a corpus of **4,913 training reviews** and **983 test reviews** (`Evaluation Metric: Classification Accuracy`), under strict academic rules prohibiting generative LLMs (such as GPT, Claude, or LLaMA).

---

## 📊 Performance Progression & Ablation Study

| Stage | Architecture / Method | 10-Fold Stratified OOF CV | Public Leaderboard | Private Leaderboard |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | Standard TF-IDF + Logistic Regression | `0.9165` | `0.9225` | `0.9054` |
| **Stage 1** | Multi-Granular TF-IDF (Word + Subword + Normalized) + VADER + Domain Lexicon + 150d Word2Vec + 7-Model Classical/GBDT Stack | `0.94736` | `0.9368` | `0.9490` |
| **Stage 2** | **3-View Pretrained Transformer Feature Extraction** (`twitter-roberta-base-sentiment` + `distilbert-sst2` over Head, Tail & Discourse-Pivot windows) + PCA(256) | `0.95015` | `0.9420` | `0.9490` |
| **Stage 3 (Final)** | **Nested 10-Fold CV Log-Odds Meta-Stacking & Threshold Calibration** across all 15 Base Learners | **`0.95193` (95.19%)** | **`0.9450` (#1)** | **`0.9490` (#1)** |

---

## 🧠 Architecture Highlights

1. **Multi-Granular Sparse & Lexical Representation (`260,055` Features)**:
   - **Word N-Grams (1–3)** with sublinear TF scaling capturing negation phrases (*"not worth"*, *"never disappoints"*).
   - **Character N-Grams (`char_wb`, 2–6)** capturing morphological roots, punctuation emphasis, and spelling variations.
   - **3-View Discourse-Aware Contextual Transformer Extraction**:
     - Long reviews often bury their true verdict at the end or immediately after contrast conjunctions (*"however"*, *"overall"*, *"but"*).
     - Each document is encoded across **3 complementary views**: **Full/Head** (first 256 tokens), **Tail** (last 192 tokens), and **Pivot** (window centered on the last contrast marker + concluding sentences).
     - Extracted using non-generative encoder backbones (`twitter-roberta-base-sentiment` and `distilbert-base-uncased-finetuned-sst-2-english`), yielding a `3,072`-dimensional representation compressed via SVD/PCA to `256` orthogonal components.
2. **15-Model Diverse Out-Of-Fold (OOF) Ensemble**:
   - Combines linear margin classifiers (`LogisticRegression`, `Calibrated LinearSVC`, `RidgeClassifier`, `SGD Modified Huber`), generative Naive Bayes (`ComplementNB +BernoulliNB`), Multi-Layer Perceptrons (`MLPClassifier`), and Gradient Boosted Decision Trees (`LightGBM`, `CatBoost`).
3. **Nested 10-Fold Log-Odds Calibration**:
   - All out-of-fold probability predictions are transformed into log-odds space $\text{logit}(p) = \ln(p / (1 - p))$ and blended via nested cross-validation to prevent threshold overfitting.

---

## 📂 Repository Structure

- [`Final_Assignment_Mastering_NLP_UPM.ipynb`](./Final_Assignment_Mastering_NLP_UPM.ipynb): Complete, self-contained Jupyter Notebook with EDA, mathematical formulation, feature engineering, 10-fold CV training, error analysis, and test inference.
- [`submission_kaggle_sentiment_analysis.csv`](./submission_kaggle_sentiment_analysis.csv): Winning Kaggle test predictions (`983` test rows).
