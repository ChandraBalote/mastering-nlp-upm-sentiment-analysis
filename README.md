# Mastering NLP: From Fundamentals to Advanced Techniques (UPM) — 1st Place Solution

**Author:** Luis Chandrasekhar Ballesteros López-Tello ([@ChandraBalote](https://github.com/ChandraBalote) · [Kaggle @chandrasaekhar](https://www.kaggle.com/chandrasaekhar))  
**Competition:** [UPM Mastering NLP Final Challenge (Kaggle)](https://www.kaggle.com/competitions/mastering-nlp-from-fundamentals-to-advanced-tech/leaderboard)

---

## Overview

This repository contains my end-to-end solution for the **Mastering NLP: From Fundamentals to Advanced Techniques (edition 3)** final evaluation hosted by **Universidad Politécnica de Madrid (UPM)** on Kaggle.

The task is binary sentiment classification over **4,913 training reviews** and **983 test reviews** evaluated by classification accuracy. Course rules prohibited generative large language models (such as GPT, Claude, or LLaMA), requiring classical NLP representations and encoder-only models trained under strict cross-validation.

---

## Validation & Leaderboard Progression

| Stage | Method | 10-Fold Stratified CV | Public LB | Private LB |
| :--- | :--- | :---: | :---: | :---: |
| Baseline | Word TF-IDF + Logistic Regression | `0.91650` | `0.9225` | `0.9054` |
| Stage 1 | Word & character TF-IDF + VADER + domain lexicon + 150d Word2Vec + 7-model linear/tree blend | `0.94736` | `0.9368` | `0.9490` |
| Stage 2 | Three-window encoder representations (`twitter-roberta-base-sentiment` + `distilbert-sst2`) + SVD/PCA (256d) | `0.95015` | `0.9420` | `0.9490` |
| Stage 3 | Nested 10-fold log-odds stacking & threshold calibration across 15 base models | **`0.95193`** | **`0.9450` (1st)** | **`0.9490` (1st)** |

---

## Technical Approach

1. **Sparse Lexical & Morphological Features (`260,055` columns)**
   - Sublinear word n-grams (1–3) to capture local negation and sentiment modifiers (*"not worth"*, *"never disappoints"*).
   - Character n-grams (`char_wb`, 2–6) to handle punctuation emphasis, morphological stems, and informal spelling variations.

2. **Discourse-Window Encoder Extraction**
   - Long reviews frequently state nuanced context early on and place the actual conclusion at the end or right after a contrast marker (*"however"*, *"overall"*, *"but"*).
   - Instead of truncating every review to the first 512 tokens, each text is pooled across three views: **Head** (opening 256 tokens), **Tail** (closing 192 tokens), and **Pivot** (window around the final contrast conjunction).
   - Embeddings from `twitter-roberta-base-sentiment` and `distilbert-base-uncased-finetuned-sst-2-english` are projected via truncated SVD/PCA to 256 orthogonal components inside cross-validation.

3. **Out-of-Fold Log-Odds Ensembling**
   - Fifteen diverse models (`LogisticRegression`, calibrated `LinearSVC`, `RidgeClassifier`, `SGDClassifier` with modified Huber loss, `ComplementNB`, `BernoulliNB`, `MLPClassifier`, `LightGBM`, and `CatBoost`) are trained on identical 10-fold stratified splits.
   - Out-of-fold probabilities are mapped to log-odds $\text{logit}(p) = \ln(p / (1 - p))$ and combined with nested cross-validation so neither blend weights nor classification thresholds overfit out-of-fold noise.

---

## Files

- [`Final_Assignment_Mastering_NLP_UPM.ipynb`](./Final_Assignment_Mastering_NLP_UPM.ipynb): Complete notebook covering exploratory analysis, feature extraction, 10-fold cross-validation, ablation checks, and test inference.
- [`submission_kaggle_sentiment_analysis.csv`](./submission_kaggle_sentiment_analysis.csv): Final test predictions (`983` rows).
