# Translationese Detection via Linguistic Feature Engineering and Machine Learning

This project presents a linguistically grounded, computational pipeline for identifying *translationese*—the subtle but systematic linguistic traits that distinguish translated texts from those originally authored in the target language. Conducted under the supervision of **Dr. Justin DeBenedetto** in the Computer Science Department at **Villanova University**, the research integrates linguistic theory with classic machine learning to support transparent and interpretable translation analysis.

## 🧠 Project Overview

- **Goal**: Classify English texts as either *original* or *translated* using interpretable linguistic features.
- **Motivation**: Translationese often manifests as decreased syntactic complexity, reduced lexical richness, increased cohesion, and structural predictability—traits that impact readability, style, and fluency in machine-generated text.
- **Approach**: Developed a modular Python pipeline that extracts 10+ handcrafted linguistic features and trains classical classifiers to detect translationese patterns.

## 🔍 Features Engineered

Each feature was implemented as a standalone Python module:
- **Lexical Diversity**: Type-Token Ratio (TTR), Moving Average TTR (MATTR)
- **Syntactic Complexity**: Clause depth, clause count, noun phrase count
- **Cohesion**: Cohesive marker frequency (e.g., "however," "in addition")
- **Information Theory**: Shannon entropy, burstiness of rare words
- **Grammatical Structure**: Part-of-speech (POS) distributions, dependency distance
- Optimized for large-scale processing with **spaCy's `nlp.pipe()`**

## 📊 Dataset & Processing

- Used English sentence pairs from the [CausalMT](https://github.com/EdisonNi-hku/CausalMT) dataset:
  - `full.en` — original English text (label 0)
  - `full.transen` — translated English text (label 1)
- Supported customizable chunk sizes (`--lines-per-sample`) for evaluating sentence vs. multi-sentence input.
- Data cleaning included:
  - Dropping sparse features (>30% missingness)
  - Mean imputation for remaining nulls
  - Scaling and normalization for classification

## 🤖 Classifiers Trained

- Logistic Regression ✅ (Best overall)
- Random Forest
- Decision Tree
- Gaussian Naive Bayes  
Each model was trained and evaluated with:
- 80/20 train-test split (stratified)
- 5-fold cross-validation
- Accuracy, Precision, Recall, F1-Score, and macro-averaged metrics

## ✅ Results

| Chunking Level | Best Model        | Test Accuracy | Macro F1 |
|----------------|-------------------|---------------|----------|
| Sentence       | Logistic Regression | 55.79%       | 55.77%   |
| 5-Line Chunk   | Logistic Regression | 64.97%       | 64.96%   |

- Chunked input significantly improved performance due to richer contextual representation.
- All experiments were logged, versioned, and visualized for reproducibility.


## 🧭 Future Work

- Integrate contextual embeddings (e.g., BERT, XLM-R)
- Explore alignment-based and semi-supervised learning
- Perform feature attribution (e.g., SHAP, permutation importance)
- Extend pipeline to multilingual settings
- Apply domain adaptation to test generalization across genres

## 📣 Citation

If referencing this project in academic or applied research, please cite:

**Kaygalak, Yasmin Lorin.** *Translationese Detection via Linguistic Feature Engineering and Machine Learning*. Villanova University, 2025.



