# COVID-19 Fake News Detection using ML & BERT
### *Spring’25 • Design Lab Project under Prof. Niloy Ganguly*

---

## Overview
This project focuses on detecting **COVID-19 related fake news** from social media platforms using both **Classical Machine Learning models** and **Transformer-based BERT models**.  
The goal is to compare traditional NLP pipelines with modern transformer architectures and evaluate which approach performs best for short, noisy social-media text.

---

## Dataset
- **Constraint@AAAI-2021 Fake News Dataset**
- Contains **10.6K social media posts**
- Labels:
  - **0 → Fake**
  - **1 → Real**

Data includes tweets, Facebook posts, captions, and statements.

---

## Tech Stack
- Python  
- Pandas  
- Scikit-learn  
- Hugging Face Transformers  
- PyTorch  
- HuggingFace Datasets  
- TF-IDF Vectorizer  

---

## Data Preprocessing
- Lowercasing  
- Removing URLs & hashtags  
- Converting emojis → text using `emoji.demojize`  
- Removing extra spaces  
- Label encoding  
- Train/Validation/Test split  
- Converted into **Hugging Face Dataset** for efficient tokenization  

---

## Classical Machine Learning Models

Using **TF-IDF (5000 features)**, the following models were trained:

| Model | Best Accuracy | Notes |
| **Logistic Regression** | **92.6%** | Best classical baseline |
| SVM (Linear Kernel) | ~92% | Good for high-dim TF-IDF |
| KNN | ~90% | Best with cosine distance |
| Neural Network (MLP) | ~90–92% | Tuned with RandomizedSearch |

### Hyperparameter Tuning:
- **GridSearchCV** → Logistic Regression, SVM, KNN  
- **RandomizedSearchCV** → Neural Network  

---

## Transformer Models (BERT Family)

Four transformer models were fine-tuned:

1. **BERT-base-uncased**  
2. **COVID-Twitter-BERT**  
3. **SocBERT**  
4. **TwHIN-BERT (Twitter Heterogeneous Information Network BERT)**  

### **Best Model → TwHIN-BERT**
- **Accuracy:** 97%  
- **F1-score:** 0.975  

TwHIN-BERT performs best because it is pretrained on **massive Twitter data**, capturing slang, emojis, hashtags, and short text patterns.

---

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score (micro, macro)  
- Confusion Matrix  

F1-score is used as the main metric due to balanced classes.

---

## Key Learnings
- Created an end-to-end NLP pipeline  
- Understood ML vs BERT performance differences  
- Learned fine-tuning, tokenization, attention masks  
- Observed impact of **domain-specific pretraining**  
- Applied hyperparameter tuning for multiple models  

---

