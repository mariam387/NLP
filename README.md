# Text Mining Projects – GUC Course

This repository contains hands-on NLP projects completed as part of the Text Mining course at the German University in Cairo (GUC). Projects cover a range of text mining techniques including Bag-of-Words, One-Hot Encoding, Sentence Embedding, and Retrieval-Augmented Generation (RAG) using a restaurant dialogue dataset.

text-mining-guc/
├── README.md
├── requirements.txt
├── .gitignore
│
├── Final_Project/
│   ├── main.py
│   ├── vector.py
│   ├── evaluate.py
│   ├── realistic_restaurant_reviews.csv
│   ├── chrome_langchain_db/         # Local vector DB (can be ignored in .gitignore)
│   └── venv/                        # Virtual environment (should be ignored)
│
├── NLP_Assignments/
│   ├── NLP_Assignments.ipynb
│   ├── Train.csv                    # For sentiment classification (BOW/OHE)
│   └── TMDB_tv_dataset_v3.csv      # For speaker prediction (sentence embeddings)




## Contents

### 1. Sentiment Classification – BOW & OHE
- **Goal:** Classify user comments as Positive or Negative.
- **Techniques:** Bag-of-Words, One-Hot Encoding, Logistic Regression.
- **Input:** User comment.
- **Output:** Sentiment label (positive/negative).

---

### 2. Speaker Identification – Sentence Embeddings
- **Goal:** Predict the speaker of a given sentence from a dialogue.
- **Techniques:** Sentence Embedding (e.g., Sentence-BERT), Multi-class Classification.
- **Input:** A sentence from a dialogue.
- **Output:** Most likely speaker.

---

### 3. Final Project – Restaurant QA Chatbot (RAG with Ollama)
- **Goal:** Build a question-answering chatbot using a restaurant dataset.
- **Method:** Retrieval-Augmented Generation (RAG) with a locally hosted pre-trained model via [Ollama](https://ollama.com/).
- **Features:**
  - Embedding and storing context in a vector store.
  - Retrieval of relevant content based on queries.
  - Answer generation using a language model.
  - Evaluation with **perplexity**.

---

## Dataset Sources
- `sentiment_data.csv`: Labeled user comments.
- `speakers_dataset.csv`: Dialogue with speaker labels.
- `restaurant_dataset.csv`: Used in RAG final project.

