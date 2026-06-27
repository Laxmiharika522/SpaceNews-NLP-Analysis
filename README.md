# 🚀 Space News NLP Analysis

A comprehensive **Natural Language Processing** project that analyzes space news articles from **2010 to 2024**, applying a full NLP pipeline — from text preprocessing to topic modeling, sentiment analysis, named entity recognition, and machine learning classification — to extract meaningful insights from real-world media data.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Analysis Pipeline](#analysis-pipeline)
- [Key Findings](#key-findings)
- [Visualizations](#visualizations)
- [How to Run](#how-to-run)

---

## Overview

This project investigates how space exploration is reported across major agencies and over time. Using a corpus of thousands of news articles, it answers questions such as:

- How has the **tone and sentiment** of space news shifted from 2010 to 2024?
- Which **space agencies** — NASA, ISRO, SpaceX, ESA, CNSA — receive the most media coverage, and how does their sentiment compare?
- What **topics and keywords** define different eras of space reporting?
- How do **language patterns** differ between mission successes and failures?
- Can a machine learning model **accurately classify** articles by mission category?

The notebook walks through every step of the analysis with visualizations and observations at each stage.

---

## Dataset

The dataset used in this project is publicly available on Kaggle:

[![Kaggle](https://img.shields.io/badge/Kaggle-Space%20News%20Dataset-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/patrickfleith/space-news-dataset)

**🔗** [https://www.kaggle.com/datasets/patrickfleith/space-news-dataset](https://www.kaggle.com/datasets/patrickfleith/space-news-dataset)

| Field | Details |
|---|---|
| **Source** | Space News articles from multiple publishers |
| **Coverage** | 2010 – 2024 |
| **Key Columns** | `title`, `text` / `body`, `date`, `source` |

After downloading, place `spacenews.csv` in the **root of the project folder** (same directory as the notebook):

---

## 📁 Project Structure

```
SpaceNews-NLP-Analysis/
│
├── README.md                        # Project documentation
├── DataScience_project.ipynb        # Main Jupyter Notebook
├── spacenews.csv                    # Dataset (download from Kaggle — not tracked in git)
└── images/                          # All generated visualizations
    ├── Data Processing Pipeline.png
    ├── Category Distribution.png
    ├── Word Clouds.png
    ├── Topic Distribution.png
    ├── Topic Trends Over Time.png
    ├── LDA Topics Analysis.png
    ├── Keyword Trends.png
    ├── NER Analysis.png
    ├── N-Gram Analysis.png
    ├── N-Gram Positive vs Negative.png
    ├── Sentiment Overview.png
    ├── Sentiment Trend Over Time.png
    ├── Text Length vs Sentiment.png
    ├── TF-IDF Overall.png
    ├── TF-IDF Per Agency.png
    ├── TF-IDF Per Period.png
    └── Naive Bayes Results.png
```

---

## 🛠️ Technologies Used

| Category | Tools & Libraries |
|---|---|
| **Data Processing** | `pandas`, `numpy` |
| **NLP & Text Processing** | `nltk`, `spaCy` (`en_core_web_sm`), `vaderSentiment` |
| **Machine Learning** | `scikit-learn` — TF-IDF Vectorizer, Multinomial Naive Bayes |
| **Topic Modeling** | `gensim` (LDA), `pyLDAvis` |
| **Visualization** | `matplotlib`, `seaborn`, `wordcloud` |
| **Environment** | Python 3, Jupyter Notebook |

---

## 🔬 Analysis Pipeline

### 1. Data Loading & Preprocessing

The raw CSV is loaded and cleaned through a structured pipeline:

- Column names are standardized and date fields are parsed
- Articles are filtered to the **2010–2024** window
- The `title` and `text` fields are merged into a single `content` field for analysis
- Text cleaning steps applied sequentially:
  - Lowercasing
  - Removal of punctuation and numbers
  - Stop-word removal using NLTK's English corpus, extended with domain-specific terms (`space`, `mission`, `launch`, `agency`, etc.)
  - Lemmatization using `WordNetLemmatizer`

### 2. Agency Tagging

Each article is tagged to one of five major space agencies using keyword matching on the article content:

| Agency | Keywords |
|---|---|
| NASA | nasa, artemis, james webb, mars rover |
| ISRO | isro, chandrayaan, mangalyaan, india |
| SpaceX | spacex, elon, falcon, starship |
| ESA | esa, european space, ariane |
| CNSA | cnsa, china, tiangong, long march |

### 3. Sentiment Analysis — VADER

Sentiment scoring is performed using the **VADER (Valence Aware Dictionary and Sentiment Reasoner)** model, which is well-suited for short, informal text like news headlines:

- `compound ≥ 0.05` → **Positive**
- `compound ≤ -0.05` → **Negative**
- Otherwise → **Neutral**

Sentiment is aggregated per agency and tracked year-by-year, with annotations marking key real-world events (JWST launch, Chandrayaan-3, SpaceX Crew Dragon, etc.).

### 4. TF-IDF Keyword Analysis

TF-IDF (Term Frequency–Inverse Document Frequency) is applied at three levels:

- **Corpus-wide** — top 20 keywords across all articles
- **Per agency** — distinct vocabulary fingerprints for NASA, ISRO, SpaceX, ESA, and CNSA
- **Per time period** — keyword evolution across five 3-year windows (2010–12, 2013–15, 2016–18, 2019–21, 2022–24)

### 5. LDA Topic Modeling

Latent Dirichlet Allocation (LDA) via `gensim` is used to discover **6 latent topics** in the corpus:

| Topic | Theme |
|---|---|
| Topic 0 | Lunar Missions |
| Topic 1 | Mars Exploration |
| Topic 2 | Satellite Launches |
| Topic 3 | Human Spaceflight |
| Topic 4 | Space Science & Telescopes |
| Topic 5 | Mission Failures & Anomalies |

Topic dominance is tracked over time to reveal long-term trends in media focus.

### 6. Word Clouds

Four word clouds are generated from cleaned text, each highlighting a different lens:

- **All Articles** — overall corpus vocabulary
- **NASA Articles** — NASA-specific language
- **ISRO Articles** — ISRO-specific language
- **Negative Articles** — terms associated with setbacks and failures

### 7. N-Gram Analysis

Frequent multi-word phrases are extracted using NLTK's `ngrams`:

- **Top-15 bigrams and trigrams** from the full corpus
- **Bigram comparison** between Positive and Negative articles, revealing distinct linguistic signals around mission outcomes

### 8. Named Entity Recognition (NER)

Using `spaCy`'s `en_core_web_sm` model on a 3,000-article sample:

- **ORG** — most mentioned space organizations and mission names
- **GPE** — most mentioned countries and locations
- **Entity type distribution** — breakdown of all entity categories identified

### 9. Naive Bayes Text Classification

A supervised classification pipeline is built to categorize articles into 6 mission types:

- **Lunar Mission**, **Mars Exploration**, **Satellite Launch**, **Human Spaceflight**, **Space Science**, **Mission Failure**

The model uses TF-IDF features (5,000 features) with an 80/20 train-test split and is evaluated using a confusion matrix, precision, recall, and F1-score.

---

## 💡 Key Findings

- **Sentiment trends upward** over the 2010–2024 period, with measurable positive peaks around the SpaceX Crew Dragon mission (2020), JWST launch (2021), and Chandrayaan-3 landing (2023).
- **NASA leads in volume** of coverage, while **ISRO** shows significant growth in article frequency post-2019, reflecting India's expanding space program.
- **Commercial spaceflight vocabulary** — SpaceX, Falcon, Starship — has grown steadily since 2015, reflecting the rise of private space industry in media coverage.
- **Mars Exploration** dominated topic trends around 2020–21, corresponding to the Perseverance rover launch; **Lunar Missions** resurged from 2022 with Artemis and Chandrayaan.
- **Bigram analysis** reveals clear linguistic separation: positive articles use phrases like "successful launch" and "historic mission", while negative articles are characterized by "mission abort", "launch delay", and "rocket failure".
- The **Naive Bayes classifier** successfully distinguishes mission types, validating that space news articles carry strong domain-specific signals in their language.

---

## 📊 Visualizations

### Data Processing Pipeline
![Data Processing Pipeline](images/Data%20Processing%20Pipeline.png)

### Category Distribution
![Category Distribution](images/Category%20Distribution.png)

### Word Clouds
![Word Clouds](images/Word%20Clouds.png)

### Topic Distribution
![Topic Distribution](images/Topic%20Distribution.png)

### Topic Trends Over Time
![Topic Trends Over Time](images/Topic%20Trends%20Over%20Time.png)

### LDA Topics Analysis
![LDA Topics Analysis](images/LDA%20Topics%20Analysis.png)

### Keyword Trends
![Keyword Trends](images/Keyword%20Trends.png)

### NER (Named Entity Recognition) Analysis
![NER Analysis](images/NER%20Analysis.png)

### N-Gram Analysis
![N-Gram Analysis](images/N-Gram%20Analysis.png)

### N-Gram Positive vs Negative Analysis
![N-Gram Positive vs Negative](images/N-Gram%20Positive%20vs%20Negative.png)

### Sentiment Overview
![Sentiment Overview](images/Sentiment%20Overview.png)

### Sentiment Trend Over Time
![Sentiment Trend Over Time](images/Sentiment%20Trend%20Over%20Time.png)

### Text Length vs Sentiment
![Text Length vs Sentiment](images/Text%20Length%20vs%20Sentiment.png)

### TF-IDF Overall Analysis
![TF-IDF Overall](images/TF-IDF%20Overall.png)

### TF-IDF Per Agency
![TF-IDF Per Agency](images/TF-IDF%20Per%20Agency.png)

### TF-IDF Per Period
![TF-IDF Per Period](images/TF-IDF%20Per%20Period.png)

### Naive Bayes Classification Results
![Naive Bayes Results](images/Naive%20Bayes%20Results.png)

---

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Laxmiharika522/SpaceNews-NLP-Analysis.git
   cd SpaceNews-NLP-Analysis
   ```

2. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/patrickfleith/space-news-dataset) and place `spacenews.csv` in the root project folder (same directory as the notebook)

3. **Install dependencies**
   ```bash
   pip install vaderSentiment nltk scikit-learn gensim pyLDAvis wordcloud matplotlib seaborn pandas numpy spacy
   python -m spacy download en_core_web_sm
   ```

4. **Open the notebook**
   ```bash
   jupyter notebook DataScience_project.ipynb
   ```

5. **Run all cells** — the notebook executes the full pipeline and saves all visualizations to the `images/` folder.

---

> 📌 *This project was built as part of a Data Science course to demonstrate end-to-end text analytics on a real-world dataset.*
