# 🚀 Space News NLP Analysis

A comprehensive Natural Language Processing (NLP) project that analyzes thousands of space news articles spanning **2010–2024** to uncover sentiment patterns, key topics, trending keywords, and agency-level insights using classical ML and NLP techniques.

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

This project applies a full NLP pipeline to a large corpus of space news articles to answer questions like:

- How has the **sentiment** of space news evolved over the years?
- Which **space agencies** (NASA, ISRO, SpaceX, ESA, CNSA) dominate the coverage?
- What are the most significant **topics and keywords** across different eras?
- Can a machine learning model **classify** articles by mission type?
- What **named entities** (organizations, locations) appear most frequently?

---

## Dataset

> **⚠️ The dataset is not included in this repository due to its large size (~85 MB).**

**Download it from Kaggle:**

[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Space%20News%20Dataset-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/patrickfleith/space-news-dataset)

🔗 **Direct Link:** [https://kaggle.com/datasets/patrickfleith/space-news-dataset](https://kaggle.com/datasets/patrickfleith/space-news-dataset)

After downloading, place the file as:
```
dataset/spacenews.csv
```

**Dataset Details:**
- **Source:** Space News articles aggregated from various publishers
- **Time Range:** 2010 – 2024
- **Key Columns:** `title`, `text`/`body`, `date`, `source`
- **File Size:** ~85 MB (CSV)

---

## 📁 Project Structure

```
SpaceNews-NLP-Analysis/
│
├── README.md                          # Project documentation
├── DataScience_project.ipynb          # Main analysis notebook
├── images/                            # All generated visualizations
│   ├── Data Processing Pipeline.png
│   ├── Category Distribution.png
│   ├── Word Clouds.png
│   ├── Topic Distribution.png
│   ├── Topic Trends Over Time.png
│   ├── LDA Topics Analysis.png
│   ├── Keyword Trends.png
│   ├── NER Analysis.png
│   ├── N-Gram Analysis.png
│   ├── N-Gram Positive vs Negative.png
│   ├── Sentiment Overview.png
│   ├── Sentiment Trend Over Time.png
│   ├── Text Length vs Sentiment.png
│   ├── TF-IDF Overall.png
│   ├── TF-IDF Per Agency.png
│   ├── TF-IDF Per Period.png
│   └── Naive Bayes Results.png
└── dataset/                           # ⚠️ Not included — download from Kaggle
    └── spacenews.csv
```

---

## 🛠️ Technologies Used

| Category | Libraries |
|---|---|
| **Data Processing** | `pandas`, `numpy` |
| **NLP & Text** | `nltk`, `spacy` (`en_core_web_sm`), `vaderSentiment` |
| **Machine Learning** | `scikit-learn` (TF-IDF, Naive Bayes, Label Encoding) |
| **Topic Modeling** | `gensim` (LDA), `pyLDAvis` |
| **Visualization** | `matplotlib`, `seaborn`, `wordcloud` |
| **Language** | Python 3 (Jupyter Notebook) |

---

## 🔬 Analysis Pipeline

The project follows a structured NLP workflow:

### 1. 📥 Data Loading & Preprocessing
- Load `spacenews.csv` and standardize column names
- Parse and filter dates to the **2010–2024** range
- Combine `title` + `text` fields into a unified `content` column
- **Text cleaning pipeline:**
  - Lowercasing
  - Punctuation & number removal
  - Stop-word removal (NLTK + domain-specific space words)
  - Lemmatization (WordNetLemmatizer)

### 2. 🏷️ Agency Tagging
Articles are keyword-tagged into five major space agencies:
- **NASA** — artemis, james webb, mars rover, nasa
- **ISRO** — isro, chandrayaan, mangalyaan, india
- **SpaceX** — spacex, elon, falcon, starship
- **ESA** — esa, european space, ariane
- **CNSA** — cnsa, china, tiangong, long march

### 3. 💬 Sentiment Analysis (VADER)
- Applied `SentimentIntensityAnalyzer` from the `vaderSentiment` library
- Compound score thresholds: `≥ 0.05` → Positive, `≤ -0.05` → Negative, else Neutral
- Tracked **year-by-year sentiment trends** with annotations for major events (Chandrayaan-3, JWST launch, etc.)

### 4. 📊 TF-IDF Analysis
- Extracted **top-20 keywords** from the entire corpus
- Computed **agency-specific** TF-IDF fingerprints (NASA vs ISRO vs SpaceX, etc.)
- Analyzed **temporal keyword shifts** across five 3-year periods (2010–2024)

### 5. 🧠 Topic Modeling (LDA)
- Built a `gensim` LDA model with **6 discovered topics**:
  - Lunar Missions, Mars Exploration, Satellite Launch, Human Spaceflight, Space Science, Mission Failures
- Visualized topic-word probability distributions
- Tracked **topic dominance over years** to detect trends (e.g., SpaceX's commercial rise from 2015+)

### 6. ☁️ Word Clouds
- Generated 4 word clouds: All Articles, NASA-specific, ISRO-specific, Negative Articles

### 7. 🔤 N-Gram Analysis
- Extracted **top-15 bigrams and trigrams** from cleaned text
- Compared bigrams between **Positive vs Negative** articles (e.g., "successful launch" vs "mission failure")

### 8. 👁️ Named Entity Recognition (NER)
- Used `spaCy` (`en_core_web_sm`) on a 3,000-article sample
- Identified top **Organizations (ORG)**, **Locations (GPE)**, and **entity type distribution**

### 9. 🤖 Naive Bayes Classification
- Created 6 article categories: Lunar Mission, Mars Exploration, Satellite Launch, Human Spaceflight, Space Science, Mission Failure
- Trained a **Multinomial Naive Bayes** classifier (TF-IDF features, 5,000 max features)
- Evaluated with a **confusion matrix**, precision, recall, and F1-score

---

## 💡 Key Findings

- 📈 **Sentiment has generally improved** over 2010–2024, with notable positive spikes around **SpaceX Crew Dragon (2020)**, **JWST Launch (2021)**, and **Chandrayaan-3 (2023)**
- 🏆 **NASA dominates** space news coverage volume, but **ISRO** showed a strong surge post-2019
- 🚀 **Commercial spaceflight** keywords (SpaceX, Falcon, Starship) have grown dramatically since 2015
- 🌙 **Mars Exploration** peaked around 2020–21 (Perseverance rover), while **Lunar Missions** gained renewed coverage from 2022 with Artemis and Chandrayaan
- 📰 Positive articles cluster around **"successful launch"** and **"historic mission"**, while negative articles focus on **"mission abort"**, **"launch delay"**, and **"rocket failure"**
- 🎯 The **Naive Bayes classifier** effectively distinguishes between mission types using TF-IDF features

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

2. **Download the dataset** from [Kaggle](https://kaggle.com/datasets/patrickfleith/space-news-dataset) and place it in the `dataset/` folder as `spacenews.csv`

3. **Install dependencies**
   ```bash
   pip install vaderSentiment nltk scikit-learn gensim pyLDAvis wordcloud matplotlib seaborn pandas numpy spacy
   python -m spacy download en_core_web_sm
   ```

4. **Launch the notebook**
   ```bash
   jupyter notebook DataScience_project.ipynb
   ```

5. **Run all cells** — the notebook will preprocess the data, run all analyses, and save all visualizations to the `images/` folder.

---

> 📌 *This project was built as part of a Data Science course to demonstrate end-to-end text analytics on a real-world dataset.*
