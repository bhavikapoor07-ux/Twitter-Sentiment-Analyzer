# 🐦 Twitter Sentiment Analyzer

> An AI-powered web application that analyzes the sentiment of tweets using a Bidirectional LSTM deep learning model — with emoji detection, sarcasm detection, toxic language filtering, Word Cloud, Compare Two Tweets, Recent Analyses &amp; Dark/Light Theme | Built with Python, Tensorflow &amp; Streamlit

🔗 **Live App:** [twitter-sentiment-analyzer-bk.streamlit.app](https://twitter-sentiment-analyzer-bk.streamlit.app)

---

## 📌 Overview

Twitter Sentiment Analyzer is a full-stack AI web application built with Python, TensorFlow, and Streamlit. It uses a custom-trained **Bidirectional LSTM (Long Short-Term Memory)** neural network to classify tweets into four sentiment categories: **Positive, Negative, Neutral, and Irrelevant** — with an impressive **86% accuracy** on the test set.

The app goes beyond basic sentiment classification by combining multiple AI-driven features into a clean, interactive, and visually stunning interface — complete with a floating stars animation, dark/light theme toggle, and full mobile support.

---

## ✨ Features

### 🧠 AI Sentiment Analysis
- Classifies any tweet into **Positive, Negative, Neutral, or Irrelevant**
- Displays **confidence scores** for all four sentiment classes
- Shows an interactive **bar chart** of model confidence per class
- Powered by a custom-trained **Bidirectional LSTM** model with 86% accuracy

### 🎭 Emoji Emotion Detection
- Detects the **emotional tone** of a tweet using keyword analysis
- Identifies 6 emotions: 😍 Love, 😊 Happy, 😡 Angry, 😢 Sad, 😱 Shocked, 😂 Funny/Sarcastic
- Shows a full **Emotion Breakdown** section with progress bars
- Falls back to 😐 Neutral when no strong emotion is detected

### 🎭 Sarcasm Detection
- Detects common sarcasm patterns and phrases in tweets
- Alerts the user when a tweet may contain sarcasm
- Warns that sentiment prediction may be less accurate for sarcastic tweets

### 🚨 Toxic Language Filter
- Scans tweets for offensive, aggressive, or harmful language
- Blocks analysis and displays a warning if toxic content is found
- Highlights the exact toxic words detected
- Promotes responsible and safe use of the application

### ⚖️ Compare Two Tweets
- Analyze **two tweets side by side** simultaneously
- View sentiment, emoji emotion, and confidence for both tweets
- Get an automatic **verdict** on which tweet is more positive
- View a **grouped bar chart** comparing confidence scores of both tweets

### 📜 Recent Analyses
- Stores and displays the **last 20 analyzed tweets**
- Shows sentiment label, confidence percentage, and emoji emotion for each
- Persists across page navigation within the same session

### 🥧 Sentiment Pie Chart
- Visual breakdown of **all tweets analyzed** in the current session
- Color-coded pie chart showing distribution across all sentiment classes
- Includes a metric summary showing counts for each sentiment

### ☁️ Word Cloud
- Generates a **Word Cloud** from all analyzed tweets
- Highlights the most frequently used words visually
- Theme-aware: dark background in dark mode, white in light mode
- Uses text preprocessing (stopword removal + stemming) for clean results

### 🌙 Dark / Light Theme
- Full **dark and light mode** toggle from the sidebar
- All UI elements, charts, cards, and backgrounds adapt to the selected theme
- Default theme is dark mode for a premium look

### 🌟 Floating Stars Animation
- Beautiful **floating stars background animation** with purple, white, and pink stars
- Adds a unique, premium visual experience to the app
- Runs smoothly without affecting app performance

---

## 🏗️ App Architecture

```
Twitter Sentiment Analyzer
│
├── 📊 Page 1 — Main Analysis
│     ├── Tweet input box
│     ├── Toxic language check
│     ├── AI sentiment prediction
│     ├── Emoji emotion detection
│     ├── Sarcasm detection
│     ├── Confidence scores (progress bars)
│     └── Confidence bar chart
│
├── 📜 Page 2 — Recent Analyses
│     └── Last 20 analyzed tweets with results
│
├── 🥧 Page 3 — Sentiment Pie Chart
│     └── Distribution of all analyzed sentiments
│
├── ☁️ Page 4 — Word Cloud
│     └── Most frequent words in analyzed tweets
│
└── ⚖️ Page 5 — Compare Two Tweets
      ├── Side-by-side sentiment analysis
      ├── Verdict on which tweet is more positive
      └── Grouped confidence comparison chart
```

---

## 🤖 Model Details

| Property | Details |
|---|---|
| Architecture | Bidirectional LSTM |
| LSTM Units | 128 → 64 |
| Regularization | L2 + SpatialDropout1D (0.3) |
| Vocabulary Size | 10,000 words |
| Max Sequence Length | 100 tokens |
| Output Classes | Positive, Negative, Neutral, Irrelevant |
| Test Accuracy | **86%** |
| Framework | TensorFlow / Keras |

### Training Pipeline
1. **Text Cleaning** — Regex cleaning, lowercasing, stopword removal (preserving "not", "no", "never"), Porter Stemming
2. **Tokenization** — Keras Tokenizer with vocabulary of 10,000 words
3. **Padding** — Post-padding to max length of 100 tokens
4. **Label Encoding** — Scikit-learn LabelEncoder
5. **Training** — Bidirectional LSTM with dropout and L2 regularization

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning model training & inference |
| Streamlit | Web application framework |
| NLTK | Text preprocessing (stopwords, stemming) |
| Scikit-learn | Label encoding |
| Matplotlib | Confidence charts and pie charts |
| WordCloud | Word cloud generation |
| Pillow | Image processing |
| NumPy | Numerical computations |

---

## 📁 Project Structure

```
Twitter-Sentiment-Analyzer/
│
├── app.py                  # Main Streamlit application
├── sentiment_model.keras   # Trained Bidirectional LSTM model
├── tokenizer.pkl           # Fitted Keras tokenizer
├── label_encoder.pkl       # Fitted label encoder
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version specification
└── README.md               # Project documentation
```

---

## 🚀 Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/bhavikapoor07-ux/Twitter-Sentiment-Analyzer.git
cd Twitter-Sentiment-Analyzer
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the app:**
```bash
streamlit run app.py
```

**4. Open in browser:**
```
http://localhost:8501
```

---

## 📊 Dataset

- **Source:** Twitter training dataset (CSV format)
- **Classes:** Positive, Negative, Neutral, Irrelevant
- **Preprocessing:** Regex cleaning, stopword removal, Porter Stemming

---

## 🎨 UI Highlights

- Custom **Share Tech Mono** font for a tech aesthetic
- Glowing neon purple color scheme
- Floating animated stars background
- Fully responsive — works on desktop and mobile
- Custom scrollbar, buttons, input fields, and cards
- Smooth hover animations on buttons

---

## 👩‍💻 Developer

**Bhavi Kapoor**
Built with ❤️ using Python, TensorFlow & Streamlit

🔗 **Live App:** [twitter-sentiment-analyzer-bk.streamlit.app](https://twitter-sentiment-analyzer-bk.streamlit.app)

🔗 **GitHub:** [github.com/bhavikapoor07-ux](https://github.com/bhavikapoor07-ux)

🔗 **LinkedIn** .[linkedin.com/in/bhavi-kapoor-aiml/](https://www.linkedin.com/in/bhavi-kapoor-aiml/).

---

© 2026 Bhavi Kapoor • Twitter Sentiment Analyzer
