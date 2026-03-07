import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import nltk
import time

nltk.download('stopwords', quiet=True)

# ── PAGE CONFIGURATION ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SESSION STATE INITIALIZATION ──────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ── THEME COLORS ──────────────────────────────────────────────────────────────

if st.session_state.dark_mode:
    BG          = "#0a0010"
    SIDEBAR_BG  = "rgba(13,0,32,0.97)"
    CARD_BG     = "rgba(13,0,32,0.85)"
    TEXT        = "#e0ccff"
    TITLE_COLOR = "#bf7fff"
    ACCENT      = "#7b2fff"
    BORDER      = "#7b2fff"
    INPUT_BG    = "rgba(13,0,32,0.9)"
    INPUT_TEXT  = "#bf7fff"
    CHART_BG    = "#0d0020"
    CAPTION     = "#9966cc"
    BTN_GRAD    = "linear-gradient(135deg,#5500cc,#7b2fff)"
    BTN_HOVER   = "linear-gradient(135deg,#7b2fff,#cc99ff)"
    SCROLL_BG   = "#0a0010"
    SCROLL_THB  = "#7b2fff"
    HR_COLOR    = "#7b2fff"
    SUBTEXT     = "#cc99ff"
else:
    BG          = "#f0f0ff"
    SIDEBAR_BG  = "rgba(230,220,255,0.98)"
    CARD_BG     = "rgba(255,255,255,0.9)"
    TEXT        = "#1a0040"
    TITLE_COLOR = "#5500cc"
    ACCENT      = "#7b2fff"
    BORDER      = "#9966cc"
    INPUT_BG    = "rgba(255,255,255,0.95)"
    INPUT_TEXT  = "#3d0099"
    CHART_BG    = "#ffffff"
    CAPTION     = "#6633cc"
    BTN_GRAD    = "linear-gradient(135deg,#7b2fff,#9933ff)"
    BTN_HOVER   = "linear-gradient(135deg,#5500cc,#7b2fff)"
    SCROLL_BG   = "#e6e0ff"
    SCROLL_THB  = "#9966cc"
    HR_COLOR    = "#9966cc"
    SUBTEXT     = "#5500cc"

# ── INJECT THEME CSS ──────────────────────────────────────────────────────────

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

.stApp {{
    background-color: {BG} !important;
    font-family: 'Share Tech Mono', monospace;
}}

.stars-container {{
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 1; pointer-events: none; overflow: hidden;
}}

.star {{
    position: absolute; background: white;
    border-radius: 50%; animation: floatStar linear infinite; opacity: 0;
}}

@keyframes floatStar {{
    0%   {{ transform: translateY(100vh) scale(0); opacity: 0; }}
    10%  {{ opacity: 1; }}
    90%  {{ opacity: 1; }}
    100% {{ transform: translateY(-10vh) scale(1); opacity: 0; }}
}}

.star.purple {{ background: #bf7fff; box-shadow: 0 0 6px 2px #7b2fff; }}
.star.white  {{ background: #ffffff; box-shadow: 0 0 4px 1px #cc99ff; }}
.star.pink   {{ background: #dd99ff; box-shadow: 0 0 5px 2px #9933ff; }}

.block-container {{
    position: relative; z-index: 5;
    background-color: transparent !important;
}}

/* ── EQUAL HEIGHT CARDS ── */
[data-testid="stHorizontalBlock"] {{
    align-items: stretch !important;
}}
[data-testid="stHorizontalBlock"] > div {{
    display: flex !important;
    flex-direction: column !important;
}}
[data-testid="stHorizontalBlock"] > div > div {{
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}}

[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG} !important;
    border-right: 1px solid {BORDER}; z-index: 1000 !important;
}}
[data-testid="stSidebar"] * {{ color: {SUBTEXT} !important; }}

h1 {{
    color: {TITLE_COLOR} !important;
    text-shadow: 0 0 20px {ACCENT}, 0 0 40px #5500cc;
    font-family: 'Share Tech Mono', monospace !important;
}}
h2, h3 {{ color: {SUBTEXT} !important; font-family: 'Share Tech Mono', monospace !important; }}
p, label, .stMarkdown {{ color: {TEXT} !important; }}

.stTextArea textarea {{
    background-color: {INPUT_BG} !important;
    color: {INPUT_TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    font-family: 'Share Tech Mono', monospace !important;
    caret-color: {ACCENT} !important;
}}
.stTextArea textarea:focus {{
    border-color: {SUBTEXT} !important;
    box-shadow: 0 0 10px {ACCENT} !important;
}}

.stButton > button {{
    background: {BTN_GRAD} !important;
    color: white !important;
    border: 1px solid {SUBTEXT} !important;
    border-radius: 8px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 16px !important;
    padding: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 15px {ACCENT}55 !important;
}}
.stButton > button:hover {{
    background: {BTN_HOVER} !important;
    box-shadow: 0 0 25px {ACCENT} !important;
    transform: translateY(-2px) !important;
}}

.stProgress > div > div {{ background: linear-gradient(90deg,#5500cc,#bf7fff) !important; }}

[data-testid="stMetric"] {{
    background-color: {CARD_BG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important; padding: 10px !important;
    box-shadow: 0 0 10px {ACCENT}33 !important;
}}
[data-testid="stMetricValue"] {{ color: {TITLE_COLOR} !important; }}

hr {{ border-color: {HR_COLOR} !important; opacity: 0.4; }}

[data-testid="stSidebar"] .stRadio label {{
    color: {SUBTEXT} !important;
    font-family: 'Share Tech Mono', monospace !important;
}}

.stCaption {{ color: {CAPTION} !important; }}

::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {SCROLL_BG}; }}
::-webkit-scrollbar-thumb {{ background: {SCROLL_THB}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: #cc99ff; }}

section[data-testid="stSidebar"] > div {{
    background-color: {SIDEBAR_BG} !important;
}}
</style>

<!-- Floating Stars Container -->
<div class="stars-container" id="stars-container"></div>
""", unsafe_allow_html=True)

# ── FLOATING STARS ANIMATION ──────────────────────────────────────────────────

components.html("""
<script>
function createStars() {
    const container = parent.document.getElementById('stars-container');
    if (!container) { setTimeout(createStars, 200); return; }
    const types = ['purple','white','pink'];
    for (let i = 0; i < 80; i++) {
        const star = parent.document.createElement('div');
        star.classList.add('star');
        star.classList.add(types[Math.floor(Math.random()*types.length)]);
        const size = Math.random()*4 + 2;
        star.style.width = size + "px";
        star.style.height = size + "px";
        star.style.left = Math.random()*100 + "vw";
        star.style.bottom = Math.random()*100 + "vh";
        star.style.animationDuration = (Math.random()*12+6) + "s";
        star.style.animationDelay = (Math.random()*15) + "s";
        container.appendChild(star);
    }
}
createStars();
</script>
""", height=0)

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def show_footer():
    st.markdown(
        f"""
        <hr style="margin-top:40px; opacity:0.3;">
        <div style="text-align:center; color:{CAPTION}; font-size:13px;">
        © 2026 Bhavi Kapoor • Twitter Sentiment Analyzer <br>
        Built with Python | TensorFlow | Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_everything():
    model = load_model("sentiment_model_new")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        lb = pickle.load(f)
    return model, tokenizer, lb

model, tokenizer, lb = load_everything()

# ── TEXT CLEANING ─────────────────────────────────────────────────────────────

stop_words = set(stopwords.words("english"))
stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("never")
stop_words.discard("but")
ps = PorterStemmer()

def clean_text(text):
    filtered_words = []
    stemmed_words = []
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    for word in text:
        if word not in stop_words:
            filtered_words.append(word)
    for word in filtered_words:
        stemmed_words.append(ps.stem(word))
    return " ".join(stemmed_words)

# ── PREDICTION FUNCTION ───────────────────────────────────────────────────────

def predict_sentiment(tweet):
    cleaned = clean_text(tweet)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    prediction = model.predict(padded)
    confidence_scores = prediction[0]
    predicted_class = np.argmax(confidence_scores)
    predicted_label = lb.classes_[predicted_class]
    return predicted_label, confidence_scores

# ── EMOJI SENTIMENT DETECTION ─────────────────────────────────────────────────

def detect_emoji_sentiment(text):
    text_lower = text.lower()
    love_words    = ["love","adore","amazing","wonderful","beautiful","perfect","fantastic","lovely","cherish","affection","heart","crush","darling","adorable","sweet"]
    happy_words   = ["happy","great","awesome","excellent","good","glad","pleased","joy","delighted","enjoy","yay","thrilled","excited","smile","fun","brilliant","best"]
    angry_words   = ["angry","furious","hate","mad","rage","annoyed","irritated","disgusting","outraged","infuriated","livid","aggressive","horrible"]
    sad_words     = ["sad","unhappy","depressed","miserable","crying","tears","heartbroken","upset","disappointed","lonely","sorrow","grief","devastated","hopeless"]
    shocked_words = ["shocked","surprised","unbelievable","wow","omg","unexpected","astonishing","stunning","remarkable","unreal","incredible","what","cant believe"]
    funny_words   = ["lol","haha","hilarious","funny","joke","laughing","comedy","amusing","witty","sarcastic","ironic","ridiculous","absurd"]
    scores = {
        "😍 Love":            sum(1 for w in love_words    if w in text_lower),
        "😊 Happy":           sum(1 for w in happy_words   if w in text_lower),
        "😡 Angry":           sum(1 for w in angry_words   if w in text_lower),
        "😢 Sad":             sum(1 for w in sad_words     if w in text_lower),
        "😱 Shocked":         sum(1 for w in shocked_words if w in text_lower),
        "😂 Funny/Sarcastic": sum(1 for w in funny_words   if w in text_lower),
    }
    best_emoji = max(scores, key=scores.get)
    best_score = scores[best_emoji]
    if best_score == 0:
        return "😐 Neutral/No strong emotion", scores
    return best_emoji, scores

# ── SARCASM DETECTION ─────────────────────────────────────────────────────────

def detect_sarcasm(text):
    sarcasm_patterns = [
        "oh great","yeah right","sure sure","oh wow","oh really","totally",
        "obviously","clearly","oh perfect","just what i needed","oh fantastic",
        "wow thanks","oh wonderful","as if","yeah okay","oh absolutely",
        "right right","sure thing","oh brilliant","how lovely","oh how nice"
    ]
    has_multiple_punctuation = bool(re.search(r'[!?]{2,}', text))
    text_lower = text.lower()
    has_sarcasm_phrase = any(phrase in text_lower for phrase in sarcasm_patterns)
    return has_sarcasm_phrase or has_multiple_punctuation

# ── TOXIC LANGUAGE DETECTOR ───────────────────────────────────────────────────

def detect_toxic(text):
    toxic_words = [
        "hate","kill","stupid","idiot","dumb","trash","garbage","disgusting",
        "pathetic","worthless","loser","moron","useless","horrible","awful",
        "terrible","despise","worst","ugly","nasty","shut up","get lost",
        "go away","nobody cares","scam","fraud","liar","fake","disgrace","harm"
    ]
    text_lower = text.lower()
    found_words = [w for w in toxic_words if re.search(rf"\b{re.escape(w)}\b", text_lower)]
    return len(found_words) > 0, found_words

# ── EMOJI & COLOR MAPPING ─────────────────────────────────────────────────────

def get_emoji(sentiment):
    mapping = {
        "Positive":   "🟢 POSITIVE",
        "Negative":   "🔴 NEGATIVE",
        "Neutral":    "🟡 NEUTRAL",
        "Irrelevant": "⚪ IRRELEVANT"
    }
    return mapping.get(sentiment, sentiment)

def get_color(sentiment):
    mapping = {
        "Positive":   "#00C853",
        "Negative":   "#D50000",
        "Neutral":    "#FFD600",
        "Irrelevant": "#9E9E9E"
    }
    return mapping.get(sentiment, "#2196F3")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

st.sidebar.title("🐦 Menu")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["📊 Main Analysis", "📜 Recent Analyses", "🥧 Sentiment Pie Chart", "☁️ Word Cloud", "⚖️ Compare Two Tweets"]
)

st.sidebar.markdown("---")

theme_label = "☀️ Switch to Light Mode" if st.session_state.dark_mode else "🌙 Switch to Dark Mode"
if st.sidebar.button(theme_label, use_container_width=True):
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"🧠 Tweets Analyzed: {len(st.session_state.history)}")
st.sidebar.caption("Model: Bidirectional LSTM | Accuracy: 86%")
st.sidebar.caption(f"Theme: {'🌙 Dark' if st.session_state.dark_mode else '☀️ Light'}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Main Analysis":

    st.title("🐦 Twitter Sentiment Analyzer")
    st.markdown("Type any tweet below and the model will predict its sentiment!")
    st.markdown("---")

    tweet_input = st.text_area(
        "Enter your tweet here:",
        placeholder="e.g. I absolutely love this product! It's amazing!",
        height=120
    )

    if st.button("🔍 Analyze Sentiment", use_container_width=True):

        if tweet_input.strip() == "":
            st.warning("⚠️ Please enter a tweet before analyzing!")

        else:
            is_toxic, toxic_found = detect_toxic(tweet_input)

            if is_toxic:
                st.markdown("---")
                st.error("🚨 Toxic language detected! This tweet contains offensive or aggressive content.")
                st.markdown(
                    f"<div style='background-color:#2d0000; padding:10px 15px; border-radius:8px; border-left: 5px solid #D50000;'>"
                    f"<p style='color:#ff6b6b; margin:0;'>⚠️ Detected toxic words: <b>{', '.join(toxic_found)}</b></p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.stop()

            # ── SPINNER — PAGE 1 ──────────────────────────────────────────
            with st.spinner("🔍 Analyzing sentiment... please wait!"):
                label, scores = predict_sentiment(tweet_input)
                emoji_result, emoji_scores = detect_emoji_sentiment(tweet_input)

            st.session_state.history.append({
                "tweet":      tweet_input,
                "label":      label,
                "confidence": float(np.max(scores)) * 100,
                "emoji":      emoji_result
            })
            if len(st.session_state.history) > 20:
                st.session_state.history.pop(0)

            st.markdown("---")

            # ── RESULT — side by side ─────────────────────────────────────
            st.subheader("📊 Result")
            color = get_color(label)

            col_sent, col_emoji = st.columns(2)

            with col_sent:
                st.markdown(
                    f"<div style='background-color:{CARD_BG}; border:1px solid {BORDER}; border-radius:10px; padding:15px; text-align:center; box-shadow:0 0 12px {ACCENT}33;'>"
                    f"<p style='color:{SUBTEXT}; margin:0; font-size:13px;'>🤖 Model Sentiment</p>"
                    f"<h2 style='color:{color}; text-shadow:0 0 20px {color}; margin:8px 0;'>{get_emoji(label)}</h2>"
                    f"<p style='color:{color}; margin:0; font-size:14px;'>{float(np.max(scores))*100:.1f}% confident</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with col_emoji:
                st.markdown(
                    f"<div style='background-color:{CARD_BG}; border:1px solid {BORDER}; border-radius:10px; padding:15px; text-align:center; box-shadow:0 0 12px {ACCENT}33;'>"
                    f"<p style='color:{SUBTEXT}; margin:0; font-size:13px;'>🎭 Emoji Emotion</p>"
                    f"<h2 style='color:{TITLE_COLOR}; text-shadow:0 0 20px {ACCENT}; margin:8px 0;'>{emoji_result}</h2>"
                    f"<p style='color:{SUBTEXT}; margin:0; font-size:14px;'>Detected from tweet content</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            if detect_sarcasm(tweet_input):
                st.info("🎭 This tweet may be sarcastic! Sentiment prediction might not be fully accurate.")

            st.markdown("---")

            # ── EMOJI EMOTION BREAKDOWN ───────────────────────────────────
            st.subheader("🎭 Emotion Breakdown")
            for emoji_name, score in emoji_scores.items():
                if score > 0:
                    st.markdown(f"**{emoji_name}**")
                    st.progress(min(score / 5, 1.0))
                    st.markdown(f"`{score} keyword(s) detected`")

            st.markdown("---")

            # ── CONFIDENCE SCORES ─────────────────────────────────────────
            st.subheader("📈 Confidence Scores")
            classes = lb.classes_
            for i, cls in enumerate(classes):
                percentage = float(scores[i]) * 100
                st.markdown(f"**{cls}**")
                st.progress(float(scores[i]))
                st.markdown(f"`{percentage:.2f}%`")

            st.markdown("---")

            # ── BAR CHART ─────────────────────────────────────────────────
            st.subheader("📉 Confidence Chart")
            with st.spinner("📊 Generating chart..."):
                time.sleep(1)
                colors = [get_color(cls) for cls in classes]
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_facecolor(BG)
                ax.set_facecolor(CHART_BG)
                bars = ax.bar(classes, scores * 100, color=colors, edgecolor=ACCENT, linewidth=0.8)

                for bar, score in zip(bars, scores):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{score * 100:.1f}%", ha='center', va='bottom',
                        fontweight='bold', fontsize=11, color=SUBTEXT)

                ax.set_ylabel("Confidence (%)", fontsize=12, color=SUBTEXT)
                ax.set_xlabel("Sentiment Class", fontsize=12, color=SUBTEXT)
                ax.set_title("Model Confidence per Sentiment Class", fontsize=14, color=TITLE_COLOR)
                ax.set_ylim(0, 115)
                ax.tick_params(colors=SUBTEXT)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color(ACCENT)
                ax.spines['bottom'].set_color(ACCENT)
                st.pyplot(fig)

    show_footer()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RECENT ANALYSES
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📜 Recent Analyses":

    st.title("📜 Recent Analyses")
    st.markdown("Your last 20 analyzed tweets appear here.")
    st.markdown("---")

    if not st.session_state.history:
        st.info("🔍 No tweets analyzed yet! Go to Main Analysis and analyze some tweets first.")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            color       = get_color(item["label"])
            emoji_label = get_emoji(item["label"])
            tweet_emoji = item.get("emoji", "")
            st.markdown(
                f"""
                <div style="
                    background-color: {CARD_BG};
                    border-left: 5px solid {color};
                    padding: 12px 18px; border-radius: 8px;
                    margin-bottom: 12px;
                    box-shadow: 0 0 10px {ACCENT}33;
                ">
                    <p style="color:{SUBTEXT}; margin:0; font-size:15px; font-family:'Share Tech Mono',monospace;">
                        🐦 <i>"{item['tweet'][:100]}{'...' if len(item['tweet'])>100 else ''}"</i>
                    </p>
                    <p style="color:{color}; margin:6px 0 0 0; font-weight:bold; font-size:16px; text-shadow:0 0 8px {color};">
                        {emoji_label} &nbsp;|&nbsp; {item['confidence']:.1f}% &nbsp;|&nbsp; {tweet_emoji}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    show_footer()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SENTIMENT PIE CHART
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🥧 Sentiment Pie Chart":

    st.title("🥧 Sentiment Pie Chart")
    st.markdown("Visual breakdown of all tweets you have analyzed so far.")
    st.markdown("---")

    if not st.session_state.history:
        st.info("🔍 No tweets analyzed yet! Go to Main Analysis and analyze some tweets first.")
    else:
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0, "Irrelevant": 0}
        for item in st.session_state.history:
            if item["label"] in sentiment_counts:
                sentiment_counts[item["label"]] += 1

        filtered = {k: v for k, v in sentiment_counts.items() if v > 0}
        labels = list(filtered.keys())
        sizes  = list(filtered.values())
        colors = [get_color(label) for label in labels]

        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=140, wedgeprops=dict(edgecolor=ACCENT, linewidth=2)
        )
        for text in texts:
            text.set_fontsize(13); text.set_fontweight('bold'); text.set_color(SUBTEXT)
        for autotext in autotexts:
            autotext.set_fontsize(12); autotext.set_color('white'); autotext.set_fontweight('bold')

        ax.set_title("Sentiment Distribution", fontsize=16, fontweight='bold', pad=20, color=TITLE_COLOR)
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("📊 Breakdown")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🟢 Positive",   sentiment_counts["Positive"])
        col2.metric("🔴 Negative",   sentiment_counts["Negative"])
        col3.metric("🟡 Neutral",    sentiment_counts["Neutral"])
        col4.metric("⚪ Irrelevant", sentiment_counts["Irrelevant"])

    show_footer()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — WORD CLOUD
# ══════════════════════════════════════════════════════════════════════════════

elif page == "☁️ Word Cloud":

    st.title("☁️ Word Cloud")
    st.markdown("Visual representation of the most frequently used words in your analyzed tweets.")
    st.markdown("---")

    if not st.session_state.history:
        st.info("🔍 No tweets analyzed yet! Go to Main Analysis and analyze some tweets first.")
    else:
        all_text     = " ".join([item["tweet"] for item in st.session_state.history])
        cleaned_text = clean_text(all_text)

        if cleaned_text.strip() == "":
            st.warning("⚠️ Not enough words to generate a Word Cloud. Analyze more tweets first!")
        else:
            wc_bg       = "#0a0010" if st.session_state.dark_mode else "#ffffff"
            wc_colormap = "Purples"

            wordcloud = WordCloud(
                width=900, height=450,
                background_color=wc_bg,
                colormap=wc_colormap,
                max_words=100, min_font_size=10, max_font_size=120,
                collocations=False
            ).generate(cleaned_text)

            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor(BG)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title("Most Frequent Words in Your Analyzed Tweets",
                         fontsize=16, fontweight='bold', pad=15, color=TITLE_COLOR)
            st.pyplot(fig)
            st.markdown("---")
            st.caption("💡 Bigger words appear more frequently in your analyzed tweets.")

    show_footer()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — COMPARE TWO TWEETS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚖️ Compare Two Tweets":

    st.title("⚖️ Compare Two Tweets")
    st.markdown("Analyze two tweets side by side and see which one is more positive!")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🐦 Tweet 1")
        tweet1 = st.text_area("Enter first tweet:", placeholder="e.g. I love Apple products!", height=120, key="tweet1")
    with col2:
        st.subheader("🐦 Tweet 2")
        tweet2 = st.text_area("Enter second tweet:", placeholder="e.g. Google services are terrible!", height=120, key="tweet2")

    if st.button("⚖️ Compare Now", use_container_width=True):

        if tweet1.strip() == "" or tweet2.strip() == "":
            st.warning("⚠️ Please enter both tweets before comparing!")
        else:
            is_toxic1, toxic_found1 = detect_toxic(tweet1)
            is_toxic2, toxic_found2 = detect_toxic(tweet2)

            if is_toxic1 or is_toxic2:
                if is_toxic1:
                    st.error(f"🚨 Tweet 1 contains toxic words: **{', '.join(toxic_found1)}**")
                if is_toxic2:
                    st.error(f"🚨 Tweet 2 contains toxic words: **{', '.join(toxic_found2)}**")
                st.stop()

            # ── SPINNER — PAGE 5 ──────────────────────────────────────────
            with st.spinner("⚖️ Comparing tweets... please wait!"):
                label1, scores1 = predict_sentiment(tweet1)
                label2, scores2 = predict_sentiment(tweet2)
                emoji1, _ = detect_emoji_sentiment(tweet1)
                emoji2, _ = detect_emoji_sentiment(tweet2)

            st.markdown("---")
            st.subheader("📊 Results")

            col1, col2 = st.columns(2)
            with col1:
                color1 = get_color(label1)
                st.markdown(
                    f"<div style='background-color:{CARD_BG}; border-left:5px solid {color1}; padding:15px; border-radius:8px; text-align:center; box-shadow:0 0 15px {color1}33;'>"
                    f"<p style='color:{SUBTEXT}; font-size:13px; margin:0;'>Tweet 1</p>"
                    f"<h3 style='color:{color1}; margin:8px 0; text-shadow:0 0 10px {color1};'>{get_emoji(label1)}</h3>"
                    f"<p style='color:{TITLE_COLOR}; margin:4px 0; font-size:22px;'>{emoji1}</p>"
                    f"<p style='color:{color1}; margin:0; font-size:16px; font-weight:bold;'>{float(np.max(scores1))*100:.1f}% confident</p>"
                    f"</div>", unsafe_allow_html=True
                )
                if detect_sarcasm(tweet1):
                    st.info("🎭 Tweet 1 may be sarcastic!")

            with col2:
                color2 = get_color(label2)
                st.markdown(
                    f"<div style='background-color:{CARD_BG}; border-left:5px solid {color2}; padding:15px; border-radius:8px; text-align:center; box-shadow:0 0 15px {color2}33;'>"
                    f"<p style='color:{SUBTEXT}; font-size:13px; margin:0;'>Tweet 2</p>"
                    f"<h3 style='color:{color2}; margin:8px 0; text-shadow:0 0 10px {color2};'>{get_emoji(label2)}</h3>"
                    f"<p style='color:{TITLE_COLOR}; margin:4px 0; font-size:22px;'>{emoji2}</p>"
                    f"<p style='color:{color2}; margin:0; font-size:16px; font-weight:bold;'>{float(np.max(scores2))*100:.1f}% confident</p>"
                    f"</div>", unsafe_allow_html=True
                )
                if detect_sarcasm(tweet2):
                    st.info("🎭 Tweet 2 may be sarcastic!")

            st.markdown("---")
            st.subheader("🏆 Verdict")

            pos_score1 = float(scores1[list(lb.classes_).index("Positive")]) if "Positive" in lb.classes_ else 0
            pos_score2 = float(scores2[list(lb.classes_).index("Positive")]) if "Positive" in lb.classes_ else 0

            if label1 == label2:
                if float(np.max(scores1)) > float(np.max(scores2)):
                    st.success("🏆 Tweet 1 wins — same sentiment but higher confidence!")
                elif float(np.max(scores2)) > float(np.max(scores1)):
                    st.success("🏆 Tweet 2 wins — same sentiment but higher confidence!")
                else:
                    st.info("🤝 It's a tie — both tweets have the same sentiment and confidence!")
            elif label1 == "Positive" and label2 != "Positive":
                st.success("🏆 Tweet 1 is more Positive!")
            elif label2 == "Positive" and label1 != "Positive":
                st.success("🏆 Tweet 2 is more Positive!")
            elif pos_score1 > pos_score2:
                st.success("🏆 Tweet 1 has a higher positivity score!")
            else:
                st.success("🏆 Tweet 2 has a higher positivity score!")

            st.markdown("---")
            st.subheader("📉 Confidence Comparison Chart")

            classes = list(lb.classes_)
            x = np.arange(len(classes))
            width = 0.35

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(CHART_BG)

            bars1 = ax.bar(x - width/2, scores1*100, width, label='Tweet 1', color='#7b2fff', edgecolor=SUBTEXT, linewidth=0.5)
            bars2 = ax.bar(x + width/2, scores2*100, width, label='Tweet 2', color='#bf7fff', edgecolor=SUBTEXT, linewidth=0.5)

            for bar in bars1:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                        f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color=SUBTEXT)
            for bar in bars2:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                        f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color=SUBTEXT)

            ax.set_ylabel("Confidence (%)", fontsize=12, color=SUBTEXT)
            ax.set_xlabel("Sentiment Class", fontsize=12, color=SUBTEXT)
            ax.set_title("Tweet 1 vs Tweet 2 — Confidence Comparison", fontsize=14, color=TITLE_COLOR)
            ax.set_xticks(x)
            ax.set_xticklabels(classes, color=SUBTEXT)
            ax.set_ylim(0, 120)
            ax.tick_params(colors=SUBTEXT)
            ax.legend(facecolor=CHART_BG, edgecolor=ACCENT, labelcolor=SUBTEXT)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(ACCENT)
            ax.spines['bottom'].set_color(ACCENT)
            st.pyplot(fig)

    show_footer()
