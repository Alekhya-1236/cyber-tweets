import streamlit as st
import pandas as pd
import re

# attempt to import optional NLP and ML libraries; handle missing packages gracefully
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError:
    nltk = None
    stopwords = None
    WordNetLemmatizer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
except ImportError:
    TfidfVectorizer = None
    LogisticRegression = None


# --- 1. SETUP & NLTK DATA ---
@st.cache_resource
def download_nltk():
    if nltk is not None:
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

download_nltk()


# --- 2. DATASET (FIXED FOR STREAMLIT CLOUD) ---
@st.cache_data
def load_data():
    path = "cyberbullying_tweets.csv"
    try:
        df = pd.read_csv(path)
    except Exception:
        st.error("Dataset not found! Upload cyberbullying_tweets.csv in same GitHub repo.")
        return pd.DataFrame(columns=["tweet", "label"])

    df = df.rename(columns={"tweet_text": "tweet"})
    df['label'] = df['cyberbullying_type'].apply(
        lambda x: 0 if str(x).lower() == 'not_cyberbullying' else 1
    )

    return df[['tweet', 'label']].sample(min(5000, len(df)), random_state=42)


# --- 3. PREPROCESSING ---
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", text)

    if nltk is None or stopwords is None or WordNetLemmatizer is None:
        return " ".join(text.split())

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)


# --- 4. MODEL TRAINING ---
@st.cache_resource
def train_model():
    if TfidfVectorizer is None or LogisticRegression is None:
        class DummyModel:
            def predict(self, X):
                return [1 if any(w in x for w in ['bully', 'hate', 'stupid', 'idiot']) else 0 for x in X]
            def predict_proba(self, X):
                probs = []
                for x in X:
                    if any(w in x for w in ['bully', 'hate', 'stupid', 'idiot']):
                        probs.append([0.2, 0.8])
                    else:
                        probs.append([0.9, 0.1])
                return probs
        return DummyModel(), None, 0.0

    df = load_data()

    if df.empty:
        st.error("Dataset empty!")
        return None, None, None

    if df['label'].nunique() < 2:
        st.error("Need both bullying and non-bullying samples!")
        return None, None, None

    df['clean_text'] = df['tweet'].apply(preprocess_text)

    vectorizer = TfidfVectorizer(max_features=1500, min_df=2, max_df=0.8)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    model = LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')
    model.fit(X, y)

    train_accuracy = model.score(X, y)

    return model, vectorizer, train_accuracy


model_result = train_model()
if model_result[0] is not None:
    model, vectorizer, train_accuracy = model_result
else:
    model, vectorizer, train_accuracy = None, None, 0


# --- 5. STREAMLIT INTERFACE ---
st.title("🛡️ Cyberbullying Detection App")
st.write("Enter text below to check if it contains harmful or bullying language.")

missing = []
if nltk is None:
    missing.append('nltk')
if TfidfVectorizer is None or LogisticRegression is None:
    missing.append('scikit-learn')
if missing:
    st.warning(f"Missing dependencies: {', '.join(missing)}")

if model is not None and vectorizer is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Status", "✅ Ready")
    with col2:
        st.metric("Training Accuracy", f"{train_accuracy:.2%}")

    st.divider()

    user_input = st.text_area("Input Text:", placeholder="Type a comment here...")

    if st.button("Analyze", type="primary"):
        if user_input:
            processed = preprocess_text(user_input)
            vec = vectorizer.transform([processed])
            prediction = model.predict(vec)
            confidence = model.predict_proba(vec)[0]

            if prediction[0] == 1:
                st.error(f"🚨 Result: Potential Cyberbullying Detected (Confidence: {confidence[1]:.2%})")
            else:
                st.success(f"✅ Result: Clean / Safe Content (Confidence: {confidence[0]:.2%})")
        else:
            st.warning("Please enter text first.")
else:
    st.error("❌ Model failed to train. Check dataset.")