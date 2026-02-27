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
    # only attempt download if nltk is available
    if nltk is not None:
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

# perform download (safe even if nltk missing)
download_nltk()

# --- 2. DATASET (Problem: Cyberbullying) ---
@st.cache_data
def load_data():
    # Load dataset from a local CSV file provided by the user
    # Expected columns: tweet_text, cyberbullying_type
    # try relative path first, then fallback to absolute location
    import os
    base = os.path.dirname(__file__)
    default_path = os.path.join(base, "cyberbullying_tweets.csv")
    if os.path.exists(default_path):
        path = default_path
    else:
        # previous hardcoded path, kept for backwards compatibility
        path = r"C:\cyberbullyingapp\cyberbullying_tweets.csv"

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Dataset not found at {path}. Please place 'cyberbullying_tweets.csv' in the application folder.")
        return pd.DataFrame(columns=["tweet", "label"])
    except Exception as exc:
        st.error(f"Error loading dataset: {exc}")
        return pd.DataFrame(columns=["tweet", "label"])

    # map the text & label columns to a common format
    df = df.rename(columns={"tweet_text": "tweet"})
    # anything other than 'not_cyberbullying' is considered bullying
    df['label'] = df['cyberbullying_type'].apply(
        lambda x: 0 if str(x).lower() == 'not_cyberbullying' else 1
    )
    # Use more data for better training (previously 1000, now 5000)
    return df[['tweet', 'label']].sample(min(5000, len(df)), random_state=42)

# --- 3. PREPROCESSING ---
def preprocess_text(text):
    # fallback simple cleaning if nltk is not installed
    text = str(text).lower()
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", text)
    if nltk is None or stopwords is None or WordNetLemmatizer is None:
        # minimal tokenization
        return " ".join(text.split())
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# --- 4. MODEL TRAINING (with caching) ---
@st.cache_resource
def train_model():
    """Load data, preprocess, and train the cyberbullying detection model."""
    # if sklearn is missing, return a simple rule-based fallback
    if TfidfVectorizer is None or LogisticRegression is None:
        # define dummy model with predict and predict_proba methods
        class DummyModel:
            def predict(self, X):
                # X will be list of processed texts
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

    try:
        df = load_data()
        
        # Validate data
        if df.empty:
            st.error("Error: Dataset is empty!")
            return None, None, None
        
        if df['label'].nunique() < 2:
            st.error("Error: Not enough classes in labels!")
            return None, None, None
        
        # Preprocess text
        df['clean_text'] = df['tweet'].apply(preprocess_text)
        
        # Vectorize with better parameters
        vectorizer = TfidfVectorizer(max_features=1500, min_df=2, max_df=0.8)
        X = vectorizer.fit_transform(df['clean_text'])
        y = df['label']
        
        # Train with better parameters
        model = LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')
        model.fit(X, y)
        
        # Calculate training accuracy
        train_accuracy = model.score(X, y)
        
        return model, vectorizer, train_accuracy
    
    except Exception as e:
        st.error(f"Error during model training: {str(e)}")
        return None, None, None

# Train and get model
model_result = train_model()
if model_result[0] is not None:
    model, vectorizer, train_accuracy = model_result
else:
    model, vectorizer, train_accuracy = None, None, 0

# --- 5. STREAMLIT INTERFACE ---
st.title("🛡️ Cyberbullying Detection App")
st.write("Enter text below to check if it contains harmful or bullying language.")

# inform user if optional libraries are missing
missing = []
if nltk is None:
    missing.append('nltk')
if TfidfVectorizer is None or LogisticRegression is None:
    missing.append('scikit-learn')
if missing:
    st.warning(f"Missing dependencies: {', '.join(missing)}. "
               "Install via pip for full functionality.")

# Show model status
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
            try:
                processed = preprocess_text(user_input)
                # support fallback model without vectorizer
                if vectorizer is not None:
                    vec = vectorizer.transform([processed])
                    prediction = model.predict(vec)
                    confidence = model.predict_proba(vec)[0]
                else:
                    prediction = model.predict([processed])
                    confidence = model.predict_proba([processed])[0]

                if prediction[0] == 1:
                    st.error(f"🚨 Result: Potential Cyberbullying Detected (Confidence: {confidence[1]:.2%})")
                else:
                    st.success(f"✅ Result: Clean / Safe Content (Confidence: {confidence[0]:.2%})")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning("Please enter text first.")
else:
    st.error("❌ Model failed to train. Please check the dataset and try refreshing the page.")
