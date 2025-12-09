import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling - Official Netflix Colors
st.markdown("""
    <style>
    /* Netflix-style background */
    .stApp {
        background: linear-gradient(to bottom, rgba(34,31,31,0.95) 0%, rgba(34,31,31,0.7) 50%, rgba(34,31,31,0.95) 100%),
                    url('https://images.pexels.com/photos/436413/pexels-photo-436413.jpeg');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    
    /* Make text visible on dark background */
    .stApp, .stMarkdown, p, label, .stTextArea label {
        color: #F5F5F1 !important;
    }
    
    /* Main header with Netflix Red */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(120deg, #E50914 0%, #B20710 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 0 10px rgba(229,9,20,0.3));
    }
    
    .sub-header {
        text-align: center;
        color: #F5F5F1 !important;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
        opacity: 0.8;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        font-size: 1.1rem;
        border-radius: 10px;
        background: rgba(34, 31, 31, 0.95) !important;
        color: #F5F5F1 !important;
        border: 2px solid rgba(86, 77, 77, 0.5) !important;
        backdrop-filter: blur(10px);
    }
    
    .stTextArea textarea:focus {
        border-color: #B20710 !important;
        box-shadow: 0 0 15px rgba(178,7,16,0.3) !important;
    }
    
    /* Button styling - Netflix Red */
    .stButton button {
        background: #E50914 !important;
        color: #F5F5F1 !important;
        border: none !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5) !important;
        letter-spacing: 0.5px;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(229,9,20,0.5) !important;
        background: #B20710 !important;
    }
    
    /* Sidebar styling - Netflix Black */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #221F1F 0%, #141414 100%) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(86,77,77,0.3);
    }
    
    [data-testid="stSidebar"] * {
        color: #F5F5F1 !important;
    }
    
    /* Card backgrounds */
    .positive-result {
        background: linear-gradient(135deg, rgba(40,180,99,0.95) 0%, rgba(34,153,84,0.9) 100%);
        padding: 2rem;
        border-radius: 15px;
        color: #F5F5F1;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(40,180,99,0.3);
    }
    
    .negative-result {
        background: linear-gradient(135deg, #E50914 0%, #831010 100%);
        padding: 2rem;
        border-radius: 15px;
        color: #F5F5F1;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(229,9,20,0.4);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(229,9,20,0.3);
    }
    
    /* Metric containers */
    [data-testid="stMetricValue"] {
        color: #F5F5F1 !important;
        font-size: 2rem !important;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #F5F5F1 !important;
        opacity: 0.7;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #221F1F !important;
        color: #F5F5F1 !important;
        border-radius: 10px;
        border: 1px solid #564d4d !important;
    }
    
    /* Progress bars - Netflix Red */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #E50914 0%, #B20710 100%);
    }
    
    /* Success/Error boxes */
    .stSuccess, .stError, .stWarning, .stInfo {
        background: rgba(34,31,31,0.95) !important;
        backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid #564d4d;
    }
    
    /* Divider */
    hr {
        border-color: #564d4d !important;
    }
    
    /* Metric cards background */
    [data-testid="stMetric"] {
        background: rgba(34,31,31,0.8);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #564d4d;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
MAX_LEN = 100

# Load word index
@st.cache_data
def load_word_index():
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    return word_index, reverse_word_index

word_index, reverse_word_index = load_word_index()

# Load model
@st.cache_resource
def load_trained_model():
    try:
        model = load_model('simple_rnn_imdb.h5', compile=False)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model, None
    except Exception as e:
        return None, str(e)

model, error = load_trained_model()

# Helper functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=MAX_LEN)
    return padded_review

def create_sentiment_bar(score):
    """Create a visual sentiment bar using Streamlit native components"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Negative**")
        st.progress(1 - score)
        st.markdown(f"<h3 style='text-align: center; color: #f5576c;'>{(1-score)*100:.1f}%</h3>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Positive**")
        st.progress(score)
        st.markdown(f"<h3 style='text-align: center; color: #667eea;'>{score*100:.1f}%</h3>", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🎬 CINEMATIX</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Movie Review Sentiment Analysis</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg", width=200)
    st.markdown("---")
    
    st.subheader("📊 Model Information")
    if model is not None:
        st.success("✅ Model Loaded")
        st.info(f"**Sequence Length:** {MAX_LEN} words")
        st.info(f"**Input Shape:** {model.input_shape}")
    else:
        st.error("❌ Model Not Loaded")
        st.warning(error)
    
    st.markdown("---")
    st.subheader("🧪 Quick Test")
    
    if st.button("🔬 Run Sample Tests", use_container_width=True):
        if model is not None:
            test_reviews = [
                ("This movie is amazing and wonderful", "Positive"),
                ("Terrible waste of time", "Negative"),
                ("I loved this film", "Positive"),
                ("Awful and boring", "Negative")
            ]
            
            for review, expected in test_reviews:
                preprocessed = preprocess_text(review)
                pred = model.predict(preprocessed, verbose=0)
                score = float(pred[0][0])
                predicted = "Positive" if score > 0.5 else "Negative"
                
                if predicted == expected:
                    st.success(f"✅ {review[:30]}... → {predicted} ({score:.2f})")
                else:
                    st.error(f"❌ {review[:30]}... → {predicted} ({score:.2f})")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Write natural movie reviews
    - Use descriptive words
    - Longer reviews work better
    - Mix positive/negative words for nuanced analysis
    """)

# Main content
if model is None:
    st.error("⚠️ Please ensure 'simple_rnn_imdb.h5' exists in the current directory!")
else:
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ✍️ Enter Your Movie Review")
        user_input = st.text_area(
            "",
            placeholder="Example: This movie was absolutely fantastic! The acting was superb, the plot was engaging, and I loved every minute of it. Highly recommended!",
            height=200,
            label_visibility="collapsed"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            analyze_btn = st.button("🎯 Analyze Sentiment", use_container_width=True, type="primary")
        
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_btn:
            st.rerun()
    
    with col2:
        st.markdown("### 📝 Sample Reviews")
        samples = [
            "This movie was absolutely fantastic!",
            "Terrible waste of time and money.",
            "One of the best films I've ever seen!",
            "Boring and poorly executed."
        ]
        
        for i, sample in enumerate(samples):
            if st.button(f"📌 {sample[:30]}...", key=f"sample_{i}", use_container_width=True):
                user_input = sample
                st.rerun()
    
    # Analysis section
    if analyze_btn:
        if user_input.strip() == '':
            st.warning("⚠️ Please enter a movie review to analyze.")
        else:
            with st.spinner("🔄 Analyzing sentiment..."):
                try:
                    # Preprocess and predict
                    preprocessed_input = preprocess_text(user_input)
                    prediction = model.predict(preprocessed_input, verbose=0)
                    score = float(prediction[0][0])
                    sentiment = 'Positive' if score > 0.5 else 'Negative'
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 🎯 Analysis Results")
                    
                    # Result banner
                    if sentiment == 'Positive':
                        st.markdown(f"""
                        <div class="positive-result">
                            😊 POSITIVE SENTIMENT<br>
                            <span style="font-size: 2rem;">{score*100:.1f}%</span> Confidence
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="negative-result">
                            😞 NEGATIVE SENTIMENT<br>
                            <span style="font-size: 2rem;">{(1-score)*100:.1f}%</span> Confidence
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Metrics in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="😊 Positive Score",
                            value=f"{score*100:.1f}%",
                            delta=f"{(score-0.5)*100:.1f}% from neutral"
                        )
                    
                    with col2:
                        st.metric(
                            label="😞 Negative Score",
                            value=f"{(1-score)*100:.1f}%",
                            delta=f"{(0.5-score)*100:.1f}% from neutral"
                        )
                    
                    with col3:
                        st.metric(
                            label="📊 Confidence",
                            value=f"{abs(score-0.5)*200:.1f}%"
                        )
                    
                    # Visual sentiment bars
                    st.markdown("### 📊 Sentiment Breakdown")
                    create_sentiment_bar(score)
                    
                    # Debug info in expander
                    with st.expander("🔍 Technical Details"):
                        st.write(f"**Number of words:** {len(user_input.split())}")
                        st.write(f"**Processed shape:** {preprocessed_input.shape}")
                        st.write(f"**Raw prediction score:** {score:.6f}")
                        st.write(f"**Threshold:** 0.5")
                        st.write(f"**Classification:** {sentiment}")
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #999; padding: 1rem;">
        <p style="color: #999 !important;">🎬 CINEMATIX | Powered by TensorFlow & Deep Learning</p>
        <p style="color: #666 !important; font-size: 0.9rem;">Built with ❤️ for movie enthusiasts</p>
    </div>
    """, unsafe_allow_html=True)