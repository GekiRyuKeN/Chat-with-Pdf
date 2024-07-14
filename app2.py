import streamlit as st
import pypdf
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import thinc
import numpy

# Check versions for compatibility
st.write(f"numpy version: {numpy.__version__}")
st.write(f"spacy version: {spacy.__version__}")
st.write(f"thinc version: {thinc.__version__}")

# Load models
@st.cache_resource
def load_models():
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    nlp = spacy.load("en_core_web_sm")
    return sentence_model, nlp

sentence_model, nlp = load_models()

# Inline CSS
def set_custom_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #333;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title {
        color: #2c3e50;
        font-size: 3em;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .subtitle {
        color: #34495e;
        font-size: 1.4em;
        font-weight: 300;
        text-align: center;
        margin-top: 0;
        margin-bottom: 30px;
    }
    .section-header {
        color: #2980b9;
        font-size: 1.8em;
        font-weight: 700;
        margin-top: 40px;
        margin-bottom: 20px;
        border-bottom: 2px solid #2980b9;
        padding-bottom: 10px;
    }
    .answer {
        background-color: #ecf0f1;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        border-left: 5px solid #3498db;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #7f8c8d;
        font-size: 0.9em;
        padding: 20px;
        border-top: 1px solid #e2e8f0;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: 700;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stTextInput>div>div>input {
        background-color: #fff;
        border: 2px solid #e2e8f0;
        border-radius: 5px;
        padding: 10px;
        font-size: 1em;
    }
    .stTextInput>div>div>input:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.5);
    }
    .stFileUploader {
        border: 2px dashed #3498db;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin-bottom: 30px;
    }
    .stFileUploader > div > div > div {
        color: #34495e;
        font-size: 1.1em;
    }
    </style>
    """, unsafe_allow_html=True)
    
# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = pypdf.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Text preprocessing
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Split text into chunks
def split_into_chunks(text, chunk_size=1000, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Find most relevant chunks
def find_most_relevant_chunks(question, chunks, top_k=3):
    question_embedding = sentence_model.encode([question])
    chunk_embeddings = sentence_model.encode(chunks)
    
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    return [chunks[i] for i in top_indices]

# Classify question type
def classify_question(question):
    doc = nlp(question.lower())
    if any(token.text in ["who", "whom"] for token in doc):
        return "PERSON"
    elif any(token.text in ["where", "location"] for token in doc):
        return "LOCATION"
    elif any(token.text in ["when", "date", "time"] for token in doc):
        return "DATE"
    else:
        return "GENERAL"

# Extract entities from text
def extract_entities(text, entity_type):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == entity_type]

# Answer question
def answer_question(question, text):
    chunks = split_into_chunks(text)
    relevant_chunks = find_most_relevant_chunks(question, chunks)
    combined_chunk = " ".join(relevant_chunks)
    
    question_type = classify_question(question)
    
    if question_type != "GENERAL":
        entities = extract_entities(combined_chunk, question_type)
        if entities:
            return f"Based on the content, the answer might be related to: {', '.join(entities)}"
    
    sentences = re.split(r'(?<=[.!?])\s+', combined_chunk)
    sentence_embeddings = sentence_model.encode(sentences)
    question_embedding = sentence_model.encode([question])
    
    similarities = cosine_similarity(question_embedding, sentence_embeddings)[0]
    top_sentences_indices = similarities.argsort()[-3:][::-1]
    
    answer = ' '.join([sentences[i] for i in top_sentences_indices])
    return answer if answer else "I couldn't find a relevant answer. Could you rephrase your question?"

# Diamante Net Hackathon specific content
diamante_info = """
Diamante Net Hackathon is an exciting event that brings together innovative minds to tackle challenges in the blockchain and cryptocurrency space. 
The hackathon focuses on developing solutions using the Diamante blockchain technology.
Key areas of interest include DeFi, NFTs, and cross-chain interoperability.
The event offers substantial prizes for the winning teams and networking opportunities with industry leaders.
Participants will have access to mentorship from blockchain experts throughout the hackathon.
"""

def main():
    set_custom_style()

    st.markdown("<h1 class='title'>Diamante Net Hackathon Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Your personal guide to the Diamante Net Hackathon</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Hackathon PDF Guide (optional)", type="pdf")

    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        full_text = diamante_info + " " + text
    else:
        full_text = diamante_info

    st.markdown("<h2 class='section-header'>Ask me anything about the Hackathon!</h2>", unsafe_allow_html=True)
    query = st.text_input("Your question:")
    if query:
        answer = answer_question(query, full_text)
        st.markdown(f"<div class='answer'><strong>Answer:</strong> {answer}</div>", unsafe_allow_html=True)

    st.markdown("<div class='footer'>Powered by Diamante Net</div>", unsafe_allow_html=True)

if __name__== "__main__":
    main()
