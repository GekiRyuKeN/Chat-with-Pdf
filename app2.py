import streamlit as st
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfFileReader(file)
    text = ""
    for page in range(pdf_reader.numPages):
        text += pdf_reader.getPage(page).extract_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    return text.strip()

# Function to split text into sentences using NLTK
def split_into_sentences(text):
    return sent_tokenize(text)

# Load models
@st.cache(allow_output_mutation=True)
def load_models():
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return sentence_model

# Answer question function
def answer_question(question, text):
    sentences = split_into_sentences(text)
    sentence_model = load_models()
    question_embedding = sentence_model.encode([question])
    sentence_embeddings = sentence_model.encode(sentences)
    
    similarities = cosine_similarity(question_embedding, sentence_embeddings)[0]
    top_sentence_indices = similarities.argsort()[-3:][::-1]
    
    top_sentences = [sentences[i] for i in top_sentence_indices]
    return ' '.join(top_sentences)

# Diamante Net Hackathon specific content
diamante_info = """
Diamante Net Hackathon is an exciting event that brings together innovative minds to tackle challenges in the blockchain and cryptocurrency space. 
The hackathon focuses on developing solutions using the Diamante blockchain technology.
Key areas of interest include DeFi, NFTs, and cross-chain interoperability.
The event offers substantial prizes for the winning teams and networking opportunities with industry leaders.
Participants will have access to mentorship from blockchain experts throughout the hackathon.
"""

def main():
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

if __name__ == "__main__":
    main()
