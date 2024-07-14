import streamlit as st
import webbrowser

# Custom theme with gradient background and feature colors
def set_custom_theme():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1E3A8A, #3B82F6);
    }
    .big-font {
        font-size:50px !important;
        color: #FFFFFF;
        font-weight: bold;
    }
    .feature-text {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .about-us {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        margin-top: 30px;
    }
    .team-member {
