import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# Load the model and tokenizer
model_save_path = './model'  # Path to your model directory
tokenizer = BertTokenizerFast.from_pretrained(model_save_path)
model = BertForSequenceClassification.from_pretrained(model_save_path)

# Set the model to evaluation mode
model.eval()

# Function to predict ratings
def predict_rating(review):
    # Tokenize the review
    encoding = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Move input tensors to the same device as the model
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class + 1  # Convert back to original rating scale (1-5)

# Streamlit UI
st.set_page_config(page_title='Review Sentiment Rating Predictor', layout='centered')
st.title('📝 Review Sentiment Rating Predictor')
st.write('Enter a product review to get a rating prediction (1-5):')

# User input for review
user_review = st.text_area("Review text", "", height=150)

# Predict button
if st.button('🔍 Predict Rating'):
    if user_review.strip():
        predicted_rating = predict_rating(user_review)
        st.success(f'Predicted Rating: **{predicted_rating}/5**')
    else:
        st.error('Please enter a valid review.')

# Footer
st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line
st.markdown("<p style='text-align: center; color: grey;'>Developed using BERT and Streamlit</p>", unsafe_allow_html=True)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f0f2f5;  /* Light gray background */
            font-family: 'Arial', sans-serif;  /* Clean font */
        }
        .stTextInput, .stTextArea {
            border: 1px solid #ccc;  /* Light gray border */
            border-radius: 8px;
            padding: 12px;
            background-color: #ffffff;  /* White background */
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);  /* Subtle shadow */
        }
        .stButton {
            background-color: #007BFF;  /* Button color */
            color: white;  /* Button text color */
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);  /* Subtle shadow */
        }
        .stButton:hover {
            background-color: #0056b3;  /* Darker blue on hover */
        }
        h1, h2, h3 {
            color: #333;  /* Dark gray headers for readability */
        }
        p {
            color: #666;  /* Gray text for paragraphs */
        }
        hr {
            border: 1px solid #007BFF;  /* Blue horizontal line */
        }
    </style>
""", unsafe_allow_html=True)
