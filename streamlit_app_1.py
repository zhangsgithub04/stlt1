import streamlit as st
from langchain import LLMMathChain
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Title
st.title("Gemini AI Chatbot")

# Instructions
st.write("Ask me anything!")

# Input text
input_text = st.text_area("Enter your question or prompt", height=100)

# Initialize the model and tokenizer
model_name = "google/gemini-1b"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the chain
def get_chain(model, tokenizer):
    return LLMMathChain(llm=model, tokenizer=tokenizer)

# Generate response
def generate_response(input_text):
    chain = get_chain(model, tokenizer)
    output = chain.generate(text_input=input_text)
    return output

# Button to generate response
if st.button("Get Response"):
    if input_text:
        response = generate_response(input_text)
        st.write("Response:")
        st.write(response)
    else:
        st.write("Please enter a prompt or question.")
