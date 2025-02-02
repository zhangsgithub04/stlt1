import os
import streamlit as st
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

# Set OpenAI organization
os.environ["OPENAI_ORGANIZATION"] = st.secrets["open_ai_api"]

# Initialize LLM
llm = OpenAI()

# Create prompt template
prompt = PromptTemplate.from_template("How to say {input} in {output_language}:\n")

# Create chain
chain = prompt | llm

# Streamlit app
st.title("Language Translation App")

# Input fields
input_text = st.text_input("Enter text to translate:")
output_language = st.selectbox("Select output language:", ["German", "French", "Spanish"])

# Translate button
if st.button("Translate"):
    # Invoke chain
    output = chain.invoke(
        {
            "output_language": output_language,
            "input": input_text,
        }
    )

    # Display output
    st.write("Translation:")
    st.code(output)
