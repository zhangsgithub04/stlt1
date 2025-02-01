from langchain import LLMChain
from transformers import GeminiModel, GeminiTokenizer

gemini_model = GeminiModel.from_pretrained("google/gemini-1b")
gemini_tokenizer = GeminiTokenizer.from_pretrained("google/gemini-1b")

def generate_response(input_text):
chain = LLMChain(llm=gemini_model, tokenizer=gemini_tokenizer)
output = chain.generate(text_input=input_text)
st.info(output) 
