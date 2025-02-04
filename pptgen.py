import os
#from langchain.chains.qa import QA
from langchain.chains.retrieval_qa import RetrievalQA
                     
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

# Initialize LLM
llm = OpenAI()

# Define prompt template for suggesting template ideas
template_suggestion_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest 5 template ideas for the following topic: {topic}",
)

# Define prompt template for generating a topic based on a template
topic_generation_prompt = PromptTemplate(
    input_variables=["template", "topic"],
    template="Generate a topic based on the following template: {template}. The topic is about {topic}.",
)

# Input topic from user
topic = input("Enter a topic idea: ")

# Suggest template ideas

# Initialize the Retrieval QA chain
retrieval_qa = RetrievalQA.from_llm(llm=llm)

question1="Suggest 5 template ideas for the following topic: "+ topic;

# Use the Retrieval QA chain to answer a question
template_ideas = retrieval_qa({"question": question1, "num_results": 5});
                      
#template_suggestion_chain = QA(prompt=template_suggestion_prompt, llm=llm)
#template_ideas = template_suggestion_chain({"topic": topic})

# Display template ideas
print("Template Ideas:")
for i, idea in enumerate(template_ideas):
    print(f"{i+1}. {idea}")

# Select a template idea
selected_template_index = int(input("Enter the number of the template idea you'd like to select: "))
selected_template = template_ideas[selected_template_index - 1]

# Generate a topic based on the selected template
question2="Generate a topic based on the following template:" + selected_template+". The topic is about "+ topic+".";

# Use the Retrieval QA chain to answer a question
 generated_topic= retrieval_qa({"question": question2, "num_results": 5})
                              
#topic_generation_chain = QA(prompt=topic_generation_prompt, llm=llm)
#generated_topic = topic_generation_chain({"template": selected_template, "topic": topic})

# Display generated topic
print("Generated topic:")
print(generated_topic)
