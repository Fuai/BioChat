import streamlit as st
import pandas as pd
import plotly.express as px
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
from utils import process_experimental_data, prepare_context, DRUG_DISCOVERY_PROMPT

# Load environment variables from key.env
load_dotenv("key.env")

# Configure page
st.set_page_config(page_title="Drug Discovery Assistant", layout="wide")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation" not in st.session_state:
    # Initialize LangChain components
    llm = ChatOpenAI(
        model_name="gpt-4-turbo-preview",
        temperature=0.7,
    )
    
    # Create a conversation chain with our custom prompt
    st.session_state.conversation = LLMChain(
        llm=llm,
        prompt=DRUG_DISCOVERY_PROMPT,
        verbose=True
    )

# Load and cache the data
@st.cache_data
def load_data():
    data = pd.read_csv("TurboID_ASK1_ML_Final.csv")
    return process_experimental_data(data)

try:
    data = load_data()
    st.sidebar.success("Data loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading data: {str(e)}")

# Main interface
st.title("Drug Discovery Knowledge Base Assistant")

# Sidebar for data exploration
with st.sidebar:
    st.title("Data Explorer")
    if 'data' in locals():
        st.write(f"Dataset Shape: {data.shape}")
        selected_columns = st.multiselect(
            "Select columns to view",
            data.columns.tolist(),
            default=data.columns.tolist()[:5]
        )
        
        if selected_columns:
            st.dataframe(data[selected_columns].head())
    
    # PubMed settings
    st.title("PubMed Settings")
    include_pubmed = st.checkbox("Include PubMed results", value=True)
    if include_pubmed:
        max_results = st.slider("Max PubMed results", 1, 10, 5)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your drug discovery data (e.g., 'What is DMR5?')..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching data and literature..."):
            # Prepare context with experimental data and PubMed results
            context_data = prepare_context(data, prompt, include_pubmed=include_pubmed)
            
            # Get response from conversation chain
            response = st.session_state.conversation.predict(
                context=context_data["context"],
                pubmed_context=context_data["pubmed_context"],
                question=prompt
            )
            
            # Display response
            st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Add visualization if requested
            if any(keyword in prompt.lower() for keyword in ['plot', 'graph', 'visualize', 'show']):
                if 'data' in locals() and len(selected_columns) >= 2:
                    fig = px.scatter(data, x=selected_columns[0], y=selected_columns[1],
                                   title=f"{selected_columns[0]} vs {selected_columns[1]}")
                    st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("Drug Discovery Assistant powered by LangChain and OpenAI with PubMed Integration") 