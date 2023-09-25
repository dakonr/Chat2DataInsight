import pandas as pd
import openai
import streamlit as st
import warnings, os, json
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import matplotlib.pyplot as plt

#Models
available_models = {"ChatGPT-3.5": "gpt-3.5-turbo-0301", "ChatGPT-4": "gpt-4"}

#Chat Logic
##Load file
def load_file(filename):
    df = pd.read_csv(filename)
    st.write(df.head())
    return df
#Generate LLM Response
def generate_openai_response(df, input_query, model_name='gpt-3.5-turbo-0301', temperature=0, callbacks=None):
    llm = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=st.secrets.get("OPENAI_API_KEY"))
    #df = load_file(file)
    #Pandas Dataframe Agent
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        max_iterations=5
         )

    #Agent Query
    response = agent(input_query, callbacks=callbacks)
    return response

#Frontend Logic
st.set_page_config(page_icon="chat2vis.png",layout="wide", page_title="Chat2DataInsight", initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center; font-weight:bold; font-family:comic sans ms; padding-top: 0rem;'> \
            Chat2DataInsight</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;padding-top: 0rem;'>Creating Visualisations and Data Analysis using Natural Language \
            with ChatGPT</h2>", unsafe_allow_html=True)

with st.sidebar:
     model_name = st.selectbox("Model: ", available_models.keys())

uploaded_file = st.file_uploader(":computer: Choose a file", accept_multiple_files=False, type=['csv'])
if uploaded_file is not None:
    #read file to dataframe in dataset session state
    st.session_state["dataset"] = load_file(uploaded_file)

if st.session_state.get("dataset") is not None:
    st.header('Output')

prompt = st.text_area(":eyes: What would you like to analyze?",height=10)
btn_go = st.button("Go...", use_container_width=True)
if prompt and btn_go:
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        df = st.session_state.get("dataset").copy()
        response = generate_openai_response(df, prompt, model_name=available_models.get(model_name), callbacks=[st_cb])
        st.write(response.get("output"))
        if len(plt.get_fignums()) > 0:
            fig = plt.gcf()
            st.pyplot(fig=fig, clear_figure=True)
