# 文件名：app.py

import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage


openai_api_key = st.secrets["openai"]["api_key"]


llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="deepseek-chat",
    openai_api_base='https://api.deepseek.com',
    temperature=0.7,
    max_tokens=1024  
)

st.title("我的DeepSeek Chat")

prompt = st.chat_input("输入你的问题...")

if prompt:
    # 在页面上显示用户发送的消息
    user_message = st.chat_message("user")
    user_message.write(prompt)

    # 用LangChain封装好的 ChatOpenAI 发消息
    ai_response = llm([HumanMessage(content=prompt)])  # 返回是一个 AIMessage
    ai_content = ai_response.content

    # 在页面上显示AI回复
    ai_message = st.chat_message("assistant")
    ai_message.write(ai_content)
