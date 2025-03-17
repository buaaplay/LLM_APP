# 文件名：app.py

import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import tiktoken

# --- 辅助函数：计算 token 数量 ---
def count_tokens(text, model_name="deepseek-chat"):
    """
    利用 tiktoken 计算给定文本的 token 数量。
    如果指定模型的编码不可用，则使用默认的编码。
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception as e:
        encoding = tiktoken.get_encoding("cl100k_base")  # fallback编码
    return len(encoding.encode(text))

# --- 配置 API Key 与初始化 ChatOpenAI ---
openai_api_key = st.secrets["openai"]["api_key"]

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="deepseek-chat",
    openai_api_base='https://api.deepseek.com',
    temperature=0.7,
    max_tokens=1024  
)

# --- 初始化对话历史和 token 记录 ---
if "messages" not in st.session_state:
    # SystemMessage 用来设定对话的整体身份和角色约束
    st.session_state.messages = [SystemMessage(content="你是一个有帮助的AI助手。")]
    # 记录每轮对话的 token 消耗（元组形式：(用户token, AI token)）
    st.session_state.token_usage = []

# --- Streamlit 页面 ---
st.title("我的DeepSeek Chat (上下文对话)")

# 用户输入
prompt = st.chat_input("输入你的问题...")

if prompt:
    # 将用户消息加入历史
    st.session_state.messages.append(HumanMessage(content=prompt))
    # 计算用户输入的 token 数量
    user_token_count = count_tokens(prompt)

    # 在页面上显示用户消息
    user_message = st.chat_message("user")
    user_message.write(prompt)

    # 调用模型，将完整对话历史传入，实现多轮上下文对话
    ai_response = llm(st.session_state.messages)
    ai_content = ai_response.content

    # 将 AI 回复加入历史
    st.session_state.messages.append(AIMessage(content=ai_content))
    # 计算 AI 回复的 token 数量
    ai_token_count = count_tokens(ai_content)

    # 在页面上显示 AI 回复
    ai_message = st.chat_message("assistant")
    ai_message.write(ai_content)

    # 显示本轮对话消耗的 token 数量
    st.write(f"本轮消耗 token 数：用户输入 {user_token_count} tokens，AI回复 {ai_token_count} tokens，总计 {user_token_count + ai_token_count} tokens。")
    
    # 将本轮 token 记录保存到 session_state
    st.session_state.token_usage.append((user_token_count, ai_token_count))
