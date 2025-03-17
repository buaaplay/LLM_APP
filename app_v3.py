# 文件名：app.py

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import tiktoken

# --- 辅助函数：计算 token 数量 ---
def count_tokens(text, model_name="deepseek-chat"):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception as e:
        encoding = tiktoken.get_encoding("cl100k_base")  # fallback编码
    return len(encoding.encode(text))

# --- 使用 Streamlit secrets 管理 API Key ---
# 在项目根目录下创建 .streamlit/secrets.toml 文件，内容示例：
# [openai]
# api_key = "sk-xxxxxxxxxxxxxxxxxxxx"
openai_api_key = st.secrets["openai"]["api_key"]

# --- 初始化 ChatOpenAI 对象 ---
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="deepseek-chat",
    openai_api_base='https://api.deepseek.com',
    temperature=0.7,
    max_tokens=1024  # 可根据需要调整，过大可能会延长响应时间
)

# --- 初始化对话历史与 token 记录 ---
if "messages" not in st.session_state:
    # 这里的 SystemMessage 用于定义 AI 的角色，可以根据需求修改
    st.session_state.messages = [SystemMessage(content="你是一个有帮助的AI助手。")]
    st.session_state.token_usage = []

# --- 页面标题 ---
st.title("我的DeepSeek Chat (上下文对话)")

# --- 对话记录容器 ---
conversation_container = st.container()

# --- 用户输入区域 ---
prompt = st.chat_input("输入你的问题...")

if prompt:
    # 将用户消息添加到对话历史，并计算 token 数量
    st.session_state.messages.append(HumanMessage(content=prompt))
    user_token_count = count_tokens(prompt)
    
    # 显示加载动画，避免界面长时间“灰屏”
    with st.spinner("AI 正在回复，请稍等..."):
        ai_response = llm(st.session_state.messages)
    ai_content = ai_response.content
    st.session_state.messages.append(AIMessage(content=ai_content))
    ai_token_count = count_tokens(ai_content)
    
    # 显示本轮对话的 token 消耗
    st.write(f"本轮消耗 token 数：用户输入 {user_token_count} tokens，AI回复 {ai_token_count} tokens，总计 {user_token_count + ai_token_count} tokens。")

# --- 始终显示完整的对话记录（不包括 SystemMessage） ---
with conversation_container:
    st.markdown("### 对话记录：")
    for msg in st.session_state.messages:
        if isinstance(msg, SystemMessage):
            continue  # 不显示系统消息
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
