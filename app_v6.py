import streamlit as st
from langchain.callbacks.manager import CallbackManager, AsyncCallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import tiktoken

class StreamlitStreamingCallbackHandler(BaseCallbackHandler):
    """
    每当大模型生成新的 token，就拼到 self.partial_text，实时更新到 Streamlit 前端。
    """
    def __init__(self, placeholder):
        self.placeholder = placeholder  # Streamlit 用来显示的占位符
        self.partial_text = ""         # 存放当前已经生成的所有文本
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # 把新 token 加到已有文本上
        self.partial_text += token
        # 实时更新到前端
        self.placeholder.write(self.partial_text)

# 计算 token 的函数
def count_tokens(text, model_name="deepseek-chat"):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

############################
# 2) Streamlit 主体逻辑
############################

# 从 secrets 中获取 API Key
openai_api_key = st.secrets["openai"]["api_key"]

# 若未初始化 session_state:
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="你是一个乐于助人的AI助手。")]

if "tokens" not in st.session_state:
    st.session_state.tokens = [0]  # 为系统消息占位

st.title("我的DeepSeek")

# 先回放历史消息
for i, msg in enumerate(st.session_state.messages):
    if i == 0 and isinstance(msg, SystemMessage):
        continue  # 不展示系统消息
    if isinstance(msg, HumanMessage):
        user_chat = st.chat_message("user")
        user_chat.write(msg.content)
        # 显示 token 消息
        user_chat.write(
            f"<p style='color:gray;font-size:0.8rem;'>[用户消耗 {st.session_state.tokens[i]} tokens]</p>",
            unsafe_allow_html=True
        )
    elif isinstance(msg, AIMessage):
        ai_chat = st.chat_message("assistant")
        ai_chat.write(msg.content)
        ai_chat.write(
            f"<p style='color:gray;font-size:0.8rem;'>[AI消耗 {st.session_state.tokens[i]} tokens]</p>",
            unsafe_allow_html=True
        )

# 输入框
prompt = st.chat_input("请输入内容...")

if prompt:
    # a) 记录用户消息
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)

    # 显示用户气泡
    user_chat = st.chat_message("user")
    user_chat.write(prompt)

    # 统计用户输入 tokens
    user_tokens = count_tokens(prompt)
    st.session_state.tokens.append(user_tokens)
    user_chat.write(
        f"<p style='color:gray;font-size:0.8rem;'>[用户消耗 {user_tokens} tokens]</p>",
        unsafe_allow_html=True
    )

    # b) AI流式输出
    #    - 先弄个 "assistant" 气泡
    #    - 然后内部放一个空 placeholder，等会儿实时写入
    assistant_chat = st.chat_message("assistant")
    stream_placeholder = assistant_chat.empty()

    # 创建回调管理器 + 我们的自定义流式回调
    callback_manager = CallbackManager([StreamlitStreamingCallbackHandler(stream_placeholder)])

    # 初始化 LLM 对象
    # 注意要 streaming=True，同时设置 callback_manager
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="deepseek-chat",
        openai_api_base='https://api.deepseek.com',
        temperature=0.7,
        max_tokens=1024,
        streaming=True,  # 关键：开启流式
        callback_manager=callback_manager
    )

    # 在 Streamlit 上给个提示转圈
    with st.spinner("AI 正在回复，请稍等..."):
        # 传入所有历史消息，让 AI 生成
        ai_response = llm(st.session_state.messages)
        ai_content = ai_response.content  # 生成结束后拿到完整文本

    # c) 记录 AI 消息到 session_state
    ai_msg = AIMessage(content=ai_content)
    st.session_state.messages.append(ai_msg)

    # 统计 AI 消耗 tokens（整段文本生成完后再数也行）
    ai_tokens = count_tokens(ai_content)
    st.session_state.tokens.append(ai_tokens)

    # （可选）如果你想在**生成完**后再更新一下贴上"AI消耗xx tokens"的标记
    # 因为我们的回调只写了文本，并没写token统计
    assistant_chat.write(
        f"<p style='color:gray;font-size:0.8rem;'>[AI消耗 {ai_tokens} tokens]</p>",
        unsafe_allow_html=True
    )
