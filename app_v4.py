import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import tiktoken

# 辅助函数：计算 token 数量，并返回整数
def count_tokens(text, model_name="deepseek-chat"):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# 从 Streamlit Secrets 中获取 API Key（避免写在代码里）
openai_api_key = st.secrets["openai"]["api_key"]

# 初始化 ChatOpenAI 对象
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="deepseek-chat",
    openai_api_base='https://api.deepseek.com',
    temperature=0.7,
    max_tokens=1024
)

if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="你是一个乐于助人的AI助手。")]
if "tokens" not in st.session_state:
    st.session_state.tokens = [0]  # 为系统消息添加占位 token


st.title("我的DeepSeek")

# 1) 首先把已经有的对话显示出来（不包括最后一条未回答的用户消息）
#    注意：Streamlit 每次交互都会从头运行脚本，所以要根据 session_state
#    中保存的对话顺序来“重放”消息。

for i, msg in enumerate(st.session_state.messages):
    # 跳过第0个SystemMessage，只展示人类 & AI
    if i == 0 and isinstance(msg, SystemMessage):
        continue
    
    # 根据是 HumanMessage 还是 AIMessage 渲染不同角色
    if isinstance(msg, HumanMessage):
        user_chat = st.chat_message("user")
        user_chat.write(msg.content)
        # 在用户消息下方显示灰色 token 信息
        # 这里 st.session_state.tokens[i] 是这个 HumanMessage 的 token 数
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

# 2) 新输入框（最新版 Streamlit 用 st.chat_input）
prompt = st.chat_input("请输入内容...")

# 如果用户在本轮输入了新内容
if prompt:
    # （a）先把用户消息立刻存到 session_state，并且当场显示到页面上
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    
    # 显示用户的聊天气泡
    user_chat = st.chat_message("user")
    user_chat.write(prompt)
    
    # 计算用户输入的 token 数量并存储
    user_tokens = count_tokens(prompt)
    st.session_state.tokens.append(user_tokens)
    
    # 在用户消息下方显示灰色 token 信息
    user_chat.write(
        f"<p style='color:gray;font-size:0.8rem;'>[用户消耗 {user_tokens} tokens]</p>",
        unsafe_allow_html=True
    )
    
    # （b）现在要生成 AI 回复，并立刻给用户“正在思考中”的感受
    with st.chat_message("assistant"):
        with st.spinner("AI 正在回复，请稍等..."):
            # 调用 LLM，传入所有消息（包括系统消息 + 历史用户/AI消息 + 当前这条用户消息）
            ai_response = llm(st.session_state.messages)
            ai_content = ai_response.content
            
            # 把 AI 回复保存到 session_state
            ai_msg = AIMessage(content=ai_content)
            st.session_state.messages.append(ai_msg)
            
            # 计算 AI 的 token 数量并存储
            ai_tokens = count_tokens(ai_content)
            st.session_state.tokens.append(ai_tokens)
            
            # 显示 AI 实际回复
            st.write(ai_content)
            st.write(
                f"<p style='color:gray;font-size:0.8rem;'>[AI消耗 {ai_tokens} tokens]</p>",
                unsafe_allow_html=True
            )
