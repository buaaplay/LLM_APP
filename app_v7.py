import streamlit as st
from langchain_openai import ChatOpenAI  # 你为了deepseek-chat兼容用这个
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import tiktoken

########################################
# 0) 配置 & CSS 美化
########################################
st.set_page_config(page_title="DeepSeek Chat (v7-修订)", layout="centered")

page_bg = """
<style>
body {
    background-color: #f7f8fa;
}
.block-container {
    max-width: 700px;  /* 页面宽度 */
}
.token-info {
    color: gray;
    font-size: 0.8rem;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("DeepSeek Chat (v7 修订版)")
st.markdown("演示 **只记住最近5条**（不删除旧消息，界面仍显示全部）。可选是否显示token数。")

########################################
# 1) 是否显示 token 数
########################################
SHOW_TOKENS = True  # 改成 False 即可隐藏所有 token 信息

########################################
# 2) 流式输出回调
########################################
class StreamlitStreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.partial_text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.partial_text += token
        self.placeholder.write(self.partial_text)

########################################
# 3) token 统计函数
########################################
def count_tokens(text, model_name="deepseek-chat"):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

########################################
# 4) SessionState 初始化
########################################
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="你是一个乐于助人的AI助手。")
    ]
if "tokens" not in st.session_state:
    # 跟 messages 同步长度，存每条消息的 token 数
    st.session_state.tokens = [0]  # 对应上面SystemMessage

########################################
# 5) 先回放所有历史对话
########################################
for i, msg in enumerate(st.session_state.messages):
    if isinstance(msg, SystemMessage):
        # 系统消息一般不展示
        continue
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
            # 如果开启显示 tokens
            if SHOW_TOKENS:
                st.write(f"<p class='token-info'>[用户消耗 {st.session_state.tokens[i]} tokens]</p>", unsafe_allow_html=True)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
            if SHOW_TOKENS:
                st.write(f"<p class='token-info'>[AI消耗 {st.session_state.tokens[i]} tokens]</p>", unsafe_allow_html=True)

########################################
# 6) 函数：获取只包括“最近N条”的对话，用于传给模型
########################################
def get_model_context(messages, n=10):
    """
    返回：包含SystemMessage + 最后n条(Human/AIMessage)
    """
    if len(messages) <= 1:
        return messages  # 只有SystemMessage，无需处理
    
    system_msg = messages[0]
    conversation_msgs = messages[1:]
    if len(conversation_msgs) <= n:
        return [system_msg] + conversation_msgs
    else:
        return [system_msg] + conversation_msgs[-n:]

########################################
# 7) 用户输入
########################################
prompt = st.chat_input("请输入内容...")

if prompt:
    # （a）先将用户消息保存
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    # 统计 tokens
    user_tokens = count_tokens(prompt)
    st.session_state.tokens.append(user_tokens)

    # （b）在界面显示
    with st.chat_message("user"):
        st.write(prompt)
        if SHOW_TOKENS:
            st.write(f"<p class='token-info'>[用户消耗 {user_tokens} tokens]</p>", unsafe_allow_html=True)

    # （c）对模型只传"系统消息 + 最近5条"，而不是全部消息
    context_for_llm = get_model_context(st.session_state.messages, n=3)

    # （d）开始流式回复
    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        callback_manager = CallbackManager([StreamlitStreamingCallbackHandler(stream_placeholder)])
        llm = ChatOpenAI(
            openai_api_key=st.secrets["openai"]["api_key"],
            model_name="deepseek-chat",
            openai_api_base='https://api.deepseek.com',
            temperature=0.7,
            max_tokens=1024,
            streaming=True,
            callback_manager=callback_manager
        )

        with st.spinner("AI 正在思考..."):
            ai_response = llm(context_for_llm)
            ai_content = ai_response.content

    # （e）将 AI 消息存储，并显示 tokens
    ai_msg = AIMessage(content=ai_content)
    st.session_state.messages.append(ai_msg)
    ai_tokens = count_tokens(ai_content)
    st.session_state.tokens.append(ai_tokens)

    # （f）可选再次输出 tokens
    if SHOW_TOKENS:
        stream_placeholder.write(
            f"{ai_content}\n\n<p class='token-info'>[AI消耗 {ai_tokens} tokens]</p>",
            unsafe_allow_html=True
        )
    else:
        # 如果不显示token，就直接替换流式的内容为整段
        stream_placeholder.write(ai_content)
