import streamlit as st
from langchain_openai import ChatOpenAI  # deepseek-chat兼容
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import tiktoken

########################################
# 0) 页面设置 & CSS 美化
########################################
st.set_page_config(page_title="我的DeepSeek", layout="centered")

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

########################################
# 1) 置顶的“重置对话”按钮
########################################
if st.button("重置对话"):
    # 只需直接清空相关的 session_state 数据
    st.session_state.messages = [SystemMessage(content="你是一个乐于助人的AI助手。")]
    st.session_state.tokens = [0]
    st.session_state.stop_requested = False
    st.session_state.partial_text = ""  # 清空临时输出
    # 不做任何强制刷新或 st.stop，继续执行脚本即可

st.title("我的DeepSeek")
st.markdown("欢迎和我聊天")

########################################
# 2) 是否显示 token 数
########################################
SHOW_TOKENS = True  # 改成 False 即可隐藏 token 信息

########################################
# 3) 自定义异常 & 回调处理器
########################################
class StopStreamingException(Exception):
    """用户请求中止流式输出时抛出的异常。"""
    pass

class StreamlitStreamingCallbackHandler(BaseCallbackHandler):
    """自定义回调，用于在流式输出时检查停止标志、累计输出到 partial_text。"""
    def __init__(self, placeholder):
        self.placeholder = placeholder

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # 每生成一个token前都检测：若 stop_requested==True，则抛异常中断
        if st.session_state.get("stop_requested", False):
            raise StopStreamingException("用户请求中止流式输出。")
        # 追加新token到 partial_text
        st.session_state.partial_text += token
        # 在界面写出累计内容
        self.placeholder.write(st.session_state.partial_text)

########################################
# 4) token 统计函数
########################################
def count_tokens(text, model_name="deepseek-chat"):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

########################################
# 5) 初始化 session_state
########################################
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="你是一个乐于助人的AI助手。")]

if "tokens" not in st.session_state:
    st.session_state.tokens = [0]  # 对应上面SystemMessage

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

# 用于存放“流式生成中的临时文本”，防止停止后已输出的部分被清空
if "partial_text" not in st.session_state:
    st.session_state.partial_text = ""

########################################
# 6) 回放对话历史
########################################
for i, msg in enumerate(st.session_state.messages):
    if isinstance(msg, SystemMessage):
        continue  # 不展示系统消息
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
            if SHOW_TOKENS:
                st.write(
                    f"<p class='token-info'>[用户消耗 {st.session_state.tokens[i]} tokens]</p>",
                    unsafe_allow_html=True
                )
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
            if SHOW_TOKENS:
                st.write(
                    f"<p class='token-info'>[AI消耗 {st.session_state.tokens[i]} tokens]</p>",
                    unsafe_allow_html=True
                )

########################################
# 7) 函数：获取“系统消息 + 最近N条”
########################################
def get_model_context(messages, n=10):
    """返回：SystemMessage + 最后 n 条 (Human/AIMessage)。"""
    if len(messages) <= 1:
        return messages
    system_msg = messages[0]
    conv_msgs = messages[1:]
    if len(conv_msgs) <= n:
        return [system_msg] + conv_msgs
    else:
        return [system_msg] + conv_msgs[-n:]

########################################
# 8) 用户输入
########################################
prompt = st.chat_input("请输入内容...")

if prompt:
    # 每次新输入时重置停止标志 & partial_text
    st.session_state.stop_requested = False
    st.session_state.partial_text = ""

    # (a) 保存用户消息
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    user_tokens = count_tokens(prompt)
    st.session_state.tokens.append(user_tokens)

    # (b) 在界面显示用户消息
    with st.chat_message("user"):
        st.write(prompt)
        if SHOW_TOKENS:
            st.write(
                f"<p class='token-info'>[用户消耗 {user_tokens} tokens]</p>",
                unsafe_allow_html=True
            )

    # (c) 只传“系统消息 + 最近3条”给模型
    context_for_llm = get_model_context(st.session_state.messages, n=3)

    # (d) 流式输出 AI 回复
    with st.chat_message("assistant"):
        stream_placeholder = st.empty()  # 用于承载流式文本
        btn_container = st.empty()       # 用于放置“停止输出”按钮

        callback_manager = CallbackManager([
            StreamlitStreamingCallbackHandler(stream_placeholder)
        ])
        llm = ChatOpenAI(
            openai_api_key=st.secrets["openai"]["api_key"],
            model_name="deepseek-chat",
            openai_api_base='https://api.deepseek.com',
            temperature=0.7,
            max_tokens=1024,
            streaming=True,
            callback_manager=callback_manager
        )

        try:
            with st.spinner("AI 正在思考..."):
                # 在模型调用前先渲染“停止输出”按钮
                stop_button = btn_container.button("停止输出")
                if stop_button:
                    st.session_state.stop_requested = True

                # 真正开始调用模型，流式生成
                ai_response = llm(context_for_llm)
                ai_content = ai_response.content

        except StopStreamingException:
            # 若中断，则只保留现有partial_text
            ai_content = st.session_state.partial_text

        # 回复完成后，清空停止按钮
        btn_container.empty()

    # (e) 将最终 AI 内容存入会话
    ai_msg = AIMessage(content=ai_content)
    st.session_state.messages.append(ai_msg)
    ai_tokens = count_tokens(ai_content)
    st.session_state.tokens.append(ai_tokens)

    # (f) 最后把完整内容 & token 数写回 placeholder
    if SHOW_TOKENS:
        stream_placeholder.write(
            f"{ai_content}\n\n<p class='token-info'>[AI消耗 {ai_tokens} tokens]</p>",
            unsafe_allow_html=True
        )
    else:
        stream_placeholder.write(ai_content)
