import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
import tiktoken

# -----------------------------
# 1) 自定义回调 Handler，用于流式逐字渲染
# -----------------------------
class StreamlitStreamHandler(BaseCallbackHandler):
    """
    当 LLM 有新的 token 生成时，会调用 on_llm_new_token 回调。
    我们在这里将新token实时添加到一个 placeholder（占位符）里，
    实现像 ChatGPT 一样 “边生成边显示” 的效果。
    """
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder  # st.empty() 或其他可写组件
        self.streamed_text = ""         # 用于拼接逐字生成的文本

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # 每当 LLM 产生一个新的 token（可能是一段文字、一个符号），就会被回调到这里
        self.streamed_text += token
        # 用 markdown 显示当前进度
        self.placeholder.markdown(self.streamed_text)


# -----------------------------
# 2) 计算 token 数的辅助函数
# -----------------------------
def count_tokens(text: str, model_name="deepseek-chat") -> int:
    """
    尝试根据模型名称获取相应的 tiktoken 编码器，如不支持则使用 cl100k_base 。
    对 DeepSeek 并不一定准确，但可提供大致参考。
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


# -----------------------------
# 3) 从 secrets.toml 读取 API Key
# -----------------------------
openai_api_key = st.secrets["openai"]["api_key"]

# -----------------------------
# 4) 创建支持流式的 LLM 对象
# -----------------------------
# 注意：如果 deepseek-chat 不支持 streaming=True，可能会报错或无法流式
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="deepseek-chat",
    openai_api_base='https://api.deepseek.com',
    temperature=0.7,
    max_tokens=1024,
    streaming=True,   # 打开流式模式
    callbacks=None,   # 临时设为空，之后会在具体调用时传入
)

# -----------------------------
# 5) 初始化多轮对话存储
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.tokens = []
    # 先放一个 system 消息
    sys_msg = SystemMessage(content="你是一个乐于助人的AI助手。")
    st.session_state.messages.append(sys_msg)
    st.session_state.tokens.append(count_tokens(sys_msg.content))

st.title("DeepSeek Chat - 改进版流式输出示例")

# -----------------------------
# 6) 先把已有对话重放
# -----------------------------
for i, (msg, tcount) in enumerate(zip(st.session_state.messages, st.session_state.tokens)):
    # 跳过 system 消息时要注意不要忘记其 tokens
    if i == 0 and isinstance(msg, SystemMessage):
        # 你也可以选择把系统消息显示给用户
        continue

    if isinstance(msg, HumanMessage):
        user_bubble = st.chat_message("user")
        user_bubble.write(msg.content)
        user_bubble.write(
            f"<p style='color:gray;font-size:0.8rem;'>[用户消耗 {tcount} tokens]</p>",
            unsafe_allow_html=True
        )
    elif isinstance(msg, AIMessage):
        ai_bubble = st.chat_message("assistant")
        ai_bubble.write(msg.content)
        ai_bubble.write(
            f"<p style='color:gray;font-size:0.8rem;'>[AI消耗 {tcount} tokens]</p>",
            unsafe_allow_html=True
        )

# -----------------------------
# 7) 接收本轮用户输入
# -----------------------------
prompt = st.chat_input("请输入内容...")

# -----------------------------
# 8) 如果用户本轮输入不为空，执行对话
# -----------------------------
if prompt:
    # 8.1 把用户消息存到 session_state 并渲染
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    
    user_token_count = count_tokens(prompt)
    st.session_state.tokens.append(user_token_count)

    # 显示用户聊天气泡
    user_bubble = st.chat_message("user")
    user_bubble.write(prompt)
    user_bubble.write(
        f"<p style='color:gray;font-size:0.8rem;'>[用户消耗 {user_token_count} tokens]</p>",
        unsafe_allow_html=True
    )

    # 8.2 创建一个AI的聊天气泡，用于“流式”输出
    #     注意要先拿到它的引用(placeholder) 才能持续更新
    ai_bubble = st.chat_message("assistant")
    placeholder = ai_bubble.empty()  # 准备一个空的占位符

    # 建立回调handler，让每个新token都写到 placeholder
    stream_handler = StreamlitStreamHandler(placeholder)

    # 调用 LLM，传入全部消息 + 回调
    # 如果 deepseek-chat 不支持流式，这里会一次性返回
    ai_response = llm(
        st.session_state.messages,
        callbacks=[stream_handler],  # 传入我们的流式回调
    )

    # 8.3 拿到最终完整文本 & 记录
    final_text = ai_response.content

    ai_token_count = count_tokens(final_text)
    st.session_state.messages.append(AIMessage(content=final_text))
    st.session_state.tokens.append(ai_token_count)

    # 最后再补一行，给 token 消耗做个灰色说明
    placeholder.write(
        f"<p style='color:gray;font-size:0.8rem;'>[AI消耗 {ai_token_count} tokens]</p>",
        unsafe_allow_html=True
    )
