import streamlit as st
from streamlit_mic_recorder import mic_recorder
from STT_module import SpeechToText
import io
import os

# --- 页面配置 ---
st.set_page_config(page_title="语音录制测试", page_icon="🎤")
st.title("🎤 语音识别 (STT) 压力测试")
st.write("点击下方按钮开始录音，说出关于“年糕”的话，看看识别准不准！")


# --- 加载模型 ---
@st.cache_resource
def init_stt():
    # 使用你之前写的 stt_module
    return SpeechToText(model_size="small", device="cpu")


stt_service = init_stt()

# --- 录音组件 ---
# start_prompt: 开始按钮文字, stop_prompt: 结束按钮文字
audio_data = mic_recorder(
    start_prompt="开始录音 🎙️",
    stop_prompt="停止录音 ⏹️",
    key='recorder'
)

# --- 处理录音数据 ---
if audio_data:
    # 1. 提取录音字节
    audio_bytes = audio_data['bytes']

    # 2. 为了交给 Whisper，我们需要先存为临时文件
    temp_filename = "temp_record.wav"
    with open(temp_filename, "wb") as f:
        f.write(audio_bytes)

    st.audio(audio_bytes, format='audio/wav')  # 页面回放刚才录的声音

    # 3. 调用 Whisper 进行识别
    with st.spinner("正在努力听清楚..."):
        try:
            # 这里会用到你加了 initial_prompt 的 transcribe 函数
            text = stt_service.transcribe(temp_filename)

            st.success("识别完成！")
            st.subheader("识别出的文字：")
            st.info(text if text else "没听清，再大声点？")

        except Exception as e:
            st.error(f"识别出错: {e}")
        finally:
            # 识别完删掉临时文件，保持项目整洁
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

