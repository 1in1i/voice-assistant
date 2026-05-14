import streamlit as st
import os
from streamlit_mic_recorder import mic_recorder
from langchain_community.chat_message_histories import ChatMessageHistory
from RAG import RAG
from STT_module import SpeechToText
from TTS_module import TextToSpeech

@st.cache_resource
def load_services():
    return RAG(), SpeechToText(model_size="small", device="cpu"), TextToSpeech()

rag_service, stt_service, tts_service = load_services()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()


for message in st.session_state.chat_history.messages:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(message.content)

st.write("---")
footer = st.container()
with footer:
    audio_data = mic_recorder(
        start_prompt="Record",
        stop_prompt="Stop",
        key='voice_input'
    )

if audio_data:
    temp_file = "temp_voice.wav"
    with open(temp_file, "wb") as f:
        f.write(audio_data['bytes'])

    with st.spinner("Listening..."):
        user_text = stt_service.transcribe(temp_file)

    os.remove(temp_file)

    if user_text:
        with st.chat_message("user"):
            st.write(user_text)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_service.ask(
                    user_text,
                    st.session_state.chat_history.messages
                )

                st.write(response)

            st.session_state.chat_history.add_user_message(user_text)
            st.session_state.chat_history.add_ai_message(response)


            audio_response_path = tts_service.speak(response)
            st.audio(audio_response_path, format="audio/wav", autoplay=True)

    # else:
    # st.warning("Please speak louder or more clearly!")