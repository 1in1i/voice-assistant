import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import time
from RAG import RAG
from STT_module import SpeechToText
from TTS_module import TextToSpeech
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stt_service = SpeechToText(model_size="small", device="cpu")
rag_service = RAG()
tts_service = TextToSpeech()

chat_memory = []
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/chat")
async def chat(file: UploadFile = File(...)):
    global chat_memory
    t_total_start = time.perf_counter()
    temp_input_path = "temp_input.wav"
    with open(temp_input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        # STT
        t_stt_start = time.perf_counter()
        user_text = stt_service.transcribe(temp_input_path)
        t_stt = time.perf_counter() - t_stt_start
        os.remove(temp_input_path)
        # RAG
        t_llm_start = time.perf_counter()
        response = rag_service.ask(user_text, chat_memory)
        t_llm = time.perf_counter() - t_llm_start
        chat_memory.append({"role": "user", "content": user_text})
        chat_memory.append({"role": "assistant", "content": response})

        if len(chat_memory) > 20:
            chat_memory = chat_memory[-20:]

        print(f"DEBUG: RAG response: {response}")
        # TTS
        t_tts_start = time.perf_counter()
        audio_response_path = tts_service.speak(response)
        t_tts = time.perf_counter() - t_tts_start
        #=========
        t_total = time.perf_counter() - t_total_start

        print("\n===== PIPELINE TIME =====")
        print(f"STT : {t_stt:.3f}s")
        print(f"LLM : {t_llm:.3f}s")
        print(f"TTS : {t_tts:.3f}s")
        print(f"TOTAL : {t_total:.3f}s")
        print("=========================\n")
        #========
        return {
            "user_text": user_text,
            "audio_response_path": audio_response_path,
            "response": response
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
