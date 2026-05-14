import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

from RAG import RAG
from STT_module import SpeechToText
from TTS_module import TextToSpeech
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    temp_input_path = "temp_input.wav"
    with open(temp_input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        # STT
        user_text = stt_service.transcribe(temp_input_path)
        os.remove(temp_input_path)
        # RAG
        response = rag_service.ask(user_text, chat_memory)
        chat_memory.append({"role": "user", "content": user_text})
        chat_memory.append({"role": "assistant", "content": response})

        if len(chat_memory) > 20:
            chat_memory = chat_memory[-20:]

        print(f"DEBUG: RAG response: {response}")
        # TTS
        audio_response_path = tts_service.speak(response)

        return {
            "user_text": user_text,
            "audio_response_path": audio_response_path,
            "response": response
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
