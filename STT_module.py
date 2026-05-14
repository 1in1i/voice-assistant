import os
from faster_whisper import  WhisperModel

class SpeechToText:
    def __init__(self, model_size="tiny", device="cpu", compute_type="int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path):
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            language="en",
            initial_prompt="This is a conversation about an art university called HBKsaar in Germany.")

        transcription = ""
        for segment in segments:
            transcription += segment.text + " "
        return transcription.strip()