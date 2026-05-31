import os
from piper import PiperVoice


class TextToSpeech:
    def __init__(self, model_name="de_DE-eva_k-x_low.onnx"):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.static_dir = os.path.join(self.base_path, "static")
        self.model_path = os.path.join(self.static_dir, model_name)

        config_path = self.model_path + ".json"
        if not os.path.exists(self.model_path):
            print(f"Error: Path doesn't exist {self.model_path}")

        self.voice = PiperVoice.load(self.model_path, config_path=config_path)

    def speak_stream(self, text_generator):

        print("--- TTS STREAM START ---")
        try:
            for text_chunk in text_generator:
                if not text_chunk.strip():
                    continue

                for chunk in self.voice.synthesize(text_chunk):
                    yield chunk.audio_int16_bytes
        except Exception as e:
            print(f"Error during TTS streaming: {e}")
