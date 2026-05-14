# import os
# import pyttsx3
#
# class TextToSpeech:
#     def __init__(self, voice_id=None, rate=150):
#         self.engine = pyttsx3.init()
#         if voice_id:
#             self.engine.setProperty('voice', voice_id)
#         self.engine.setProperty('rate', rate)
#     def speak(self, text, output_file="response1.wav"):
#         print("--- TTS START ---")
#         self.engine.stop()
#         self.engine.save_to_file(text, output_file)
#         self.engine.runAndWait()
#         print(f"DEBUG: Audio saved to {os.path.abspath(output_file)}")
#         return output_file


import os
import subprocess
import shlex

class TextToSpeech:
    def __init__(self, model_name="en_US-amy-low.onnx"):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.static_dir = os.path.join(self.base_path, "static")
        self.model_path = os.path.join(self.base_path, model_name)

        if not os.path.exists(self.model_path):
            print(f"Error: Path doesn't exist {self.model_path}")

    def speak(self, text, output_file="response.wav"):
        print("--- TTS START ---")
        output_path = os.path.join(self.static_dir, output_file)

        # clean_text = text[:150].replace('"', '')

        try:
            command = f'echo "{text}" | piper --model "{self.model_path}" --output_file "{output_path}"'
            subprocess.run(command, shell=True, check=True, capture_output=True)
            print(f"TTS Success: {output_path}")

            return f"/static/{output_file}"

        except Exception as e:
            print(f"TTS Error: {e}")
            return None