import os
import subprocess


class TextToSpeech:

    def __init__(self, model_name="en_US-amy-low.onnx"):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.static_dir = os.path.join(self.base_path, "static")
        self.model_path = os.path.join(self.static_dir, model_name)

        if not os.path.exists(self.model_path):
            print(f"Error: Path doesn't exist {self.model_path}")

    def speak(self, text, output_file="response.wav"):
        print("--- TTS START ---")
        output_path = os.path.join(self.static_dir, output_file)

        try:
            command = ['piper', '--model', self.model_path, '--output_file', output_path]
            process = subprocess.run(
                command,
                input=text.encode('utf-8'),
                check=True,
                capture_output=True
            )

            print(f"TTS Success: {output_path}")

            return f"/static/{output_file}"

        except Exception as e:
            print(f"TTS Error: {e}")
            return None