from faster_whisper import  WhisperModel

class SpeechToText:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        self.model = WhisperModel(model_size, device=device, cpu_threads=4, num_workers=2,compute_type=compute_type)

    def transcribe(self, audio_path):
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=1,
            temperature=0.0,
            vad_filter=True,
            language="de")

        text_list = [segment.text for segment in segments]
        return "".join(text_list).strip()
