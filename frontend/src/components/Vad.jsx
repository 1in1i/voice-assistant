import React, { useState } from "react";
import { useMicVAD, utils } from "@ricky0123/vad-react";

export default function Vad() {
  const [userText, setUserText] = useState("");
  const [agentText, setAgentText] = useState("");
  const [loading, setLoading] = useState(false);

  const vad = useMicVAD({
    model: "v5",
    baseAssetPath: "/",
    onnxWASMBasePath: "/",

    onSpeechEnd: async (audioData) => {
      if (loading) return;

      setLoading(true);

      try {
        // PCM -> WAV
        const wavBuffer = utils.encodeWAV(audioData);

        // WAV -> Blob
        const blob = new Blob([wavBuffer], {
          type: "audio/wav",
        });

        // FormData
        const form = new FormData();
        form.append("file", blob, "input.wav");

        // FastAPI
        const res = await fetch("http://127.0.0.1:8000/chat", {
          method: "POST",
          body: form,
        });

        const data = await res.json();

        setUserText(data.user_text);
        setAgentText(data.response);

        const player = new Audio(
          `http://127.0.0.1:8000${data.audio_response_path}?t=${Date.now()}`,
        );

        await player.play();

        player.onended = () => {
          setLoading(false);
        };
      } catch (err) {
        console.error("Error:", err);
        setLoading(false);
      }
    },
  });

  return (
    <div style={{ padding: 20 }}>
      <h1>AI Voice Assistant</h1>

      <button onClick={vad.toggle}>{vad.listening ? "Stop" : "Start"}</button>

      <div style={{ marginTop: 10 }}>
        {vad.userSpeaking ? "User Speaking" : "User Silent"}
      </div>

      {loading && <div style={{ marginTop: 10 }}>Thinking...</div>}

      <hr />

      <p>
        <b>You:</b> {userText}
      </p>

      <p>
        <b>AI:</b> {agentText}
      </p>
    </div>
  );
}
