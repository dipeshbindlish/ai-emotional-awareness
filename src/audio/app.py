import gradio as gr
from src.audio.inference import predict_emotion

def infer_audio(audio_file):
    if audio_file is None:
        return "No audio provided", 0.0

    result = predict_emotion(audio_file)
    return result["emotion"], result["confidence"]


audio_interface = gr.Interface(
    fn=infer_audio,
    inputs=gr.Audio(
        sources=["upload", "microphone"],
        type="filepath",
        label="Speak or upload WAV"
    ),
    outputs=[
        gr.Label(label="Predicted Emotion"),
        gr.Number(label="Confidence")
    ],
    title="🎧 Audio Emotion Recognition",
    description="Wav2Vec2-based emotion detection from speech"
)

if __name__ == "__main__":
    audio_interface.launch()