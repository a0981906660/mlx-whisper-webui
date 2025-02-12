"""STT (Speech-to-Text) Web UI with mlx-whisper and language selection"""

from functools import wraps
import gradio as gr
import mlx_whisper

HF_REPO = "mlx-community/distil-whisper-large-v3"

def main():
    iface = build_interface(hf_repo=HF_REPO)
    iface.launch()

def build_interface(hf_repo: str):
    # Define a wrapper function that accepts both audio and language from the UI.
    def transcribe_wrapper(audio: str | None, language: str) -> str | None:
        # If the user selects "Auto", set language to None so that the model auto-detects.
        if language.lower() == "auto":
            language_param = None
        else:
            language_param = language
        return transcribe(audio, hf_repo=hf_repo, language=language_param)

    iface = gr.Interface(
        fn=for_gradio(transcribe_wrapper),
        inputs=[
            gr.Audio(type="filepath", label="Upload Audio"),
            gr.Dropdown(
                choices=["Auto", "en", "zh", "ja"],
                value="Auto",
                label="Language"
            ),
        ],
        outputs=gr.Textbox(label="Transcription"),
        title="MLX-Whisper STT Web UI"
    )
    return iface

def for_gradio(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        try:
            return f(*args, **kwds)
        except Exception as err:
            raise gr.Error(str(err)) from err
    return wrapper

def transcribe(audio: str | None, hf_repo: str, language: str | None) -> str | None:
    if audio is None:
        return None

    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=hf_repo,
        language=language,  # If language is None, the model will auto-detect.
    )
    return result["text"]

if __name__ == "__main__":
    main()
