import json
import os
import subprocess
import signal
from datetime import datetime

try:
    import openai
except ImportError:
    raise SystemExit("openai is required. Install with `pip install openai`")

try:
    import whisper
except ImportError:
    raise SystemExit("whisper is required. Install with `pip install -U openai-whisper`")


CONFIG_PATH = os.environ.get("MEETING_SETTINGS", "settings.json")

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def record_audio(output_file, device, duration=None):
# ffmpeg -f avfoundation -i ":0" -ar 48000 -ac 2 -sample_fmt s16
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "avfoundation",
        "-i",
        f":{device}",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-sample_fmt",
        "s16"
    ]

    print(f"{cmd}")

    if duration and duration > 0:
        cmd += ["-t", str(duration)]
    cmd.append(output_file)
    proc = subprocess.Popen(cmd)
    print("Press Ctrl+C to stop recording ...")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait()

def transcribe_audio(audio_path, model_size="base", language="en"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language=language)
    return result["text"].strip()

def summarize_text(text, sentences=5, provider="openai", model="gpt-3.5-turbo"):
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        openai.api_key = api_key
        prompt = (
            "Summarize the following meeting transcript into "
            f"{sentences} concise bullet points."
        )
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )
        return response["choices"][0]["message"]["content"].strip()
    elif provider == "gemini":
        try:
            import google.generativeai as genai
        except ImportError:
            raise SystemExit("google-generativeai is required. Install with `pip install google-generativeai`")
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        prompt = (
            "Summarize the following meeting transcript into "
            f"{sentences} concise bullet points."
        )
        model_name = model or "gemini-pro"
        generative_model = genai.GenerativeModel(model_name)
        response = generative_model.generate_content(prompt + "\n" + text)
        return response.text.strip()
    else:
        raise ValueError("Unknown provider for summarization")

def save_output(text, path, fmt="text"):
    if fmt == "markdown":
        text = text.replace('\n', '  \n')
    with open(path, 'w') as f:
        f.write(text)

def main():
    cfg = load_config(CONFIG_PATH)
    os.makedirs(cfg.get("output_dir", "output"), exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file = os.path.join(cfg.get("output_dir", "output"), f"recording_{ts}.wav")
    print(f"Recording audio to {audio_file} ...")
    record_audio(
        audio_file,
        cfg.get("audio_device", "0"),
        cfg.get("duration_seconds", 60),
    )

    print("Transcribing ...")
    lang = cfg.get("language", "en")
    transcript = transcribe_audio(
        audio_file,
        cfg.get("transcription_model", "base"),
        lang
    )
    transcript_path = os.path.join(cfg.get("output_dir", "output"), f"transcript_{ts}.txt")
    save_output(transcript, transcript_path, "text")
    print(f"Transcript saved to {transcript_path}")

    print("Summarizing transcript with LLM ...")
    provider = cfg.get("llm_provider", "openai")
    model_key = "openai_model" if provider == "openai" else "gemini_model"
    notes = summarize_text(
        transcript,
        cfg.get("summary_sentences", 5),
        provider,
        cfg.get(model_key, "gpt-3.5-turbo" if provider == "openai" else "gemini-pro"),
    )
    notes_path = os.path.join(cfg.get("output_dir", "output"), f"notes_{ts}.md" if cfg.get("output_format", "text") == "markdown" else f"notes_{ts}.txt")
    save_output(notes, notes_path, cfg.get("output_format", "text"))
    print(f"Notes saved to {notes_path}")

if __name__ == "__main__":
    main()
