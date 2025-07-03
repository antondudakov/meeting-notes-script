import json
import os
import subprocess
import signal
from datetime import datetime
import shutil

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

def summarize_text(text, sentences=5, provider="openai", model="gpt-3.5-turbo", language="en"):
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        client = openai.OpenAI(api_key=api_key)
        prompt = (
            "Summarize the following meeting transcript. "
            "- First, provide a concise summary in bullet points. "
            "- Then, list all action items separately, each with the responsible person (if mentioned). "
            f"The most probably meeting language code is {language}"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content.strip()
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

def switch_audio_sources(input_source=None, output_source=None):
    """Switch macOS audio input/output using SwitchAudioSource.

    Returns previous input and output device names so they can be restored.
    If the SwitchAudioSource utility is not available, nothing happens.
    """
    cmd = shutil.which("SwitchAudioSource")
    if not cmd:
        print(
            "SwitchAudioSource utility not found. Install with `brew install switchaudio-osx` for automatic switching."
        )
        return None, None

    def current(dev_type):
        try:
            out = subprocess.check_output([cmd, "-t", dev_type, "-c"])
            return out.decode().strip()
        except subprocess.CalledProcessError:
            return None

    prev_in = prev_out = None
    if input_source:
        prev_in = current("input")
        subprocess.run([cmd, "-t", "input", "-s", input_source])
    if output_source:
        prev_out = current("output")
        subprocess.run([cmd, "-t", "output", "-s", output_source])
    return prev_in, prev_out


def restore_audio_sources(prev_input=None, prev_output=None):
    cmd = shutil.which("SwitchAudioSource")
    if not cmd:
        return
    if prev_input:
        subprocess.run([cmd, "-t", "input", "-s", prev_input])
    if prev_output:
        subprocess.run([cmd, "-t", "output", "-s", prev_output])

def main():
    cfg = load_config(CONFIG_PATH)
    os.makedirs(cfg.get("output_dir", "output"), exist_ok=True)

    prev_in, prev_out = switch_audio_sources(
        cfg.get("input_source"), cfg.get("output_source")
    )
    try:
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
    finally:
        restore_audio_sources(prev_in, prev_out)

    if not cfg.get("keep_audio", True):
        try:
            os.remove(audio_file)
            print(f"Deleted audio file {audio_file}")
        except OSError as e:
            print(f"Failed to delete audio file: {e}")

if __name__ == "__main__":
    main()
