import json
import os
import subprocess
import signal
from datetime import datetime
import shutil
import platform
import urllib.parse
import argparse
import re
import torch

try:
    import openai
except ImportError:
    raise SystemExit("openai is required. Install with `pip install openai`")

try:
    import pycountry
except ImportError:
    pycountry = None

try:
    import whisper
except ImportError:
    raise SystemExit("whisper is required. Install with `pip install -U openai-whisper`")


CONFIG_PATH = os.environ.get("MEETING_SETTINGS", "settings.json")


def language_name(code):
    """Return the full language name for an ISO 639-1 code."""
    if pycountry:
        lang = pycountry.languages.get(alpha_2=code)
        if lang and hasattr(lang, "name"):
            return lang.name
    return code

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def sanitize_name(name):
    """Return a filesystem-safe version of ``name``."""
    if not name:
        return None
    safe = re.sub(r"[^\w\-. ]", "_", name).strip()
    return safe or None


def get_calendar_event_title():
    """Return the title of the current Calendar event on macOS, if any."""
    if platform.system() != "Darwin":
        return None

    cmd = shutil.which("icalBuddy")
    if not cmd:
        return None

    try:
        out = subprocess.check_output(
            [cmd, "-n", "-eep", "*", "-b", "", "-ea", "-nc", "eventsNow"], timeout=5
        )

        line = out.decode("utf-8").strip()

        if line:
            all_events = line.split('\n')
            last = all_events[-1]

            return last.lstrip("•").strip()
            
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"Failed to read Calendar event via icalBuddy: {exc}")
    return None

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

def transcribe_audio(
    audio_path,
    model_size="base",
    language="en",
    backend="whisper",
    whispercpp_bin=None,
    whispercpp_model=None,
):
    """Transcribe ``audio_path`` using the selected backend.

    ``backend`` can be ``"whisper"`` (default) or ``"whispercpp"``.
    """

    if backend == "whispercpp":
        binary = whispercpp_bin or "whisper-cli"
        model_path = whispercpp_model or f"models/ggml-{model_size}.bin"
        cmd = [
            binary,
            "-m",
            model_path,
            "-f",
            audio_path,
            "-l",
            language,
            "-nt",
            "-np",
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            raise SystemExit(
                "whisper-cli not found. Build whisper.cpp and set 'whispercpp_binary' in settings.json"
            )
        return out.decode("utf-8", errors="replace").strip()

    device = "mps" if torch.has_mps else "cpu"
    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(audio_path, language=language)
    return result["text"].strip()

def summarize_text(text, sentences=5, provider="openai", model="gpt-3.5-turbo", language="en"):
    lang_name = language_name(language)
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        client = openai.OpenAI(api_key=api_key)
        prompt = (
            "Summarize the following meeting transcript. "
            "- First, provide a concise summary in bullet points. "
            "- Then, list all action items separately, each with the responsible person (if mentioned). "
            f"The most probable meeting language code is {lang_name}. "
            f"Write the answer in {lang_name}."
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
            f"{sentences} concise bullet points. "
            f"Write the answer in {lang_name}."
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

def open_file(path):
    """Open ``path`` with the default application for the OS."""
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", path])
        elif system == "Windows":
            os.startfile(path)
        else:
            subprocess.run(["xdg-open", path])
    except Exception as exc:  # pragma: no cover - user environment dependent
        print(f"Failed to open file {path}: {exc}")

def run_post_save_command(cmd, path, base_dir):
    """Run a shell command after saving notes.

    The command can use the following placeholders:
    ``{path}``          – absolute path to the notes file.
    ``{url}``           – URL-encoded absolute path.
    ``{relative}``      – path relative to the parent of ``base_dir``.
    ``{relative_url}``  – URL-encoded relative path.
    """

    url = urllib.parse.quote(path)
    relative = os.path.relpath(path, os.path.dirname(base_dir))
    relative_url = urllib.parse.quote(relative)
    try:
        subprocess.run(
            cmd.format(
                path=path,
                url=url,
                relative=relative,
                relative_url=relative_url,
            ),
            shell=True,
        )
    except Exception as exc:  # pragma: no cover - user environment dependent
        print(f"Failed to run post-save command: {exc}")

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

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="use existing audio file instead of recording",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(CONFIG_PATH)
    base_dir = cfg.get("output_dir", "output")
    os.makedirs(base_dir, exist_ok=True)

    prev_in, prev_out = switch_audio_sources(
        cfg.get("input_source"), cfg.get("output_source")
    )
    try:
        if args.audio_file:
            base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
            suffix = sanitize_name(base_name)
            meeting_dir = os.path.dirname(os.path.abspath(args.audio_file))
            audio_src = args.audio_file
            audio_file = args.audio_file
        else:
            suffix = datetime.now().strftime("%Y.%m.%d_%a_%H-%M")
            event_title = None
            if cfg.get("use_calendar_title"):
                event_title = sanitize_name(get_calendar_event_title())
            base_name = event_title or f"meeting_{suffix}"
            meeting_dir = os.path.join(base_dir, base_name)
            os.makedirs(meeting_dir, exist_ok=True)
            audio_file = os.path.join(meeting_dir, f"recording_{suffix}.wav")
            audio_src = audio_file
            print(f"Recording audio to {audio_file} ...")
            record_audio(
                audio_file,
                cfg.get("audio_device", "0"),
                cfg.get("duration_seconds", 60),
            )

        print("Transcribing ...")
        lang = cfg.get("language", "en")

        transcript = transcribe_audio(
            audio_src,
            cfg.get("transcription_model", "base"),
            lang,
            cfg.get("transcription_backend", "whisper"),
            cfg.get("whispercpp_binary"),
            cfg.get("whispercpp_model"),
        )

        print("Summarizing transcript with LLM ...")
        provider = cfg.get("llm_provider", "openai")
        model_key = "openai_model" if provider == "openai" else "gemini_model"

        notes_summary = summarize_text(
            transcript,
            cfg.get("summary_sentences", 5),
            provider,
            cfg.get(model_key, "gpt-3.5-turbo" if provider == "openai" else "gemini-pro"),
            cfg.get("language", "en"),
        )

        links = ""

        if cfg.get("keep_audio", True) and audio_file:
            links = f"[Audio]({os.path.basename(audio_file)})"

        notes_content = (
            notes_summary
            + "\n\n"
            + links
            + "\n\n## Transcript\n\n"
            + transcript
        )

        notes_filename = (
            f"notes_{suffix}.md"
            if cfg.get("output_format", "text") == "markdown"
            else f"notes_{suffix}.txt"
        )
        
        notes_path = os.path.join(meeting_dir, notes_filename)
        save_output(notes_content, notes_path, cfg.get("output_format", "text"))
        print(f"Notes saved to {notes_path}")

        if cfg.get("post_save_command"):
            run_post_save_command(
                cfg.get("post_save_command"),
                notes_path,
                base_dir,
            )
        elif cfg.get("open_notes", False):
            open_file(notes_path)
    finally:
        restore_audio_sources(prev_in, prev_out)

    if audio_file and not args.audio_file and not cfg.get("keep_audio", True):
        try:
            os.remove(audio_file)
            print(f"Deleted audio file {audio_file}")
        except OSError as e:
            print(f"Failed to delete audio file: {e}")

if __name__ == "__main__":
    main()
