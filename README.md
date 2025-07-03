# Meeting Notes Script

This repository provides a simple Python script for recording a meeting on macOS, transcribing the audio locally, and generating AI-assisted notes. The entire process runs on your machine.

## Requirements

- macOS with `ffmpeg` installed (via Homebrew or other method).
- Python 3.11+
- Python packages: `openai-whisper`, `openai`, `numpy`, `nltk`, `google-generativeai`.
- A virtual audio device such as **BlackHole** is required if you want to capture sound coming from the browser. Without it the script only records your microphone.
- [`switchaudio-osx`](https://github.com/deweller/switchaudio-osx) is required for automatic switching of input/output devices (optional).

## Installation

1. Install ffmpeg:
   ```bash
   brew install ffmpeg
   ```
2. Install the command-line tool for switching audio devices (optional but recommended for automatic switching):
   ```bash
   brew install switchaudio-osx
   ```
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   The remaining commands assume the virtual environment is active.
4. Install Python dependencies (run inside the virtual environment):
   ```bash
   pip install -U openai-whisper openai numpy nltk google-generativeai
   ```
   Download NLTK data for tokenization:
   ```bash
   python -m nltk.downloader punkt punkt_tab
   ```

## Usage

1. Adjust `settings.json` to configure recording duration, output format, and other options.
2. Run the script:
   ```bash
   python meeting_notes.py
   ```
   Press **Ctrl+C** to stop recording before the configured maximum duration if needed.
3. The script records your meeting, transcribes the audio with Whisper, and summarizes the transcript using an LLM (OpenAI or Gemini). Files are saved in the directory specified by `output_dir` in `settings.json`.

## Configuration

`settings.json` options:

- `audio_device` – device index for ffmpeg (macOS `avfoundation`).
- `input_source` – name of the macOS input device to select before recording.
- `output_source` – name of the macOS output device to select before recording.
- `duration_seconds` – max recording length in seconds. Set to `0` to record until you press **Ctrl+C**.
- `transcription_model` – Whisper model size (e.g., `tiny`, `base`, `small`).
- `language` – ISO language code used by Whisper (e.g., `ru`, `en`).
- `summary_sentences` – number of sentences to include in the generated notes.
- `output_format` – `text` or `markdown` for notes output.
- `llm_provider` – `openai` or `gemini` for summarization.
- `openai_model` – OpenAI chat model to use (default `gpt-3.5-turbo`).
- `gemini_model` – Gemini model name (default `gemini-pro`).
- `output_dir` – folder where audio, transcripts, and notes are stored.
- `keep_audio` – set to `false` to delete the recording after transcription.

## Capturing Browser and Microphone Audio with BlackHole

1. Install the virtual audio driver (2‑channel version is enough for meetings):
   ```bash
   brew install blackhole-2ch
   ```
   Alternatively download the installer from
   [BlackHole on GitHub](https://github.com/ExistentialAudio/BlackHole).
2. Open **Audio MIDI Setup** and create two devices:
   - A **Multi-Output Device** containing your usual speakers/headphones and
     **BlackHole**. Set this as the macOS output so you can hear the meeting.
   - An **Aggregate Device** that combines your microphone with **BlackHole**.
3. In the browser or meeting app, select the Aggregate Device as the microphone
   and choose the Multi-Output Device for output.
4. List available devices to find the index of the Aggregate Device:
   ```bash
   ffmpeg -f avfoundation -list_devices true -i ""
   ```
   Use the displayed index for `audio_device` in `settings.json`.
   Recording will then include audio from the browser and your microphone.

## Notes

- Capturing system audio from Google Meet may require a virtual audio driver such as BlackHole. Select the device in `settings.json`.
- Whisper models are downloaded automatically when first used; make sure you have sufficient disk space.
- Set `language` to `ru` in `settings.json` to generate transcripts and summaries in Russian.
- Set the `OPENAI_API_KEY` environment variable to use OpenAI-based summarization.
- Set the `GOOGLE_API_KEY` environment variable to use Gemini-based summarization.

