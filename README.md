# Meeting Notes Script

This repository provides a simple Python script for recording a meeting on macOS, transcribing the audio locally, and generating AI-assisted notes. The entire process runs on your machine.

## Requirements

- macOS with `ffmpeg` installed (via Homebrew or other method).
- Python 3.11+
 - Python packages: `openai-whisper`, `openai`, `numpy`, `nltk`, `google-generativeai`.
 - Compiled `whisper-cli` from [`ggml-org/whisper.cpp`](https://github.com/ggml-org/whisper.cpp) if you want to use the whispercpp backend.
- A virtual audio device such as **BlackHole** is required if you want to capture sound coming from the browser. Without it the script only records your microphone.
- [`switchaudio-osx`](https://github.com/deweller/switchaudio-osx) is required for automatic switching of input/output devices (optional).
- [`icalBuddy`](https://github.com/ali-rantakari/icalBuddy) is used to read the current Calendar event title when `use_calendar_title` is enabled (optional).

## Installation

1. Install ffmpeg:
   ```bash
   brew install ffmpeg
   ```
2. Install the command-line tool for switching audio devices (optional but recommended for automatic switching):
   ```bash
   brew install switchaudio-osx
   ```
3. Clone and build `whisper.cpp` if you plan to use the `whispercpp` backend:
   ```bash
   git clone https://github.com/ggml-org/whisper.cpp
   cd whisper.cpp
   make -j
   ./models/download-ggml-model.sh base
   cd ..
   ```
   The command above downloads the multilingual `ggml-base.bin` model which
   supports Russian. See the [whisper.cpp models list](https://github.com/ggml-org/whisper.cpp#models)
   for other available models such as `large-v3-turbo`.
4. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   The remaining commands assume the virtual environment is active.
5. Install Python dependencies (run inside the virtual environment):
   ```bash
    pip install -U openai-whisper openai numpy nltk google-generativeai pycountry
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
   You can also skip recording and pass an existing audio file:
   ```bash
   python meeting_notes.py path/to/audio.wav
   ```
3. The script records (or processes the provided file), transcribes the audio with Whisper, and summarizes the transcript using an LLM (OpenAI or Gemini). Each run creates a new folder inside the directory specified by `output_dir`, storing the audio and transcript. When you pass an existing audio file, the script uses its directory for all output files and appends the sanitized filename to the notes and transcript.
   If `use_calendar_title` is enabled on macOS, the current Calendar event title
   is used for the folder name instead of a timestamp.
   Example structure when recording:
   ```
   output/
   output/meeting_20250703_104658/
   output/meeting_20250703_104658/notes_20250703_104658.md
   output/meeting_20250703_104658/recording_20250703_104658.wav
   output/meeting_20250703_104658/transcript_20250703_104658.txt
   ```
   Example structure when providing `team_sync.wav`:
   ```
   path/to/team_sync.wav
   path/to/notes_team_sync.md
  path/to/transcript_team_sync.txt
  ```

The folder name (taken from the Calendar event title or the provided audio file
name) is also passed to the language model to help it understand what the
meeting is about.

## Configuration

`settings.json` options:

- `audio_device` – audio device name for ffmpeg (macOS `avfoundation`). The
  script automatically looks up the corresponding device index.
- `input_source` – name of the macOS input device to select before recording.
- `output_source` – name of the macOS output device to select before recording.
- `duration_seconds` – max recording length in seconds. Set to `0` to record until you press **Ctrl+C**.
- `transcription_model` – Whisper model size (e.g., `tiny`, `base`, `small`).
- `transcription_backend` – `whisper` (Python implementation) or `whispercpp`.
- `whispercpp_binary` – path to the compiled `whisper-cli` executable.
- `whispercpp_model` – path to the Whisper model file in ggml format.
- `language` – ISO language code used by Whisper (e.g., `ru`, `en`). Generated notes use this language. The script converts it to a full language name when prompting the LLM.
- `summary_sentences` – number of sentences to include in the generated notes.
- `output_format` – `text` or `markdown` for notes output.
- `llm_provider` – `openai` or `gemini` for summarization.
- `openai_model` – OpenAI chat model to use (default `gpt-3.5-turbo`).
- `gemini_model` – Gemini model name (default `gemini-pro`).
- `output_dir` – root folder where meeting subfolders are created.
- `keep_audio` – set to `false` to delete the recording after transcription.
- `open_notes` – automatically open the saved notes file with the OS default
  application when the run completes.
`post_save_command` – optional shell command run after saving notes. The
  placeholders `{path}` and `{url}` are replaced with the absolute notes path
  and its URL-encoded form. Two additional placeholders `{relative}` and
  `{relative_url}` provide the path relative to the parent of `output_dir` and
  its encoded form. Example for opening the notes in Obsidian on macOS:
  `open "obsidian://open?vault=obsidian-vault&file={relative_url}"`.
- `use_calendar_title` – if `true` on macOS, the script uses the current
  Calendar event title for the output folder name instead of a timestamp. This
  requires the [`icalBuddy`](https://github.com/ali-rantakari/icalBuddy) utility
  to be installed. The folder name is also provided to the LLM prompt as the
  meeting title for additional context.


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
4. List available devices to find the name of the Aggregate Device:
   ```bash
   ffmpeg -f avfoundation -list_devices true -i ""
   ```
   Use the displayed name for `audio_device` in `settings.json`.
   The script will resolve the index automatically so recording includes audio
   from the browser and your microphone.

## Creating an Apple Shortcut for Quick Recording

You can add a menu bar shortcut to run the script with one click:

1. Open **Shortcuts** and create a new shortcut.
2. Add the **Run Shell Script** action.
3. Enter the command to start the script, for example:
   ```bash
   /path/to/venv/bin/python /path/to/meeting-notes-script/meeting_notes.py
   ```
   Use absolute paths so the shortcut works from any location.
4. Click the shortcut's **i** button and enable **Pin in Menu Bar**.
5. Optionally assign a keyboard shortcut.
6. Click the new icon in the menu bar (top-right panel) whenever you want to start recording.
## Notes

- Capturing system audio from Google Meet may require a virtual audio driver such as BlackHole. Select the device in `settings.json`.
- Whisper models are downloaded automatically when first used; make sure you have sufficient disk space.
- Set `language` to `ru` in `settings.json` to generate transcripts and notes in Russian.
- Set the `OPENAI_API_KEY` environment variable to use OpenAI-based summarization.
- Set the `GOOGLE_API_KEY` environment variable to use Gemini-based summarization.

