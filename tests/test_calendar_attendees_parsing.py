import sys
import types
import importlib
from pathlib import Path

# Ensure repo root is on sys.path for direct module import
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Stub heavy imports so meeting_notes can be imported without side effects
sys.modules.setdefault("torch", types.SimpleNamespace(has_mps=False))
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *a, **k: None))
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=lambda *a, **k: None))

meeting_notes = importlib.import_module("meeting_notes")

def test_print_result():
    attendees = meeting_notes.get_calendar_event_attendees()

    print("test")
    print(attendees)

def test_parses_bullet_attendees(monkeypatch):
    # Simulate macOS and presence of icalBuddy
    monkeypatch.setattr(meeting_notes.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(meeting_notes.shutil, "which", lambda name: "/usr/local/bin/icalBuddy")

    sample = (
        "• Standup meeting\n"
        "  location: Zoom\n"
        "  attendees:\n"
        "    • John Doe <john@example.com> (Accepted)\n"
        "    • Jane Roe (Declined) <jane@x.com>\n"
        "  notes: Something else\n"
    )

    def fake_check_output(cmd, timeout):
        assert cmd[-1] == "eventsNow"
        return sample.encode("utf-8")

    monkeypatch.setattr(meeting_notes.subprocess, "check_output", fake_check_output)

    attendees = meeting_notes.get_calendar_event_attendees()
    assert attendees == ["John Doe", "Jane Roe"]


def test_parses_inline_attendees(monkeypatch):
    monkeypatch.setattr(meeting_notes.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(meeting_notes.shutil, "which", lambda name: "/usr/local/bin/icalBuddy")

    sample = (
        "• Event\n"
        "attendees: Alice <a@x>, Bob (Declined); Carol\n"
        "location: Somewhere\n"
    )

    monkeypatch.setattr(
        meeting_notes.subprocess,
        "check_output",
        lambda cmd, timeout: sample.encode("utf-8"),
    )

    attendees = meeting_notes.get_calendar_event_attendees()
    assert attendees == ["Alice", "Bob", "Carol"]


def test_deduplicates_case_insensitively(monkeypatch):
    monkeypatch.setattr(meeting_notes.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(meeting_notes.shutil, "which", lambda name: "/usr/local/bin/icalBuddy")
    sample = (
        "• Event\n"
        "attendees:\n"
        "  - Alice\n"
        "  - alice\n"
        "  - ALICE\n"
    )
    monkeypatch.setattr(
        meeting_notes.subprocess,
        "check_output",
        lambda cmd, timeout: sample.encode("utf-8"),
    )

    attendees = meeting_notes.get_calendar_event_attendees()
    assert attendees == ["Alice"]


def test_non_darwin_returns_empty(monkeypatch):
    monkeypatch.setattr(meeting_notes.platform, "system", lambda: "Linux")
    assert meeting_notes.get_calendar_event_attendees() == []


def test_missing_icalbuddy_returns_empty(monkeypatch):
    monkeypatch.setattr(meeting_notes.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(meeting_notes.shutil, "which", lambda name: None)
    assert meeting_notes.get_calendar_event_attendees() == []

