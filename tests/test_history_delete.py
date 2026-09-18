"""Taking a take back out: the line, the recording, and nothing else.

The console lists what you have dictated and, until it could delete a row,
the only way to remove one sentence from ~/.local/state was to edit a JSONL
file by hand -- and the WAV of you saying it stayed either way. A history
you cannot delete from is a worse place to keep a transcript of everything
than one you can.

The identity tests are the point of the others: a take is addressed by its
id, so what these assert is that the id the console is given is the id the
daemon deletes by, including for a line written before ids were recorded.
"""
from __future__ import annotations

import json
import stat

from omavoi import config, history, paths


def _make(home, takes):
    """Write these takes, each with a recording beside it, without recording."""
    hist = history.History(config.load())
    paths.private_dir(hist.audio_dir)
    for entry in takes:
        if entry.get("wav") is True:
            wav = hist.audio_dir / f"{entry['id']}.wav"
            wav.write_bytes(b"RIFF....WAVE")
            entry["wav"] = str(wav)
        hist.record(dict(entry))
    return hist


def _texts(hist):
    return [e.get("text") for e in hist.iter_entries()]


def test_removing_a_take_takes_its_recording_with_it(home):
    hist = _make(home, [{"id": "a1", "ts": 1.0, "text": "first", "wav": True},
                        {"id": "b2", "ts": 2.0, "text": "second", "wav": True}])
    wav = hist.audio_dir / "a1.wav"
    assert wav.exists()

    report = hist.remove(["a1"])

    assert report == {"removed": 1, "audio": 1, "missing": []}
    assert _texts(hist) == ["second"]
    assert not wav.exists()
    assert (hist.audio_dir / "b2.wav").exists(), "the other take's audio went too"


def test_an_id_that_is_not_there_deletes_nothing_and_says_so(home):
    hist = _make(home, [{"id": "a1", "ts": 1.0, "text": "first"}])

    report = hist.remove(["nope"])

    assert report == {"removed": 0, "audio": 0, "missing": ["nope"]}
    assert _texts(hist) == ["first"]


def test_several_at_once_report_what_was_and_was_not_found(home):
    hist = _make(home, [{"id": "a1", "ts": 1.0, "text": "first"},
                        {"id": "b2", "ts": 2.0, "text": "second"},
                        {"id": "c3", "ts": 3.0, "text": "third"}])

    report = hist.remove(["a1", "c3", "gone"])

    assert report["removed"] == 2
    assert report["missing"] == ["gone"]
    assert _texts(hist) == ["second"]


def test_a_take_written_before_ids_existed_is_still_addressable(home):
    """`record` has always written an id, but this file is never migrated."""
    path = paths.history_file()
    paths.private_dir(path.parent)
    with paths.open_private(path, "w") as fh:
        fh.write(json.dumps({"ts": 1.5, "text": "from before"}) + "\n")
    hist = history.History(config.load())

    listed = list(hist.iter_entries())
    assert listed[0]["id"] == f"{1500:x}"

    assert hist.remove([listed[0]["id"]])["removed"] == 1
    assert _texts(hist) == []


def test_a_line_that_is_not_json_is_kept_rather_than_guessed_at(home):
    path = paths.history_file()
    paths.private_dir(path.parent)
    with paths.open_private(path, "w") as fh:
        fh.write("half a line, written as the disk filled\n")
        fh.write(json.dumps({"id": "a1", "ts": 1.0, "text": "first"}) + "\n")
    hist = history.History(config.load())

    assert hist.remove(["a1"])["removed"] == 1
    assert path.read_text(encoding="utf-8").splitlines() == [
        "half a line, written as the disk filled"]


def test_clear_empties_the_file_and_the_recordings_directory(home):
    hist = _make(home, [{"id": "a1", "ts": 1.0, "text": "first", "wav": True},
                        {"id": "b2", "ts": 2.0, "text": "second", "wav": True}])
    orphan = hist.audio_dir / "c3.wav"
    orphan.write_bytes(b"RIFF....WAVE")

    report = hist.clear()

    assert report["removed"] == 2
    # The orphan too: its entry was trimmed away long ago and nothing names it.
    assert report["audio"] == 3
    assert list(hist.audio_dir.glob("*.wav")) == []
    assert _texts(hist) == []
    assert paths.history_file().read_text(encoding="utf-8") == ""


def test_clear_with_no_history_at_all_is_not_an_error(home):
    hist = history.History(config.load())
    assert hist.clear() == {"removed": 0, "audio": 0}
    assert not paths.history_file().exists(), "an empty history was created to empty it"


def test_a_take_recorded_after_a_delete_still_lands(home):
    hist = _make(home, [{"id": "a1", "ts": 1.0, "text": "first"}])
    hist.remove(["a1"])

    hist.record({"id": "b2", "ts": 2.0, "text": "second"})

    assert _texts(hist) == ["second"]


def test_the_rewrite_leaves_the_history_private(home):
    hist = _make(home, [{"id": "a1", "ts": 1.0, "text": "first"},
                        {"id": "b2", "ts": 2.0, "text": "second"}])

    hist.remove(["a1"])

    path = paths.history_file()
    assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0, "a delete widened the history"
    lock = paths.history_lock()
    assert stat.S_IMODE(lock.stat().st_mode) & 0o077 == 0, "the lock file is world-readable"
