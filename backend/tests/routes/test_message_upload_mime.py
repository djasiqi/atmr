"""Validation MIME upload chat (vocaux mobiles)."""

from __future__ import annotations

from types import SimpleNamespace

from routes.messages import _resolve_upload_mime, _validate_file_upload


def test_resolve_mime_infers_m4a_when_content_type_empty():
    file = SimpleNamespace(content_type="")
    assert _resolve_upload_mime(file, "voice-123.m4a") == "audio/mp4"


def test_resolve_mime_infers_when_octet_stream():
    file = SimpleNamespace(content_type="application/octet-stream")
    assert _resolve_upload_mime(file, "clip.mp3") == "audio/mpeg"


def test_validate_upload_accepts_m4a_with_empty_mime(monkeypatch):
    monkeypatch.setattr("routes.messages.scan_bytes", lambda _b: (True, None))
    file = SimpleNamespace(content_type="")
    err, code = _validate_file_upload(file, "voice.m4a", b"\x00\x01\x02\x03")
    assert err is None
    assert code == 0
