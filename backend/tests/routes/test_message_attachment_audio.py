"""PJ message : audio_url servi via endpoint authentifié (SEC-06)."""

from __future__ import annotations


def test_message_attachment_serves_audio_url(
    client, app, db, sample_user, sample_company, auth_headers, tmp_path
):
    """GET /messages/<id>/attachment renvoie le fichier audio privé."""
    uploads = tmp_path / "uploads"
    chat = uploads / "chat"
    chat.mkdir(parents=True)
    audio_path = chat / "voice_test.m4a"
    audio_path.write_bytes(b"fake-m4a-bytes")

    app.config["UPLOADS_DIR"] = str(uploads)
    app.config["UPLOAD_FOLDER"] = str(uploads)

    from models import Message
    from models.enums import SenderRole

    message = Message(
        sender_id=int(sample_user.id),
        company_id=int(sample_company.id),
        sender_role=SenderRole.COMPANY,
        content="Message vocal",
        audio_url="/uploads/chat/voice_test.m4a",
        message_type="audio",
    )
    db.session.add(message)
    db.session.commit()

    public = client.get("/uploads/chat/voice_test.m4a")
    assert public.status_code == 404

    resp = client.get(
        f"/api/v1/messages/{message.id}/attachment",
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert resp.data == b"fake-m4a-bytes"


def test_message_attachment_rejects_other_company(
    client, app, db, sample_user, sample_company, auth_headers, tmp_path, company
):
    """Un user d'une autre entreprise ne peut pas lire la PJ."""
    uploads = tmp_path / "uploads"
    chat = uploads / "chat"
    chat.mkdir(parents=True)
    (chat / "voice_other.m4a").write_bytes(b"secret")

    app.config["UPLOADS_DIR"] = str(uploads)
    app.config["UPLOAD_FOLDER"] = str(uploads)

    from models import Message
    from models.enums import SenderRole

    other_company_id = int(company.id)
    assert other_company_id != int(sample_company.id)

    message = Message(
        sender_id=int(sample_user.id),
        company_id=other_company_id,
        sender_role=SenderRole.COMPANY,
        content="Message vocal",
        audio_url="/uploads/chat/voice_other.m4a",
        message_type="audio",
    )
    db.session.add(message)
    db.session.commit()

    resp = client.get(
        f"/api/v1/messages/{message.id}/attachment",
        headers=auth_headers,
    )
    assert resp.status_code == 403
