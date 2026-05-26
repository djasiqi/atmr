from pathlib import Path


def test_pdf_module_does_not_import_invalid_reportlab_pt() -> None:
    """Empêche la réintroduction de `from reportlab.lib.units import pt`."""
    pdf_path = Path(__file__).resolve().parents[2] / "services" / "documents" / "pdf.py"
    content = pdf_path.read_text(encoding="utf-8")
    assert "from reportlab.lib.units import pt" not in content
