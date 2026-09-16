"""Tests pour format_billing_party_recipient_name (débiteur ≠ contact facturation)."""

from services.documents.invoice_recipient import format_billing_party_recipient_name


def test_organism_only_without_contact():
    assert format_billing_party_recipient_name("Hospice général") == "Hospice général"


def test_organism_plus_attention_contact():
    assert (
        format_billing_party_recipient_name(
            "Hospice général",
            "Amandine HAUSER",
            separator="\n",
        )
        == "Hospice général\nÀ l'att. de Amandine HAUSER"
    )


def test_html_separator_for_template_builder():
    assert (
        format_billing_party_recipient_name(
            "Service des curatelles",
            "Pierre Martin",
            separator="<br/>",
        )
        == "Service des curatelles<br/>À l'att. de Pierre Martin"
    )


def test_contact_same_as_display_name_skips_attn():
    """Évite « Jean Dupont / À l'att. de Jean Dupont » redondant."""
    assert (
        format_billing_party_recipient_name("Jean Dupont", "Jean Dupont")
        == "Jean Dupont"
    )


def test_curatorship_type_does_not_affect_label():
    """Le helper ignore le type : le contact n'est jamais libellé « Curateur »."""
    name = format_billing_party_recipient_name(
        "Service des curatelles",
        "Mme X",
    )
    assert "Curateur" not in name
    assert name == "Service des curatelles\nÀ l'att. de Mme X"
