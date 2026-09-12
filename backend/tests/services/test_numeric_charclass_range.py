"""Contrat de la classe numérique ``[0-9.+eE-]`` (plus de plage ``+``–``e``)."""

from __future__ import annotations

import re

from services.documents.pdf import _inset_swiss_qrbill_svg_text

_NUMERIC = r"[0-9.+eE-]+"
_METRIC_LINE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})?\s+[0-9.+eE-]+$")


def _is_numeric_token(value: str) -> bool:
    return re.fullmatch(_NUMERIC, value) is not None


class TestNumericCharclassContract:
    def test_formats_legitimes(self):
        for value in (
            "0",
            "12",
            "12.5",
            "-12.5",
            "+12.5",
            "1e3",
            "1E3",
            "1.2e-3",
            "1.2E+3",
        ):
            assert _is_numeric_token(value), value
            float(value)

    def test_caracteres_accidentels_de_l_ancienne_plage_rejetes(self):
        for value in ("12:0", "12@3", "12A3", "12/3", "12;1", "12<1"):
            assert not _is_numeric_token(value), value


class TestQrbillSvgNumericAttrs:
    def test_coordonnees_scientifiques_reste_parsables(self):
        svg = (
            '<svg viewBox="0 0 210 105">'
            '<text x="10">a</text>'
            '<text x="12.5">b</text>'
            '<text x="-2.5">c</text>'
            '<text x="+3.5">d</text>'
            '<text x="1e2">e</text>'
            '<text x="1.2E+1">f</text>'
            '<text x="1.2e-1">g</text>'
            "</svg>"
        )
        out = _inset_swiss_qrbill_svg_text(svg)
        assert 'viewBox="0 0 210 105"' in out
        assert "<text" in out

    def test_attribut_non_numerique_ignore_sans_exception(self):
        svg = '<svg viewBox="0 0 210 105"><text x="12:00">a</text></svg>'
        assert _inset_swiss_qrbill_svg_text(svg) == svg


class TestPrometheusMetricLine:
    def test_valeur_prometheus_acceptee(self):
        assert _METRIC_LINE.match("dispatch_runs_total 0")
        assert _METRIC_LINE.match('osrm_cache_hits_total{cache="l1"} 1.2e-3')

    def test_valeur_hors_contrat_rejetee(self):
        assert _METRIC_LINE.match("dispatch_runs_total 12:00") is None
        assert _METRIC_LINE.match("dispatch_runs_total 12A3") is None
