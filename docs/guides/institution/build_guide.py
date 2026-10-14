"""Génère le guide Word éditable du portail Institution LIRIE.

Les textes restent de vrais paragraphes, les captures de vraies images.
Relancer ce script après remplacement d'un PNG officiel.

Depuis la racine du dépôt, avec l'image API :

    docker run --rm --user root --entrypoint python \\
        -v "<racine du dépôt>:/work" -w /work atmr-api \\
        docs/guides/institution/build_guide.py

Si l'image ne contient pas encore python-docx : ``pip install 'python-docx>=1.1,<2'``.

Les chemins sont relatifs à ce fichier. Aucun chemin machine n'est inscrit
dans le document.
"""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
CAPTURES = ROOT / "captures"
OUTPUT = ROOT / "output" / "Guide_LIRIE_Portail_Institution.docx"

GREEN = RGBColor(0x00, 0x79, 0x6B)
GREEN_HEX = "00796B"
INK = RGBColor(0x1C, 0x28, 0x2E)
INK_HEX = "1C282E"
MUTED = RGBColor(0x5C, 0x6B, 0x73)
SOFT = "F4F7F6"
LINE = "D5E3E0"

CONTENT_MM = 182.0  # A4 210 mm, marges 14 mm

# Largeurs d'affichage. La hauteur suit le ratio du PNG (jamais étiré).
WIDTHS_MM = {
    "01-dashboard.png": 182,
    "02-new-request-patient.png": 168,
    "03-new-request-delivery.png": 112,
    "04-new-request-external.png": 182,
    "05-external-transport-pdf.png": 78,
    "06-external-email-prefilled.png": 110,
    "07-transport-list.png": 140,
    "08-transport-detail.png": 64,
    "09-transport-edit.png": 70,
    "10-patients-list.png": 176,
    "11-notifications.png": 52,
}

REQUIRED = (
    "01-dashboard.png",
    "02-new-request-patient.png",
    "03-new-request-delivery.png",
    "04-new-request-external.png",
    "07-transport-list.png",
    "08-transport-detail.png",
    "10-patients-list.png",
)

OPTIONAL = (
    "05-external-transport-pdf.png",
    "06-external-email-prefilled.png",
    "09-transport-edit.png",
    "11-notifications.png",
)


class GuideBuilder:
    """Assemble le .docx sans aplatir les pages en images."""

    def __init__(self) -> None:
        self.doc = Document()
        self.used: list[str] = []
        self.missing_required: list[str] = []
        self.missing_optional: list[str] = []
        self._num_seq = 40
        self._prepare_document()
        self._prepare_numbering()

    def _prepare_document(self) -> None:
        section = self.doc.sections[0]
        section.page_width = Mm(210)
        section.page_height = Mm(297)
        section.left_margin = Mm(14)
        section.right_margin = Mm(14)
        section.top_margin = Mm(12)
        section.bottom_margin = Mm(13)
        section.header_distance = Mm(4)
        section.footer_distance = Mm(4)
        section.different_first_page_header_footer = False

        sect_pr = section._sectPr
        grid = sect_pr.find(qn("w:docGrid"))
        if grid is not None:
            sect_pr.remove(grid)

        normal = self.doc.styles["Normal"]
        normal.font.name = "Calibri"
        normal.font.size = Pt(11)
        normal.font.color.rgb = INK
        self._set_run_font(normal.element, size_half_points="22", color=INK_HEX)
        pf = normal.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(4)
        pf.line_spacing = 1.05
        pf.widow_control = True

        props = self.doc.core_properties
        props.title = "Guide du portail Institution"
        props.subject = "Créer et suivre une demande de transport"
        props.category = "Guide utilisateur"
        props.language = "fr-CH"
        props.comments = (
            "Guide opérationnel éditable. Textes en paragraphes Word, "
            "captures insérées comme images."
        )

        self._set_update_fields()
        self._add_footer(section.footer)

    def _set_update_fields(self) -> None:
        settings = self.doc.settings.element
        update = OxmlElement("w:updateFields")
        update.set(qn("w:val"), "true")
        settings.append(update)

    def _set_run_font(self, parent, *, size_half_points: str, color: str, bold: bool = False) -> None:
        r_pr = parent.find(qn("w:rPr"))
        if r_pr is None:
            r_pr = parent.get_or_add_rPr() if hasattr(parent, "get_or_add_rPr") else OxmlElement("w:rPr")
            if not hasattr(parent, "get_or_add_rPr"):
                parent.append(r_pr)
        r_fonts = r_pr.find(qn("w:rFonts"))
        if r_fonts is None:
            r_fonts = OxmlElement("w:rFonts")
            r_pr.append(r_fonts)
        for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            r_fonts.set(qn(attr), "Calibri")
        lang = r_pr.find(qn("w:lang"))
        if lang is None:
            lang = OxmlElement("w:lang")
            r_pr.append(lang)
        lang.set(qn("w:val"), "fr-CH")
        if bold:
            b = r_pr.find(qn("w:b"))
            if b is None:
                b = OxmlElement("w:b")
                r_pr.append(b)

    def _prepare_numbering(self) -> None:
        numbering = self.doc.part.numbering_part._element
        numbering.append(self._abstract(100, kind="number"))
        numbering.append(self._abstract(101, kind="bullet"))

    def _abstract(self, abstract_id: int, *, kind: str) -> OxmlElement:
        abstract = OxmlElement("w:abstractNum")
        abstract.set(qn("w:abstractNumId"), str(abstract_id))
        multi = OxmlElement("w:multiLevelType")
        multi.set(qn("w:val"), "hybridMultilevel")
        abstract.append(multi)

        lvl = OxmlElement("w:lvl")
        lvl.set(qn("w:ilvl"), "0")
        start = OxmlElement("w:start")
        start.set(qn("w:val"), "1")
        lvl.append(start)
        fmt = OxmlElement("w:numFmt")
        fmt.set(qn("w:val"), "decimal" if kind == "number" else "bullet")
        lvl.append(fmt)
        text = OxmlElement("w:lvlText")
        text.set(qn("w:val"), "%1." if kind == "number" else "•")
        lvl.append(text)
        jc = OxmlElement("w:lvlJc")
        jc.set(qn("w:val"), "left")
        lvl.append(jc)
        suff = OxmlElement("w:suff")
        suff.set(qn("w:val"), "space")
        lvl.append(suff)

        p_pr = OxmlElement("w:pPr")
        ind = OxmlElement("w:ind")
        ind.set(qn("w:left"), "420")
        ind.set(qn("w:hanging"), "280")
        p_pr.append(ind)
        lvl.append(p_pr)

        r_pr = OxmlElement("w:rPr")
        r_fonts = OxmlElement("w:rFonts")
        r_fonts.set(qn("w:ascii"), "Calibri")
        r_fonts.set(qn("w:hAnsi"), "Calibri")
        r_pr.append(r_fonts)
        color = OxmlElement("w:color")
        color.set(qn("w:val"), GREEN_HEX)
        r_pr.append(color)
        if kind == "number":
            r_pr.append(OxmlElement("w:b"))
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), "22")
        r_pr.append(sz)
        lvl.append(r_pr)
        abstract.append(lvl)
        return abstract

    def _new_num(self, abstract_id: int) -> int:
        self._num_seq += 1
        num_id = self._num_seq
        numbering = self.doc.part.numbering_part._element
        num = OxmlElement("w:num")
        num.set(qn("w:numId"), str(num_id))
        abs_ref = OxmlElement("w:abstractNumId")
        abs_ref.set(qn("w:val"), str(abstract_id))
        num.append(abs_ref)
        # Chaque liste repart à 1, y compris si elles partagent le même modèle.
        override = OxmlElement("w:lvlOverride")
        override.set(qn("w:ilvl"), "0")
        start = OxmlElement("w:startOverride")
        start.set(qn("w:val"), "1")
        override.append(start)
        num.append(override)
        numbering.append(num)
        return num_id

    def _add_footer(self, footer) -> None:
        p = footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        self._run(p, "LIRIE  ·  Guide du portail Institution", size=8, color=MUTED)
        self._run(p, "          ", size=8, color=MUTED)
        self._add_page_field(p)

        p_pr = p._p.get_or_add_pPr()
        tabs = OxmlElement("w:tabs")
        tab = OxmlElement("w:tab")
        tab.set(qn("w:val"), "right")
        tab.set(qn("w:pos"), str(int(CONTENT_MM * 56.6929)))
        tabs.append(tab)
        p_pr.append(tabs)
        # Le numéro de page est poussé à droite par une tabulation.
        # On reconstruit le paragraphe pour placer la tabulation au bon endroit.
        for child in list(p._p):
            if child.tag == qn("w:r"):
                p._p.remove(child)
        self._run(p, "LIRIE  ·  Guide du portail Institution", size=8, color=MUTED)
        tab_run = p.add_run()
        tab_el = OxmlElement("w:tab")
        tab_run._r.append(tab_el)
        self._style_run(tab_run, size=8, color=MUTED)
        self._add_page_field(p)

    def _add_page_field(self, paragraph) -> None:
        run = paragraph.add_run()
        self._style_run(run, size=8, color=MUTED)
        begin = OxmlElement("w:fldChar")
        begin.set(qn("w:fldCharType"), "begin")
        instr = OxmlElement("w:instrText")
        instr.set(qn("xml:space"), "preserve")
        instr.text = " PAGE "
        end = OxmlElement("w:fldChar")
        end.set(qn("w:fldCharType"), "end")
        run._r.append(begin)
        run._r.append(instr)
        run._r.append(end)

    def _style_run(self, run, *, size: float, color: RGBColor, bold: bool = False, italic: bool = False) -> None:
        run.font.name = "Calibri"
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.bold = bold
        run.italic = italic
        r_pr = run._r.get_or_add_rPr()
        r_fonts = r_pr.find(qn("w:rFonts"))
        if r_fonts is None:
            r_fonts = OxmlElement("w:rFonts")
            r_pr.append(r_fonts)
        for attr in ("w:ascii", "w:hAnsi", "w:cs"):
            r_fonts.set(qn(attr), "Calibri")
        lang = r_pr.find(qn("w:lang"))
        if lang is None:
            lang = OxmlElement("w:lang")
            r_pr.append(lang)
        lang.set(qn("w:val"), "fr-CH")

    def _run(self, paragraph, text: str, *, size: float = 11, color: RGBColor = INK,
             bold: bool = False, italic: bool = False):
        run = paragraph.add_run(text)
        self._style_run(run, size=size, color=color, bold=bold, italic=italic)
        return run

    def _paragraph(self, *, before: float = 0, after: float = 3, align=None,
                   keep_next: bool = False, page_break: bool = False,
                   line: float = 1.05):
        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(before)
        pf.space_after = Pt(after)
        pf.line_spacing = line
        pf.widow_control = True
        if keep_next:
            pf.keep_with_next = True
        if page_break:
            pf.page_break_before = True
        if align is not None:
            p.alignment = align
        return p

    def _outline(self, paragraph, level: int) -> None:
        p_pr = paragraph._p.get_or_add_pPr()
        outline = OxmlElement("w:outlineLvl")
        outline.set(qn("w:val"), str(level))
        p_pr.append(outline)

    def title_block(self) -> None:
        brand = self._paragraph(before=0, after=0, keep_next=True, line=1.0)
        self._run(brand, "LIRIE", size=22, color=GREEN, bold=True)
        self._outline(brand, 0)

        guide = self._paragraph(before=0, after=1, keep_next=True, line=1.0)
        self._run(guide, "Guide du portail Institution", size=16, color=INK, bold=True)

        subtitle = self._paragraph(before=0, after=6, keep_next=True, line=1.0)
        self._run(subtitle, "Créer et suivre une demande de transport", size=11, color=MUTED)
        self._bottom_border(subtitle)

    def _bottom_border(self, paragraph) -> None:
        p_pr = paragraph._p.get_or_add_pPr()
        borders = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "12")
        bottom.set(qn("w:space"), "1")
        bottom.set(qn("w:color"), GREEN_HEX)
        borders.append(bottom)
        p_pr.append(borders)

    def heading(self, text: str, *, level: int = 0, page_break: bool = False) -> None:
        size = 15 if level == 0 else 12
        p = self._paragraph(before=2 if level else 0, after=3, keep_next=True,
                            page_break=page_break, line=1.0)
        self._run(p, text, size=size, color=GREEN, bold=True)
        self._outline(p, level)

    def body(self, text: str, *, after: float = 3, italic: bool = False, keep_next: bool = False) -> None:
        p = self._paragraph(after=after, keep_next=keep_next)
        self._run(p, text, size=11, italic=italic, color=MUTED if italic else INK)

    def steps(self, items: list[str]) -> None:
        num_id = self._new_num(100)
        for index, item in enumerate(items):
            p = self._paragraph(before=0, after=1, line=1.02, keep_next=index < len(items) - 1)
            self._apply_num(p, num_id)
            self._run(p, item, size=11)

    def bullets(self, items: list[str]) -> None:
        num_id = self._new_num(101)
        for item in items:
            p = self._paragraph(before=0, after=0, line=1.02)
            self._apply_num(p, num_id)
            self._run(p, item, size=11)

    def _apply_num(self, paragraph, num_id: int) -> None:
        p_pr = paragraph._p.get_or_add_pPr()
        num_pr = OxmlElement("w:numPr")
        ilvl = OxmlElement("w:ilvl")
        ilvl.set(qn("w:val"), "0")
        nid = OxmlElement("w:numId")
        nid.set(qn("w:val"), str(num_id))
        num_pr.append(ilvl)
        num_pr.append(nid)
        p_pr.append(num_pr)

    def figure(self, filename: str, caption: str, *, required: bool = True) -> None:
        self._figure_into(None, filename, caption, required=required)

    def _figure_into(self, cell, filename: str, caption: str, *, required: bool) -> None:
        path = CAPTURES / filename
        if not path.is_file():
            bucket = self.missing_required if required else self.missing_optional
            bucket.append(filename)
            print(f"{'ERREUR' if required else 'INFO'} capture absente : {filename}", file=sys.stderr)
            return
        width = WIDTHS_MM[filename]
        if cell is None:
            p = self._paragraph(before=1, after=0, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.0)
            cap_parent = None
        else:
            cell.text = ""
            p = cell.paragraphs[0]
            self._tighten(p, after=0)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap_parent = cell
        run = p.add_run()
        shape = run.add_picture(str(path), width=Mm(width))
        doc_pr = shape._inline.docPr
        doc_pr.set("descr", caption)
        doc_pr.set("title", caption)
        self.used.append(filename)
        if cap_parent is None:
            cap = self._paragraph(before=0, after=2, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.0)
        else:
            cap = cap_parent.add_paragraph()
            self._tighten(cap, after=0)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self._run(cap, caption, size=8.5, color=MUTED, italic=True)

    def _separate_from_previous_table(self) -> None:
        """Empêche Word de fusionner deux tableaux consécutifs (bordure continue)."""
        body = self.doc.element.body
        previous = None
        for child in body:
            if child.tag == qn("w:sectPr"):
                continue
            previous = child
        if previous is None or previous.tag != qn("w:tbl"):
            return
        p = self._paragraph(before=0, after=0, line=1.0)
        p_pr = p._p.get_or_add_pPr()
        spacing = p_pr.find(qn("w:spacing"))
        if spacing is None:
            spacing = OxmlElement("w:spacing")
            p_pr.append(spacing)
        spacing.set(qn("w:before"), "0")
        spacing.set(qn("w:after"), "60")
        spacing.set(qn("w:line"), "20")
        spacing.set(qn("w:lineRule"), "exact")

    def callout(self, title: str, lines: list[str]) -> None:
        self._separate_from_previous_table()
        table = self.doc.add_table(rows=1, cols=1)
        self._style_table(table, [CONTENT_MM], fill=SOFT, accent=True)
        cell = table.cell(0, 0)
        cell.text = ""
        first = cell.paragraphs[0]
        self._tighten(first, after=1)
        self._run(first, title, size=9, color=GREEN, bold=True)
        for line in lines:
            p = cell.add_paragraph()
            self._tighten(p, after=0)
            self._run(p, line, size=10.5)

    def checklist(self, title: str, items: list[str]) -> None:
        self._separate_from_previous_table()
        table = self.doc.add_table(rows=1, cols=1)
        self._style_table(table, [CONTENT_MM], fill=SOFT, accent=True)
        cell = table.cell(0, 0)
        cell.text = ""
        title_p = cell.paragraphs[0]
        self._tighten(title_p, after=2)
        self._run(title_p, title, size=9, color=GREEN, bold=True)
        for item in items:
            p = cell.add_paragraph()
            self._tighten(p, after=0)
            mark = p.add_run("☐  ")
            self._style_run(mark, size=11, color=GREEN)
            mark.font.name = "Segoe UI Symbol"
            r_fonts = mark._r.get_or_add_rPr().find(qn("w:rFonts"))
            if r_fonts is not None:
                for attr in ("w:ascii", "w:hAnsi", "w:cs"):
                    r_fonts.set(qn(attr), "Segoe UI Symbol")
            self._run(p, item, size=10.5)

    def _tighten(self, paragraph, *, after: float) -> None:
        pf = paragraph.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(after)
        pf.line_spacing = 1.0

    def _style_table(self, table, widths_mm: list[float], *, fill: str, accent: bool) -> None:
        table.autofit = False
        table.allow_autofit = False
        tbl = table._tbl
        tbl_pr = tbl.tblPr
        tbl_style = tbl_pr.find(qn("w:tblStyle"))
        if tbl_style is not None:
            tbl_pr.remove(tbl_style)
        total = sum(widths_mm)
        tbl_w = tbl_pr.find(qn("w:tblW"))
        if tbl_w is None:
            tbl_w = OxmlElement("w:tblW")
            tbl_pr.append(tbl_w)
        tbl_w.set(qn("w:w"), str(self._twips(total)))
        tbl_w.set(qn("w:type"), "dxa")

        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            el = OxmlElement(f"w:{edge}")
            if accent and edge == "left":
                el.set(qn("w:val"), "single")
                el.set(qn("w:sz"), "16")
                el.set(qn("w:space"), "0")
                el.set(qn("w:color"), GREEN_HEX)
            else:
                el.set(qn("w:val"), "nil")
            borders.append(el)
        existing = tbl_pr.find(qn("w:tblBorders"))
        if existing is not None:
            tbl_pr.remove(existing)
        tbl_pr.append(borders)

        grid = tbl.find(qn("w:tblGrid"))
        if grid is None:
            grid = OxmlElement("w:tblGrid")
            tbl_pr.addnext(grid)
        for child in list(grid):
            grid.remove(child)
        for width in widths_mm:
            col = OxmlElement("w:gridCol")
            col.set(qn("w:w"), str(self._twips(width)))
            grid.append(col)

        for row in table.rows:
            tr = row._tr
            tr_pr = tr.get_or_add_trPr()
            if tr_pr.find(qn("w:cantSplit")) is None:
                tr_pr.append(OxmlElement("w:cantSplit"))
            for index, cell in enumerate(row.cells):
                tc = cell._tc
                tc_pr = tc.get_or_add_tcPr()
                tc_w = tc_pr.find(qn("w:tcW"))
                if tc_w is None:
                    tc_w = OxmlElement("w:tcW")
                    tc_pr.append(tc_w)
                width = widths_mm[min(index, len(widths_mm) - 1)]
                tc_w.set(qn("w:w"), str(self._twips(width)))
                tc_w.set(qn("w:type"), "dxa")
                shd = tc_pr.find(qn("w:shd"))
                if shd is None:
                    shd = OxmlElement("w:shd")
                    tc_pr.append(shd)
                shd.set(qn("w:val"), "clear")
                shd.set(qn("w:color"), "auto")
                shd.set(qn("w:fill"), fill)
                mar = tc_pr.find(qn("w:tcMar"))
                if mar is None:
                    mar = OxmlElement("w:tcMar")
                    tc_pr.append(mar)
                for edge, value in (("top", 50), ("left", 90), ("bottom", 50), ("right", 80)):
                    node = mar.find(qn(f"w:{edge}"))
                    if node is None:
                        node = OxmlElement(f"w:{edge}")
                        mar.append(node)
                    node.set(qn("w:w"), str(value))
                    node.set(qn("w:type"), "dxa")

    @staticmethod
    def _twips(mm: float) -> int:
        return int(round(mm * 56.6929133858))

    def help_line(self) -> None:
        p = self._paragraph(before=1, after=0, line=1.0)
        self._run(p, "LIRIE", size=11, bold=True, color=GREEN)
        self._run(p, "   ·   ", size=11, color=MUTED)
        self._hyperlink(p, "info@lirie.ch", "mailto:info@lirie.ch")
        self._run(p, "   ·   ", size=11, color=MUTED)
        self._hyperlink(p, "www.lirie.ch", "https://www.lirie.ch")

    def _hyperlink(self, paragraph, text: str, url: str) -> None:
        rel_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
        hyperlink = OxmlElement("w:hyperlink")
        hyperlink.set(qn("r:id"), rel_id)
        run = OxmlElement("w:r")
        r_pr = OxmlElement("w:rPr")
        r_fonts = OxmlElement("w:rFonts")
        r_fonts.set(qn("w:ascii"), "Calibri")
        r_fonts.set(qn("w:hAnsi"), "Calibri")
        r_pr.append(r_fonts)
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), "22")
        r_pr.append(sz)
        color = OxmlElement("w:color")
        color.set(qn("w:val"), GREEN_HEX)
        r_pr.append(color)
        underline = OxmlElement("w:u")
        underline.set(qn("w:val"), "single")
        r_pr.append(underline)
        run.append(r_pr)
        text_el = OxmlElement("w:t")
        text_el.text = text
        run.append(text_el)
        hyperlink.append(run)
        paragraph._p.append(hyperlink)

    def build(self) -> None:
        self._page_accueil()
        self._page_patient()
        self._page_livraison_externe()
        self._page_suivi()
        self._page_patients()

    def _page_accueil(self) -> None:
        self.title_block()
        self.body("Le portail Institution permet de :", after=1, keep_next=True)
        self.bullets([
            "créer une demande",
            "retrouver un patient",
            "suivre les transports",
            "modifier une demande",
            "communiquer avec le transporteur",
        ])
        self.figure("01-dashboard.png", "Tableau de bord")
        self.body("Zones utiles :", after=1, keep_next=True)
        self.steps([
            "Nouvelle demande — ouvre le formulaire.",
            "Indicateurs du jour — demandes, en cours, en attente, terminés, annulés.",
            "Transports du jour — départs prévus aujourd’hui.",
            "Demandes récentes — derniers envois et brouillons.",
            "Notifications — cloche et badge des événements non lus.",
        ])

    def _page_patient(self) -> None:
        self.heading("1. Créer une demande de transport", page_break=True)
        self.steps([
            "Cliquez sur « Nouvelle demande ».",
            "Vérifiez que « Transport de personne » est sélectionné.",
            "Recherchez le patient existant.",
            "Indiquez la date de mission et l’heure de départ.",
            "Vérifiez le lieu de départ.",
            "Indiquez la destination.",
            "Pour une destination médicale, renseignez le service ou le médecin.",
            "Indiquez les besoins utiles au transport.",
            "Vérifiez le contact de la course.",
            "Laissez le mode « LIRIE », puis cliquez sur « Envoyer aux transporteurs LIRIE ».",
        ])
        self.body(
            "L’heure à côté de la destination est l’heure du rendez-vous. "
            "Elle peut différer de l’heure de départ.",
            italic=True,
            after=2,
        )
        self.figure("02-new-request-patient.png", "Nouvelle demande — transport de personne")
        self.callout(
            "AVANT D’ENVOYER",
            ["Vérifiez : Patient  ·  Date  ·  Heure  ·  Départ  ·  Destination  ·  Besoins"],
        )

    def _page_livraison_externe(self) -> None:
        self.heading("2. Créer une livraison", page_break=True)
        self.body(
            "Une livraison concerne du matériel ou des documents, "
            "et non le transport d’une personne.",
            after=2,
        )
        self.steps([
            "Ouvrez « Nouvelle demande ».",
            "Sélectionnez « Livraison ».",
            "Indiquez ce qui doit être transporté dans « Description de la livraison ».",
            "Complétez la date de mission et l’heure de départ.",
            "Renseignez le départ et la destination.",
            "Choisissez « LIRIE » ou « Externe », puis envoyez ou enregistrez.",
        ])
        self.figure("03-new-request-delivery.png", "Nouvelle demande — livraison")

        self.heading("3. Utiliser un transporteur externe", level=0)
        self.body(
            "Le transporteur externe ne reçoit pas d’offre LIRIE. "
            "La demande est enregistrée, puis vous lui transmettez le bon.",
            after=2,
        )
        self.steps([
            "Préparez la demande normalement.",
            "Dans le mode d’envoi, sélectionnez « Externe ».",
            "Indiquez le nom du transporteur.",
            "Ajoutez son téléphone et son e-mail.",
            "Cliquez sur « Enregistrer ».",
        ])
        self.figure(
            "04-new-request-external.png",
            "Mode Externe — nom, téléphone et e-mail du transporteur",
        )
        self.body("Après l’enregistrement, si un e-mail est indiqué :", after=1, keep_next=True)
        self.steps([
            "le bon de transport PDF est téléchargé ;",
            "la messagerie s’ouvre ;",
            "le destinataire, l’objet et le message sont préremplis ;",
            "vérifiez le message, puis joignez le bon PDF téléchargé avant l’envoi. "
            "Le fichier n’est pas joint automatiquement.",
        ])
        self.figure("05-external-transport-pdf.png", "Bon de transport PDF", required=False)
        self.figure("06-external-email-prefilled.png", "Message prérempli", required=False)

    def _page_suivi(self) -> None:
        self.heading("4. Suivre une demande", page_break=True)
        self.steps([
            "Ouvrez « Transports ».",
            "Recherchez le patient ou la demande.",
            "Utilisez les filtres si nécessaire.",
            "Vérifiez l’heure, le trajet et le statut.",
            "Cliquez sur la demande pour ouvrir son détail.",
        ])
        self.figure("07-transport-list.png", "Liste des transports")
        self.heading("Comprendre le détail", level=1)
        self._detail_with_capture()
        self.callout(
            "MODIFIER",
            [
                "Si une heure, une destination ou une information change, "
                "modifiez la demande existante lorsque l’action « Modifier » est disponible.",
                "Évitez de recréer une deuxième demande pour le même transport.",
            ],
        )
        self.figure("09-transport-edit.png", "Modification d’une demande", required=False)

    def _detail_with_capture(self) -> None:
        """Texte à gauche, panneau détail plus étroit à droite — une seule rangée."""
        table = self.doc.add_table(rows=1, cols=2)
        self._style_table(table, [108, 74], fill="FFFFFF", accent=False)
        notes = table.cell(0, 0)
        notes.text = ""
        items = [
            "Statut",
            "Horaire : départ et rendez-vous",
            "Patient",
            "Trajet",
            "Type",
            "Besoins",
            "Actions disponibles",
        ]
        num_id = self._new_num(101)
        first = notes.paragraphs[0]
        self._tighten(first, after=1)
        self._apply_num(first, num_id)
        self._run(first, items[0], size=11)
        for item in items[1:]:
            p = notes.add_paragraph()
            self._tighten(p, after=1)
            self._apply_num(p, num_id)
            self._run(p, item, size=11)
        self._figure_into(
            table.cell(0, 1),
            "08-transport-detail.png",
            "Détail d’une demande",
            required=True,
        )

    def _page_patients(self) -> None:
        self.heading("5. Retrouver un patient", page_break=True)
        self.steps([
            "Ouvrez « Patients ».",
            "Recherchez le patient par son nom.",
            "Vérifiez qu’il existe déjà avant de créer une nouvelle fiche.",
            "Utilisez ensuite cette fiche lors des prochaines demandes.",
        ])
        self.body("Évitez de créer plusieurs fiches pour la même personne.", after=2)
        self.figure("10-patients-list.png", "Liste des patients")

        self.heading("Notifications", level=1)
        self.figure("11-notifications.png", "Notifications", required=False)
        self.body("La cloche en haut de l’écran signale les nouveaux événements.", after=2)
        self.body("Les notifications permettent notamment de retrouver rapidement :", after=1, keep_next=True)
        self.bullets([
            "une demande envoyée",
            "un transport confirmé",
            "un nouveau message",
            "un changement important concernant une demande",
        ])
        self.body("Le badge rouge indique les notifications non lues.", after=2)
        self.body(
            "Le statut affiché dans le détail de la demande reste la référence pour suivre le transport.",
            after=4,
        )
        self.checklist(
            "AVANT D’ENVOYER UNE DEMANDE",
            [
                "Bon patient",
                "Bonne date",
                "Bonne heure",
                "Bon départ",
                "Bonne destination",
                "Service / médecin si nécessaire",
                "Besoins utiles indiqués",
                "Bon mode d’envoi : LIRIE ou Externe",
            ],
        )
        self.heading("Besoin d’aide ?", level=1)
        self.help_line()

    def save(self) -> Path:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(OUTPUT)
        return OUTPUT


def main() -> int:
    builder = GuideBuilder()
    builder.build()
    path = builder.save()
    print(f"FICHIER {path}")
    print("CAPTURES UTILISÉES")
    for name in builder.used:
        print(f"  {name}")
    print("CAPTURES MANQUANTES")
    missing = builder.missing_required + builder.missing_optional
    if not missing:
        print("  (aucune)")
    for name in missing:
        flag = "requise" if name in builder.missing_required else "optionnelle"
        print(f"  {name} ({flag})")
    return 1 if builder.missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
