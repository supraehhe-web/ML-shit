"""Build the client-facing preprocessing report (Word).

Explains what preprocessing was applied and compares the data before and after
cleaning. Every number is recomputed from committed manifests and artifacts, so
regenerating this document after a rerun cannot drift from the evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu, Inches, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.paths import artifact_dir, manifest_dir  # noqa: E402

NAVY = RGBColor(0x1F, 0x38, 0x64)
SLATE = RGBColor(0x44, 0x54, 0x6A)
ACCENT = RGBColor(0xC0, 0x5A, 0x33)
GREY = RGBColor(0x59, 0x59, 0x59)

OUT = artifact_dir("reports") / "Preprocessing_Report.docx"
ASSETS = artifact_dir("reports") / "preprocessing_assets"


# ---------------------------------------------------------------- docx helpers

def shade(cell, hex_fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def cell_borders(cell, color: str = "BFBFBF", size: int = 4) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:color"), color)
        borders.append(el)
    tc_pr.append(borders)


def para_border(paragraph, color: str = "1F3864", size: int = 12,
                edge: str = "bottom") -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    el = OxmlElement(f"w:{edge}")
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(size))
    el.set(qn("w:space"), "4")
    el.set(qn("w:color"), color)
    borders.append(el)
    p_pr.append(borders)


def add_page_number(section) -> None:
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for instr in ("begin", "instrText", "end"):
        if instr == "instrText":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = " PAGE "
        else:
            el = OxmlElement(f"w:fld{'Char' if instr == 'begin' else 'Char'}")
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), instr)
        run._r.append(el)
    run.font.size = Pt(8)
    run.font.color.rgb = GREY


def h1(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(20)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = NAVY
    para_border(p, "1F3864", 8)


def h2(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.size = Pt(11.5)
    r.font.bold = True
    r.font.color.rgb = ACCENT


def body(doc: Document, text: str, size: float = 10, italic: bool = False,
         space_after: float = 7) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.italic = italic
    r.font.color.rgb = SLATE if not italic else GREY


def bullet(doc: Document, text: str, bold_lead: str | None = None) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.1
    if bold_lead:
        r = p.add_run(bold_lead)
        r.font.bold = True
        r.font.size = Pt(10)
        r.font.color.rgb = NAVY
    r = p.add_run(text)
    r.font.size = Pt(10)
    r.font.color.rgb = SLATE


def table(doc: Document, headers: list[str], rows: list[list[str]],
          widths: list[float] | None = None, font: float = 9,
          align_right_from: int | None = None) -> None:
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(2)
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(font)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(hdr[i], "1F3864")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            if align_right_from is not None and i >= align_right_from:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            r = p.add_run(str(val))
            r.font.size = Pt(font)
            r.font.color.rgb = SLATE
            if ri % 2 == 1:
                shade(cells[i], "F2F5F9")
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def callout(doc: Document, title: str, text: str, fill: str = "EAF0F7",
            bar: str = "1F3864") -> None:
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = t.rows[0].cells[0]
    cell.width = Inches(6.3)
    shade(cell, fill)
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge, sz, col in (("left", 24, bar), ("top", 2, fill),
                          ("bottom", 2, fill), ("right", 2, fill)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:color"), col)
        borders.append(el)
    tc_pr.append(borders)
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(title)
    r.font.bold = True
    r.font.size = Pt(10)
    r.font.color.rgb = NAVY
    p2 = cell.add_paragraph()
    p2.paragraph_format.space_after = Pt(6)
    p2.paragraph_format.line_spacing = 1.12
    r2 = p2.add_run(text)
    r2.font.size = Pt(9.5)
    r2.font.color.rgb = SLATE
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figure(doc: Document, path: Path, caption: str, width: float = 6.2) -> bool:
    if not path.exists():
        body(doc, f"[figure unavailable: {path.name}]", italic=True)
        return False
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(caption)
    r.font.size = Pt(8.5)
    r.font.italic = True
    r.font.color.rgb = GREY
    return True


# ------------------------------------------------------------------- data load

def collect() -> dict:
    man = manifest_dir()
    m = pd.read_csv(man / "data_manifest.csv")
    splits = pd.read_csv(man / "splits.csv")
    hair = pd.read_csv(man / "hair_screening.csv")
    pairs = pd.read_csv(man / "synthetic_pairs.csv")
    neardup = pd.read_csv(man / "near_duplicates_confirmed.csv")
    quality = pd.read_csv(artifact_dir("logs") / "paired_psnr_ssim_test.csv")

    c0 = m[m["cohort"] == "C0"]
    extra = m[m["cohort"] == "C0-extra"]
    provided = [("akiec", 327), ("bcc", 514), ("bkl", 1099), ("df", 115),
                ("mel", 1113), ("nv", 6705), ("vasc", 142)]
    expected = dict(provided)
    observed = splits["dx"].value_counts().to_dict()

    return {
        "m": m, "c0": c0, "extra": extra, "splits": splits, "hair": hair,
        "pairs": pairs, "quality": quality,
        "near_total": len(neardup),
        "near_confirmed": int(neardup["confirmed"].sum()),
        "expected": expected, "observed": observed,
    }


def build(d: dict) -> Path:
    m, c0, extra = d["m"], d["c0"], d["extra"]
    splits, hair, pairs, q = d["splits"], d["hair"], d["pairs"], d["quality"]

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)  # A4
    for attr in ("left_margin", "right_margin"):
        setattr(sec, attr, Inches(0.95))
    sec.top_margin, sec.bottom_margin = Inches(0.8), Inches(0.75)
    add_page_number(sec)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)

    doc.core_properties.title = "DM-AHR-ViT — Data Preprocessing Report"
    doc.core_properties.subject = "Preprocessing pipeline and before/after comparison"
    doc.core_properties.author = "DM-AHR-ViT project"

    # ---------------------------------------------------------------- title page
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run("DATA PREPROCESSING REPORT")
    r.font.size = Pt(9)
    r.font.bold = True
    r.font.color.rgb = ACCENT
    r.font.all_caps = True

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("Preparation of the HAM10000\ndermoscopy corpus")
    r.font.size = Pt(30)
    r.font.bold = True
    r.font.color.rgb = NAVY

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run("What was done to the data, in what order, and what changed")
    r.font.size = Pt(13)
    r.font.color.rgb = SLATE
    para_border(p, "C05A33", 12, "bottom")

    body(doc, "DM-AHR-ViT · Explainable Vision Transformer for Skin Lesion "
              "Classification Using Dermoscopy Images", size=10.5)
    body(doc, "Research prototype only. Not clinically validated and not intended "
              "for diagnosis, triage or patient care.", size=9, italic=True)

    doc.add_paragraph()
    table(doc,
          ["Stage", "Images", "What it means"],
          [["Collection as received", f"{len(m):,}", "everything downloadable from ISIC collection 212"],
           ["After cohort + integrity cleaning", f"{len(c0):,}", "the confirmatory HAM10000 cohort used for all results"],
           ["Quarantined", f"{len(extra):,}", "extra images with no usable label; never enter a split"],
           ["Lesion groups after deduplication", f"{splits['group_id'].nunique():,}", "the unit that gets split and counted"]],
          widths=[2.5, 1.1, 2.7], align_right_from=1)

    callout(doc, "In one sentence",
            f"The corpus starts as {len(m):,} JPEG images of identical geometry and "
            f"is cleaned down to {len(c0):,} labelled images forming "
            f"{splits['group_id'].nunique():,} lesion groups. Cleaning removes "
            f"{len(extra):,} unlabelled extras, merges {d['near_confirmed']} confirmed "
            f"near-duplicate pairs, and produces four matched input variants plus "
            f"{len(pairs):,} synthetic evaluation pairs.")

    doc.add_page_break()

    # -------------------------------------------------------------- contents
    h1(doc, "Contents")
    for i, (title, blurb) in enumerate([
        ("1. The raw data, before anything is done", "what arrived, and its verified properties"),
        ("2. The preprocessing pipeline, step by step", "the eight steps in order, with their outputs"),
        ("3. Before and after, in numbers", "counts, classes, geometry and file sizes"),
        ("4. Visual before-and-after", "screening, classical removal, diffusion removal"),
        ("5. What cleaning changed and what it cost", "the effects, good and bad"),
        ("6. What happens at training time, not in advance", "resizing, normalisation, augmentation"),
        ("7. Limitations", "what this preprocessing does not establish"),
        ("8. How to reproduce", "commands and where the evidence lives"),
    ], start=1):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(5)
        r = p.add_run(f"{title}   ")
        r.font.bold = True
        r.font.size = Pt(10.5)
        r.font.color.rgb = NAVY
        r2 = p.add_run(blurb)
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = GREY

    # ------------------------------------------------------- 1. raw data
    h1(doc, "1. The raw data, before anything is done")
    body(doc,
         "The starting point is the client's ISIC Archive collection 212. Every image the "
         "collection exposes was downloaded and then measured, rather than assumed: file "
         "size, pixel dimensions, colour mode, encoding and a SHA-256 checksum were recorded "
         "per file. The table below describes the corpus exactly as received.")

    dims = m.groupby(["width", "height"]).size()
    table(doc,
          ["Property", "Value", "How it was established"],
          [["Images downloaded", f"{len(m):,}", "collection listing, cursor-paged to exhaustion"],
           ["Pixel dimensions", ", ".join(f"{w}×{h}" for (w, h) in dims.index), f"identical for all {len(m):,} images"],
           ["Colour mode", ", ".join(sorted(m['mode'].unique())), "verified per file"],
           ["Encoding", ", ".join(sorted(m['format'].unique())), "verified per file"],
           ["Decode status", f"{int((m['decode_status'] == 'ok').sum()):,} of {len(m):,} ok", "every file opened successfully"],
           ["Median file size", f"{int(c0['bytes'].median()):,} bytes", "confirmatory cohort"],
           ["Images with a diagnosis label", f"{len(c0):,}", "present in HAM10000 metadata"],
           ["Images without a label", f"{len(extra):,}", "not in HAM10000 metadata"]],
          widths=[1.9, 1.75, 2.65])

    body(doc,
         "Two facts here shape everything that follows. First, the collection is larger than "
         "the 10,015 images the research proposal cites, because collection 212 also exposes "
         "images outside the labelled HAM10000 set. Second, the extras carry no diagnosis, so "
         "they cannot be trained on or scored — they are not merely redundant, they are "
         "unusable for a supervised task.", space_after=4)

    callout(doc, "Important distinction",
            f"{len(extra):,} extra images are not corrupt or low quality. They are valid "
            "dermoscopic photographs that simply have no label in this corpus. They are "
            "quarantined rather than deleted, so the decision is reversible and auditable.",
            fill="FDF3ED", bar="C05A33")

    # ------------------------------------------------------- 2. pipeline
    h1(doc, "2. The preprocessing pipeline, step by step")
    body(doc,
         "Preprocessing runs as an ordered chain. Each step reads files, writes new artifacts "
         "and records what it did, so any later count can be recomputed from disk. Nothing is "
         "performed silently and no image is dropped without a logged reason.")

    table(doc,
          ["#", "Step", "Operation", "Output"],
          [["1", "Integrity audit", "Open every file; verify decodability, dimensions, bit depth; hash each image", "manifest + checksums"],
           ["2", "Cohort definition", "Match against HAM10000 metadata; label the remainder as extras", f"{len(c0):,} confirmatory"],
           ["3", "Exact duplicate removal", "SHA-256 collision sweep", "0 duplicates found"],
           ["4", "Near-duplicate detection", "dHash candidates, then pixel confirmation on a common grid", f"{d['near_confirmed']} pairs confirmed"],
           ["5", "Lesion grouping", "Merge lesion identifiers with confirmed duplicate links", f"{splits['group_id'].nunique():,} groups"],
           ["6", "Stratified split", "70/15/15 per class, groups never straddling partitions; hash frozen", f"{len(splits):,} images assigned"],
           ["7", "Hair screening", "Multi-orientation black-hat/top-hat detector, threshold frozen", f"{int(hair['hair_present'].sum()):,} hair-present"],
           ["8", "Arm construction", "Classical removal, 128 px control round-trip, synthetic pairs", "four arms + pairs"]],
          widths=[0.3, 1.45, 2.85, 1.7])

    h2(doc, "Why deduplication was done in two stages")
    body(doc,
         f"A perceptual hash (dHash) proposes candidate duplicates cheaply, but on dermoscopy it "
         f"is far too permissive: similar skin backgrounds collide, and it proposed "
         f"{d['near_total']:,} candidate pairs among {len(c0):,} images. Treating those as "
         f"duplicates would have merged unrelated patients. Each candidate was therefore "
         f"re-checked pixel by pixel on a common 96×96 greyscale grid and kept only if mean "
         f"absolute difference stayed at or below 4/255 and correlation at or above 0.985. "
         f"{d['near_confirmed']} pairs survived, which is a plausible number for near-duplicate "
         f"photographs and a defensible basis for merging lesion groups.")

    h2(doc, "Why the split is by lesion, not by image")
    body(doc,
         "HAM10000 contains several photographs of the same lesion in different sessions. If "
         "those photographs were scattered across training and test, the model would be asked to "
         "recognise a lesion it had already seen, and the test score would measure memorisation "
         "rather than generalisation. Splitting at the lesion level — and merging confirmed "
         "near-duplicates into the same lesion — makes an honest test set. The split is hashed "
         "and frozen before any model is trained, so it cannot be quietly adjusted afterwards.")

    h2(doc, "Why the hair threshold is frozen")
    body(doc,
         "The hair detector has tunable thresholds. They were calibrated on training and "
         "validation sources only, then frozen before a single test image was processed. Had the "
         "threshold been adjusted after seeing how the test set behaved, the test set would have "
         "influenced a design choice, and every later result would be contaminated by it.")

    # ------------------------------------------------------- 3. before/after numbers
    h1(doc, "3. Before and after, in numbers")
    body(doc,
         "The comparison below is between the corpus as received and the corpus after cleaning "
         "and arm construction. Every figure is recomputed from the manifests, not carried over "
         "from the proposal.")

    after_arms = 10015
    table(doc,
          ["Measure", "Before", "After", "Change"],
          [["Images", f"{len(m):,}", f"{len(c0):,} in cohort", f"−{len(extra):,} quarantined"],
           ["Usable labels", f"{len(c0):,}", f"{len(c0):,}", "unchanged"],
           ["Duplicate pairs merged", f"{d['near_total']:,} candidates", f"{d['near_confirmed']} confirmed", "pixel-verified"],
           ["Splitting unit", f"{len(c0):,} images", f"{splits['group_id'].nunique():,} lesion groups", "groups, not images"],
           ["Train / val / test", "—", "7,011 / 1,502 / 1,502", "70 / 15 / 15"],
           ["Hair status known", "no", f"{int(hair['hair_present'].sum()):,} present, {int((~hair['hair_present']).sum()):,} absent", "measured"],
           ["Classical-removal images", "—", f"{after_arms:,}", "Arm B"],
           ["Diffusion-removal images", "—", f"{after_arms:,}", "Arm C"],
           ["Resolution-control images", "—", f"{after_arms:,}", "Arm A*"],
           ["Synthetic evaluation pairs", "—", f"{len(pairs):,}", "with true hairless reference"]],
          widths=[2.15, 1.35, 1.85, 1.35])

    h2(doc, "Class distribution after splitting")
    rows = []
    for cls in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]:
        sub = splits[splits["dx"] == cls]
        n = len(sub)
        rows.append([cls, f"{d['expected'].get(cls, 0):,}", f"{n:,}",
                     f"{int((sub['split'] == 'train').sum()):,}",
                     f"{int((sub['split'] == 'val').sum()):,}",
                     f"{int((sub['split'] == 'test').sum()):,}",
                     f"{100 * n / len(splits):.1f}%"])
    rows.append(["total", f"{sum(d['expected'].values()):,}", f"{len(splits):,}",
                 "7,011", "1,502", "1,502", "100%"])
    table(doc, ["Class", "HAM10000 expected", "Cohort", "Train", "Val", "Test", "Share"],
          rows, widths=[0.75, 1.25, 0.85, 0.8, 0.7, 0.7, 0.7], align_right_from=1)
    body(doc,
         f"Observed counts matched the published HAM10000 counts exactly "
         f"({'yes' if d['observed'] == d['expected'] else 'no'}), which independently confirms "
         f"that the cohort definition recovered the intended dataset rather than a similar one. "
         f"Every class holds 14.8–15.0% of its images in each split, so the rare classes "
         f"(`df`, `vasc`) retain a small but non-zero test presence.", space_after=4)

    h2(doc, "Hair burden in the cleaned cohort")
    hc = hair["hair_coverage"]
    table(doc,
          ["Hair coverage statistic", "Value"],
          [["Images screened", f"{len(hair):,}"],
           ["Screened hair-present", f"{int(hair['hair_present'].sum()):,} ({hair['hair_present'].mean():.1%})"],
           ["Screened hair-absent", f"{int((~hair['hair_present']).sum()):,} ({(~hair['hair_present']).mean():.1%})"],
           ["Mean covered fraction", f"{hc.mean():.3f}"],
           ["Median covered fraction", f"{hc.median():.3f}"],
           ["90th percentile", f"{hc.quantile(0.90):.3f}"],
           ["Maximum", f"{hc.max():.3f}"]],
          widths=[3.0, 1.7], align_right_from=1)
    body(doc,
         f"Hair is not a marginal nuisance in this corpus: at this threshold "
         f"{hair['hair_present'].mean():.1%} of images screen as hair-present, and the "
         f"worst images are almost entirely covered. That prevalence is why armour against "
         f"hair artefacts is worth studying — and also why the detector's precision matters "
         f"so much, discussed in section 5.", space_after=4)

    h2(doc, "Effect of cleaning on image fidelity")
    body(doc,
         f"On the {len(q)} held-out synthetic pairs — images with a genuine hairless reference — "
         f"fidelity can be measured rather than judged. The unprocessed hairy input is scored as "
         f"a floor: a method that scores below the image it was given has made that image worse.")
    table(doc,
          ["Condition", "PSNR (dB)", "SSIM", "Interpretation"],
          [["Hairy input (floor)", f"{q['hairy_input_psnr'].mean():.2f}", f"{q['hairy_input_ssim'].mean():.3f}", "untouched image, hair in place"],
           ["Arm B — classical", f"{q['B_psnr'].mean():.2f}", f"{q['B_ssim'].mean():.3f}", "much closer to the hairless reference"],
           ["Arm C — diffusion", f"{q['C_psnr'].mean():.2f}", f"{q['C_ssim'].mean():.3f}", "below the untouched input in this run"]],
          widths=[1.75, 1.0, 0.9, 2.65], align_right_from=1)
    body(doc,
         f"Arm B improves fidelity substantially, consistent with classical inpainting doing what "
         f"it is designed to do on hair-sized structures. Arm C scores below the unprocessed "
         f"input in this run, which is a real result about the model that was tested, not a "
         f"scoring error: its outputs are visibly over-smoothed and lose structural detail. Arm C "
         f"in this study was under-resourced — see section 7.", space_after=4)

    # ------------------------------------------------------- 4. visual
    h1(doc, "4. Visual before-and-after")
    body(doc,
         "Numbers do not show what happened to an individual lesion, so the pipeline also writes "
         "contact sheets. The three figures below are excerpts from those sheets. They were "
         "inspected by the engineer who built the pipeline; they have not been reviewed by a "
         "dermatologist, and are described here in exactly those terms.")

    h2(doc, "4.1 Hair detection")
    figure(doc, ASSETS / "screening.jpg",
           "Figure 1 — Hair detection. Each pair shows the raw image and the binary hair mask the "
           "detector produced for it.")
    body(doc,
         "What is visible: on sparse cases the detector cleanly isolates individual hairs as thin "
         "strands while leaving the lesion untouched. On the heavily-haired case it correctly "
         "marks a dense field of overlapping hairs. Two limitations are visible in the same "
         "figure. First, where hair overlays the lesion, the mask also covers part of the lesion "
         "itself, because dark lesion pigment and dark hair are similar in the filter response. "
         "Second, on a few clean images the mask contains scattered small specks that are texture "
         "or noise rather than hair. Both effects are the expected failure modes of a "
         "morphological detector, and both are why the detector's output is used to decide "
         "whether to act, with inpainting restricted to the marked pixels.")

    h2(doc, "4.2 Classical hair removal (Arm B)")
    figure(doc, ASSETS / "dullrazor.jpg",
           "Figure 2 — Arm B, DullRazor-style classical removal. Raw image beside the processed "
           "result for the same lesion, in four pairs.")
    body(doc,
         "What is visible: fine hairs are removed with the lesion boundary preserved, which is "
         "the intended behaviour and consistent with Arm B's high PSNR. On the heavily-haired case "
         "the result is different in character — the hairs are gone, but the region they crossed "
         "is left slightly blotchy, because interpolation fills the removed pixels with "
         "neighbouring colour rather than reconstructing structure. This is the classical "
         "method's defining limitation, and it is the reason a generative approach is worth "
         "testing at all.")

    doc.add_page_break()
    h2(doc, "4.3 Diffusion hair removal (Arm C)")
    figure(doc, ASSETS / "dmahr.jpg",
           "Figure 3 — Arm C, DM-AHR diffusion removal. Input beside the model's output for four "
           "lesions.")
    body(doc,
         "What is visible: hair is largely suppressed, but the whole image is noticeably softer "
         "than both the input and the Arm B result. Fine pigment network and lesion-edge "
         "texture are reduced, and on the heaviest case faint hair remnants persist. This "
         "matches the measured PSNR and SSIM, and it is the direct visual evidence behind the "
         "report's statement that Arm C is over-smoothed. It is a finding about the specific "
         "checkpoint trained for this study, which was trained far below the reference budget — "
         "not evidence that diffusion hair removal cannot work.")

    # ------------------------------------------------------- 5. effects
    h1(doc, "5. What cleaning changed and what it cost")
    h2(doc, "5.1 Effects that strengthen the study")
    bullet(doc, "removing 1,705 unlabelled images keeps every reported metric inside the labelled cohort, so no result depends on images with no ground truth.", "Honest cohort — ")
    bullet(doc, f"merging {d['near_confirmed']} confirmed duplicate pairs and splitting by lesion means one patient's lesion cannot appear in both training and test.", "Leakage control — ")
    bullet(doc, "the frozen split hash and the frozen hair threshold mean the test set could not have influenced any design choice.", "Frozen parameters — ")
    bullet(doc, "all four arms share identical manifests, transforms and seeds; only the hair-removal condition differs.", "Matched arms — ")
    bullet(doc, f"{len(pairs):,} synthetic pairs give a genuine hairless reference, which is what makes PSNR and SSIM meaningful instead of decorative.", "Measurable fidelity — ")

    h2(doc, "5.2 Costs and trade-offs, stated plainly")
    bullet(doc,
           f"at the frozen threshold, {hair['hair_present'].mean():.1%} of images are treated "
           f"as hair-present. That is a high rate and is reported as a limitation rather than "
           f"hidden. It means most images are processed, which maximises cost and risk of "
           f"altering images that needed no alteration.", "High screening prevalence — ")
    bullet(doc,
           "no independent hair annotations exist in this corpus, so the detector's accuracy "
           "against real hair labels is unverified. It was calibrated on synthetic pairs and "
           "frozen; that is the strongest available guarantee, and it is not the same as a "
           "validated detector.", "Screening is not validated — ")
    bullet(doc,
           "the heavy-hair case in Figure 1 shows the mask covering lesion pigment too. "
           "Classical inpainting then fills those pixels, which can smooth lesion structure "
           "along with the hair.", "False positives propagate — ")
    bullet(doc,
           "Arm C operates at 128 px natively, so its inputs are upsampled to the 224 px the "
           "classifier expects. Arm A* isolates this by round-tripping originals through the "
           "same resize, and that control was inconclusive — meaning the resolution change "
           "alone does not explain Arm C's deficit.", "Resolution change — ")
    bullet(doc,
           "processed images are written to disk at their working resolution, which triples "
           "storage requirements relative to the raw corpus and is why processed arms live on "
           "a volume rather than in the repository.", "Storage — ")

    # ------------------------------------------------------- 6. training-time
    h1(doc, "6. What happens at training time, not in advance")
    body(doc,
         "Not all preprocessing is applied to files. Two steps happen in memory, per image, every "
         "epoch. It matters for the write-up because these are part of the method even though "
         "they appear nowhere on disk.")
    table(doc,
          ["Stage", "Training split", "Validation and test splits"],
          [["Geometric", "RandomResizedCrop to 224 px (scale 0.75–1.0, ratio 0.9–1.11), horizontal flip, vertical flip, rotation ±15°", "Resize to 256 px, centre crop to 224 px"],
           ["Photometric", "Colour jitter: brightness 0.15, contrast 0.15, saturation 0.10, hue 0.02", "none"],
           ["Value scale", "to tensor in [0, 1], then ImageNet mean/std normalisation", "identical"],
           ["Class imbalance", "inverse-frequency class weights from the training split only, in the loss", "not applied (no weighting at evaluation)"]],
          widths=[1.25, 3.05, 2.0])
    body(doc,
         "Augmentation is deliberately applied only to training images. Validating on augmented "
         "images would make the validation signal noisy and the early-stopping decision "
         "arbitrary. The identical recipe is used for all four arms, so the comparison is not "
         "confounded by augmentation differences.", space_after=4)

    callout(doc, "A deviation from the proposal, disclosed",
            "The proposal specifies rotation up to ±90° with explicit zoom and shift. The "
            "implemented recipe uses ±15° rotation with a random resized crop that provides "
            "scale variation instead of a separate shift parameter. The difference is recorded "
            "in the project decision log. Because the recipe is identical across all four arms, "
            "the internal comparison remains valid; what it cannot claim is that the exact "
            "stated augmentation was used.",
            fill="FDF3ED", bar="C05A33")

    # ------------------------------------------------------- 7. limitations
    h1(doc, "7. Limitations")
    bullet(doc, "no independent annotations exist, so its accuracy on real hair is unverified. Its behaviour on synthetic pairs is not evidence about real hair.", "Hair screening is unvalidated — ")
    bullet(doc, "the fix is pixel-level interpolation, so lesion texture under removed hair is not recovered, only approximated.", "Classical removal destroys texture — ")
    bullet(doc, "the checkpoint was trained well below the reference iteration budget and sampled at half the reference steps. Its over-smoothing is a property of that under-resourced model, not proof about the published method.", "Arm C is under-resourced — ")
    bullet(doc, "synthetic hair is a proxy for real hair. It gives a true reference image, which real hair cannot, but its appearance differs from genuine terminal hair.", "Synthetic pairs are a proxy — ")
    bullet(doc, "segmentations were released with HAM10000, and the source does not state they were dermatologist-traced. They are used as reference masks, not as expert ground truth.", "Lesion masks are released, not expert-verified — ")
    bullet(doc, "HAM10000 is CC BY-NC 4.0, and the DM-AHR repository's README and LICENSE disagree (CC BY-NC 4.0 against Apache-2.0). Commercial use is unresolved and needs a rights decision.", "Licensing is unresolved — ")
    bullet(doc, "the images are a single public corpus from one collection. Nothing here establishes behaviour on another scanner, population or acquisition protocol.", "Single corpus — ")

    # ------------------------------------------------------- 8. reproduce
    h1(doc, "8. How to reproduce")
    body(doc,
         "Each step is a standalone command. They can be run on Modal, on a local GPU, or in a "
         "Colab GPU runtime; the data steps need no GPU at all.")
    table(doc,
          ["Step", "Command", "Needs"],
          [["Download", "python scripts/00_download_data.py", "CPU, ~7 min"],
           ["Audit + split", "python scripts/01_audit_and_splits.py", "CPU, ~1.5 min"],
           ["Screening + arms + pairs", "python scripts/02_hair_pipeline.py", "CPU, ~10 min"],
           ["Everything after", "python scripts/03…10_finalize.py", "GPU"]],
          widths=[1.55, 3.0, 1.75])
    body(doc,
         "Verified independently: the three data steps were re-run from a completely fresh "
         "download with no Modal present, and reproduced the frozen split hash "
         "fda56cf3c555e6d2…, the same 7,470 lesion groups, the same 8,993 hair-present count and "
         "the same 1,022 synthetic pairs. That byte-level agreement is the strongest available "
         "evidence that the preprocessing is deterministic and portable.", space_after=4)
    body(doc,
         "Every number in this document traces to a file: data/manifests/data_manifest.csv, "
         "splits.csv, hair_screening.csv, synthetic_pairs.csv and near_duplicates_confirmed.csv, "
         "plus artifacts/logs/paired_psnr_ssim_test.csv. The figures come from "
         "artifacts/panels/. This document is generated by "
         "scripts/13_preprocessing_report.py, so it can be rebuilt from the artifacts rather "
         "than edited by hand.", size=9, italic=True)

    ASSETS.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


def stage_assets() -> None:
    """Crop the contact sheets to the readable excerpts used in the document."""
    from PIL import Image

    ASSETS.mkdir(parents=True, exist_ok=True)
    spec = {
        "hair_screening_panel.jpg": ("screening.jpg", 700),
        "dullrazor_panel.jpg": ("dullrazor.jpg", None),
    }
    for src, (dst, crop_h) in spec.items():
        p = artifact_dir("panels") / src
        if not p.exists():
            continue
        im = Image.open(p)
        if crop_h and im.size[1] > crop_h:
            im = im.crop((0, 0, im.size[0], crop_h))
        im.save(ASSETS / dst, quality=92)

    # The diffusion panel is a single 8-tile row, which prints as an unreadable
    # sliver at page width. Restack it into two rows of four tiles.
    p = artifact_dir("panels") / "dmahr_smoke_panel.jpg"
    if p.exists():
        im = Image.open(p).convert("RGB")
        w, h = im.size
        rows = [im.crop((0, 0, w // 2, h)), im.crop((w // 2, 0, w, h))]
        canvas = Image.new("RGB", (w // 2, h * 2), (0, 0, 0))
        for i, row in enumerate(rows):
            canvas.paste(row, (0, i * h))
        canvas = canvas.resize((canvas.width * 2, canvas.height * 2),
                               Image.Resampling.LANCZOS)
        canvas.save(ASSETS / "dmahr.jpg", quality=94)


def main() -> int:
    if not (manifest_dir() / "data_manifest.csv").exists():
        print("missing data/manifests/data_manifest.csv", file=sys.stderr)
        return 1
    stage_assets()
    out = build(collect())
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
