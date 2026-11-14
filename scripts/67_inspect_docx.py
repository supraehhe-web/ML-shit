"""Print the paragraph styles, tables and images of a .docx (used to model the paper draft)."""
import collections
import sys

import docx

d = docx.Document(sys.argv[1])
print("sections", len(d.sections), "tables", len(d.tables), "inline images", len(d.inline_shapes))
print(collections.Counter(p.style.name for p in d.paragraphs).most_common(15))
for i, p in enumerate(d.paragraphs):
    if p.style.name != "Normal" and len(p.text) > 0 and i < 400:
        if p.style.name.lower().startswith(("heading", "mdpi_1", "mdpi_title", "title")):
            print(i, p.style.name, "|", p.text[:80])
for ti, t in enumerate(d.tables):
    print("table", ti, len(t.rows), "x", len(t.columns), "|", t.rows[0].cells[0].text[:50], "|", t.rows[0].cells[1].text[:40] if len(t.columns) > 1 else "")
