#!/usr/bin/env python3
"""Render PDF pages to PNG for a visual check:  python report/render.py <pdf> <outdir> [dpi]"""
import sys
import fitz
doc = fitz.open(sys.argv[1])
dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 70
for i, page in enumerate(doc):
    page.get_pixmap(dpi=dpi).save(f'{sys.argv[2]}/pg-{i + 1:02d}.png')
print(len(doc), 'pages')
