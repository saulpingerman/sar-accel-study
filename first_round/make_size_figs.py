#!/usr/bin/env python3
"""Panels of the image-size figure of the report, from the maps written by `sizes.py analyze`
(float64 image on a 50 dB scale; one minus the coherence on a log scale from 1e-5 to 1)."""
from PIL import Image

F = 'results/sizes/figs'
for N in (1024, 4096, 8192, 16384):
    Image.open(f'{F}/sz{N}_img_ref.png').convert('L').resize((640, 640), Image.LANCZOS).save(f'report/fig/sz_img_{N}.jpg', quality=88)
    Image.open(f'{F}/sz{N}_loss_tpu-v5e_ffbp_fp32_fast.png').convert('RGB').resize((640, 640), Image.LANCZOS).save(f'report/fig/sz_loss_{N}.jpg', quality=90)
