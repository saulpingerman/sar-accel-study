"""Collect the other-modes records (results/oss/modes/<tag>/*.json) into one table: per collection and implementation,
time per full image (measured, or estimated from the regions by pixel count) and the error against the float64 reference
at the implementation's own pixels with its own aperture (range over the three regions).
    python3 modes_table.py results/oss/modes results/oss/modes_table.json"""
import sys, os, json, glob

root, out = sys.argv[1:3]
rows = []
for d in sorted(glob.glob(f'{root}/*/')):
    tag = os.path.basename(d.rstrip('/'))
    for f in sorted(glob.glob(f'{d}*.json')):
        r = json.load(open(f)); name = os.path.basename(f)[:-5]
        if 'regions' not in r or not isinstance(r['regions'], dict):
            continue
        errs = [v['error_db'] for v in r['regions'].values() if 'error_db' in v]
        if name.startswith('fastsar'):
            t, est = r.get('seconds'), False
        elif name.startswith('isce3'):
            t, est = r.get('full_image_estimate_seconds'), True
        else:
            continue
        rows.append(dict(collection=tag, implementation=name, seconds=t, estimated=est,
                         error_db=[min(errs), max(errs)] if errs else None, pulses=r.get('pulses'), samples=r.get('samples'),
                         sicd=r.get('sicd'), grid=r.get('grid')))
json.dump(rows, open(out, 'w'), indent=1)
for x in rows:
    e = x['error_db']
    print(f"{x['collection']:10s} {x['implementation']:14s} {x['seconds'] or 0:10.1f} s{' (est.)' if x['estimated'] else '       '} "
          f"{'' if e is None else f'{e[1]:.1f} to {e[0]:.1f} dB'}")
