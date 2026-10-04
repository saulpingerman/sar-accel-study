#!/usr/bin/env python3
"""Index the Umbra open-data archive: every CPHD file with its size, grouped
by site, and (with --site) the collection geometry of each collect at one
site so that repeat-geometry pairs can be found."""
import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import defaultdict

BASE = 'https://umbra-open-data-catalog.s3.amazonaws.com/'


def list_keys(prefix):
    token = None
    while True:
        q = {'list-type': '2', 'prefix': prefix}
        if token:
            q['continuation-token'] = token
        x = urllib.request.urlopen(BASE + '?' + urllib.parse.urlencode(q)).read().decode()
        for m in re.finditer(r'<Contents><Key>(.*?)</Key>.*?<Size>(\d+)</Size>', x):
            yield m.group(1).replace('&amp;', '&'), int(m.group(2))
        m = re.search(r'<NextContinuationToken>(.*?)</NextContinuationToken>', x)
        if not m:
            return
        token = m.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', help='write the CPHD index to this JSON file')
    ap.add_argument('--site', help='print geometry for every collect under this site prefix (uses the index)')
    ap.add_argument('--from-index', default='umbra_cphd_index.json')
    ap.add_argument('--show-meta', action='store_true')
    a = ap.parse_args()
    if a.index:
        out = []
        for key, size in list_keys('sar-data/tasks/'):
            if key.endswith('.cphd') or key.endswith('METADATA.json'):
                out.append((key, size))
            if len(out) % 2000 == 0 and out:
                print(len(out), key[:90], file=sys.stderr, flush=True)
        json.dump(out, open(a.index, 'w'))
        sites = defaultdict(list)
        for key, size in out:
            if key.endswith('.cphd'):
                sites[key.split('/')[2]].append(size)
        for s, v in sorted(sites.items(), key=lambda kv: -len(kv[1]))[:60]:
            print(f'{len(v):5d} collects  min {min(v) / 1e9:6.2f} GB  median {sorted(v)[len(v) // 2] / 1e9:6.2f} GB  {s}')
        print('total CPHD files', sum(len(v) for v in sites.values()))
        return
    idx = json.load(open(a.from_index))
    if a.show_meta:
        k = [k for k, _ in idx if k.endswith('METADATA.json') and k.split('/')[2] == a.site][0]
        print(k)
        print(json.dumps(json.load(urllib.request.urlopen(BASE + urllib.parse.quote(k))), indent=0)[:3500])
        return
    metas = [k for k, _ in idx if k.endswith('METADATA.json') and k.split('/')[2] == a.site]
    sizes = {k.rsplit('/', 1)[0]: s for k, s in idx if k.endswith('.cphd')}
    rows = []
    for k in metas:
        try:
            m = json.load(urllib.request.urlopen(BASE + urllib.parse.quote(k)))
        except Exception as e:
            continue
        c = m['collects'][0]
        rows.append(dict(key=k.rsplit('/', 1)[0], start=c.get('startAtUTC'), graze=c.get('angleGrazingDegrees'),
                         inc=c.get('angleIncidenceDegrees'), squint=c.get('angleSquintDegrees'),
                         az=c.get('angleAzimuthDegrees'), look=c.get('observationDirection'), orbit=c.get('satelliteTrack'),
                         res_r=c.get('maxGroundResolution', {}).get('rangeMeters'), mode=m.get('imagingMode'),
                         cphd_gb=sizes.get(k.rsplit('/', 1)[0], 0) / 1e9))
    json.dump(rows, open(f'umbra_geom_{a.site.replace(" ", "_").replace(",", "")}.json', 'w'), indent=1)
    best = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            r, s = rows[i], rows[j]
            if None in (r['graze'], s['graze'], r['az'], s['az']) or r['look'] != s['look'] or r['orbit'] != s['orbit']:
                continue
            daz = abs((r['az'] - s['az'] + 180) % 360 - 180)
            best.append((abs(r['graze'] - s['graze']) + daz, abs(r['graze'] - s['graze']), daz, i, j))
    best.sort()
    print(len(rows), 'collects with metadata;', len(best), 'same-side pairs')
    for tot, dg, daz, i, j in best[:12]:
        r, s = rows[i], rows[j]
        print(f"dgraze {dg:5.2f} daz {daz:5.2f}  {r['start']} / {s['start']}  graze {r['graze']:.1f} az {r['az']:.1f} squint {r['squint']} / {s['squint']}  "
              f"cphd {r['cphd_gb']:.1f}/{s['cphd_gb']:.1f} GB\n    {r['key']}\n    {s['key']}")


if __name__ == '__main__':
    main()
