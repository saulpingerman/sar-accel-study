"""Second held-out round: choose the collections at random, before the fixes of the first round are written, so that
the fixes cannot be tuned to them. Excludes every collection used in development or in the first held-out round.
Writes heldout_round2.txt (label|bucket|cphd key|sicd key). Pass criteria are in HELDOUT_ROUND2.md.
   python3 heldout_round2_select.py <capella_cphd.txt> <capella_sicd.txt> <umbra_cphd.txt> <umbra_sicd.txt> <iceye_cphd.txt> <out>
(the inputs are `aws s3 ls --recursive` listings of the open-data buckets, 2026-10-09)"""
import random, re, sys

SEED = 20261009
cap_cphd, cap_sicd, umb_cphd, umb_sicd, ice_cphd, out = sys.argv[1:7]
rng = random.Random(SEED)


def rows(path):
    for line in open(path):
        p = line.rstrip('\n').split(None, 3)
        if len(p) == 4:
            yield int(p[2]), p[3]


# used in development (the paper's collections) or in the first held-out round
USED_CAPELLA = {'20241004204249', '20250303131732', '20220930212640', '20211112205103', '20251208125808',     # development
                '20240921193754', '20251102175916', '20220227175416', '20240514051800', '20260821153140',     # round 1
                '20221011221232', '20260627074032'}
USED_UMBRA = ('Panama', 'Melbourne', 'Johnston', 'Bingham', 'David Fuentes', 'Tanna', 'Silver Peak')       # sites
USED_ICEYE = {'9307864', '9239184', '6770949'}

lines = []
# Capella: two spotlights, two sliding spotlights, two stripmaps, each from a satellite not used in development or
# round 1 where possible, with a SICD of the same collection, CPHD under 15 GB
sicd = {}
for size, key in rows(cap_sicd):
    m = re.search(r'_SICD_\w\w_(\d{14})_', key)
    if m:
        sicd[m.group(1)] = key
cands = {'SP': [], 'SS': [], 'SM': []}
for size, key in rows(cap_cphd):
    b = key.split('/')[-1].split('_')
    sat, mode, t0 = b[1], b[2], b[5]
    if t0 in sicd and t0 not in USED_CAPELLA and size < 15e9 and mode in cands:
        cands[mode].append((sat, key, sicd[t0]))
used_sats = {'C02', 'C03', 'C08', 'C09', 'C10', 'C11', 'C14', 'C15', 'C17', 'C18', 'C20'}
for mode in ('SP', 'SS', 'SM'):
    pool = sorted(cands[mode])
    fresh = [c for c in pool if c[0] not in used_sats] or pool
    picks = []
    for c in rng.sample(fresh, len(fresh)):
        if c[0] not in {p[0] for p in picks}:
            picks.append(c)
        if len(picks) == 2:
            break
    for sat, ck, sk in picks:
        lines.append(f"cap2_{sat.lower()}_{mode.lower()}|capella-open-data|{ck}|{sk}")
# Umbra: four collections from sites and satellites not used, CPHD 2 to 15 GB, with the task's SICD
usicd = {k.rsplit('/', 1)[0]: k for _, k in rows(umb_sicd)}
ucands = []
for size, key in rows(umb_cphd):
    d = key.rsplit('/', 1)[0]
    site = key.split('/')[2]
    m = re.search(r'(UMBRA-\d+)', key)
    if d in usicd and 2e9 < size < 15e9 and not any(u in site for u in USED_UMBRA) and m:
        ucands.append((m.group(1), site, key, usicd[d]))
picks = []
for c in rng.sample(sorted(ucands), len(ucands)):
    if c[0] not in {p[0] for p in picks} and c[1] not in {p[1] for p in picks}:
        picks.append(c)
    if len(picks) == 4:
        break
for sat, site, ck, sk in picks:
    lines.append(f"umb2_{sat.lower().replace('-', '')}_{re.sub(r'[^a-z]', '', site.lower())[:10]}|umbra-open-data-catalog|{ck}|{sk}")
# ICEYE: the remaining collections of satellites not used, one dwell and one spot where available
ice = []
for size, key in rows(ice_cphd):
    if not key.endswith('.cphd'):
        continue
    m = re.search(r'ICEYE_(X\d+)_CPHD_(\w+?)_(\d+)_', key.split('/')[-1])
    if m and m.group(3) not in USED_ICEYE and m.group(1) not in ('X38', 'X49', 'X50'):
        ice.append((m.group(2), m.group(1), key))
for mode, sat, ck in sorted(ice):
    sk = ck.rsplit('/', 1)[0] + '/' + ck.split('/')[-1].replace('_CPHD_', '_SICD_').replace('.cphd', '.nitf')
    lines.append(f"ice2_{sat.lower()}_{mode.lower()}|iceye-open-data-catalog|{ck}|{sk}")
open(out, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(l.split('|')[0] for l in lines))
