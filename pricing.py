#!/usr/bin/env python3
"""Hourly prices by region and purchase mode from a Cloud Billing Catalog snapshot.

  python pricing.py <ce_skus.json>      first run: filter the full snapshot to the SKUs used here
  python pricing.py                     rebuild results/pricing/prices.json from the filtered snapshot

Instance prices are assembled from their parts: an L4 instance is one GPU plus
the G2 cores and memory of its host; a TPU is priced per chip with the host
included; the CPU instance is 16 C4D cores and 62 GB.
"""
import json
import re
import sys

KEEP = re.compile(r'TpuV5e|TpuV6e|Nvidia L4|Nvidia Tesla T4|N1 Predefined Instance (Core|Ram)|G2 Instance|C4D Instance (Core|Ram)|Commitment v1: (C4D|G2)')
SNAP = 'results/pricing/skus_2026-10-01.json'


def price(s):
    r = s['pricingInfo'][0]['pricingExpression']['tieredRates'][-1]['unitPrice']
    return int(r.get('units', 0) or 0) + r.get('nanos', 0) / 1e9


def main():
    if len(sys.argv) > 1:
        sk = [s for s in json.load(open(sys.argv[1])) if KEEP.search(s['description']) and 'Sole' not in s['description']]
        json.dump(sk, open(SNAP, 'w'))
    sk = json.load(open(SNAP))
    P = {}                                        # (item, mode) -> {region: $/h}
    for s in sk:
        d, mode = s['description'], s['category']['usageType']
        if 'Capacity Optimized' in d or 'Calendar' in d or 'Local SSD' in d:
            continue
        if 'DWS' in d:
            mode = 'Flex'
        item = ('l4' if 'Nvidia L4' in d else 'v5e' if 'TpuV5e' in d else 'v6e' if 'TpuV6e' in d else
                'g2core' if re.search(r'G2 (Instance )?(Core|Cpu)', d) else 'g2ram' if re.search(r'G2 (Instance )?Ram', d) else
                'c4dcore' if re.search(r'C4D (Instance )?(Core|Cpu)', d) else 'c4dram' if re.search(r'C4D (Instance )?Ram', d) else None)
        if item is None:
            continue
        for reg in s['serviceRegions']:
            P.setdefault((item, mode), {})[reg] = price(s)
    regions = sorted({r for v in P.values() for r in v})
    out = {}
    for reg in regions:
        e = {}
        for mode in ('OnDemand', 'Preemptible', 'Commit1Yr', 'Commit3Yr'):
            def g(item, fallback=None):
                v = P.get((item, mode), {}).get(reg)
                return v if v is not None else (P.get((item, fallback), {}).get(reg) if fallback else None)
            m = {}
            for dev in ('v5e', 'v6e'):
                if g(dev) is not None:
                    m[dev] = g(dev)
            # host cores and memory at the committed rate where one is listed, otherwise on demand
            parts = [g('l4'), g('g2core', 'OnDemand'), g('g2ram', 'OnDemand')]
            if all(v is not None for v in parts):
                m['l4_g2s4'] = parts[0] + 4 * parts[1] + 16 * parts[2]
                m['l4_g2s8'] = parts[0] + 8 * parts[1] + 32 * parts[2]
            parts = [g('c4dcore', 'OnDemand'), g('c4dram', 'OnDemand')]
            if all(v is not None for v in parts):
                m['c4d16'] = 16 * parts[0] + 62 * parts[1]
            e[mode] = m
        out[reg] = e
    json.dump(out, open('results/pricing/prices.json', 'w'), indent=1)
    for reg in ('us-central1', 'us-east5', 'us-west4', 'us-west2', 'us-east1'):
        print(reg, {k: {d: round(v, 4) for d, v in m.items()} for k, m in out.get(reg, {}).items()})
    print(len(regions), 'regions;', sorted({k for k in P}))


if __name__ == '__main__':
    main()
