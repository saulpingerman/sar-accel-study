#!/usr/bin/env python3
"""Print bench.py JSON lines as one compact line each."""
import json, sys
for l in sys.stdin:
    if not l.startswith('{'):
        continue
    r = json.loads(l)
    if 'error' in r:
        print(f"{r['algo']:7s} {r['policy']:11s} N={r['N']} ERROR {r['error'][:220]}")
    elif 'skipped' in r:
        print(f"{r['algo']:7s} {r['policy']:11s} N={r['N']} skipped ({r['skipped']})")
    else:
        print(f"{r['algo']:9s} {r['policy']:11s} N={r['N']:5d} run {r['run_s']*1e3:10.1f} ms  rate {r['rate']:.3e}  first {r.get('first_s', 0):.1f}s", flush=True)
