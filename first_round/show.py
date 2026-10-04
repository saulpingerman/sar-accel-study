#!/usr/bin/env python3
"""Print a metrics JSON from `quality.py analyze` as compact text."""
import json
import sys

r = json.load(open(sys.argv[1]))
for k, v in r.items():
    if k != 'rows':
        print(k, v)
for row in r['rows']:
    tag = f"{row['algo']}/{row['policy']}"
    if row.get('nonfinite'):
        print(f'{tag:22s} NON-FINITE')
        continue
    c, m = row['ccd'], row['ccd_mixed']
    print(f"{tag:22s} err {row['err_db']:7.1f} dB | coh-vs-fp64 mean {row['coh_vs_fp64_mean']:.5f} p01 {row['coh_vs_fp64_p01']:.4f} "
          f"| phase {row['phase_rms_deg']:6.2f} deg | floor {row['floor_db']:6.1f} dB")
    print(f"{'':22s} CCD same-policy: unch {c['coh_unchanged']:.3f} chg {c['coh_changed']:.3f} auc {c['auc']:.4f} "
          f"pd {c['pd_at_thresh']:.3f} pfa {c['pfa_at_thresh']:.4f} | mixed: unch {m['coh_unchanged']:.3f} auc {m['auc']:.4f} "
          f"| map diff mean {row['ccd_map_mean_abs_diff']:.4f} max {row['ccd_map_max_abs_diff']:.3f}")
    for i, t in enumerate(row['irf'][:3]):
        if 'error' in t:
            print(f"{'':22s} irf{i} error {t['error']}")
            continue
        a, b = t['ax0'], t['ax1']
        print(f"{'':22s} irf{i}: ax0 irw {a['irw']:.3f} m pslr {a['pslr_db']:6.1f} islr {a['islr_db']:6.1f} | "
              f"ax1 irw {b['irw']:.3f} m pslr {b['pslr_db']:6.1f} islr {b['islr_db']:6.1f} | "
              f"peak {t['peak_db']:+.3f} dB {t['peak_phase_deg']:+.2f} deg shift {t['shift_px']:.3f} px")
