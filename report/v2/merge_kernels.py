"""Merge the fused-kernel runs into the results of record.

  python merge_kernels.py            (reads gs://.../v2/{pallas,pallas2,pallas3,pallas4,pallas5,cuda,cuda2}/<scene>/<label>/timing.json)

Writes
  results/v2r/timing/<scene>_<label>.json   the final builds' records under their tags (pallas4 for the TPUs, cuda6
                                            for the L4), alongside the existing records
  results/v2r/kernels.json                  every build's Panama device time per device and precision, for the
                                            kernel ablation table
"""
import json, os, subprocess

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'v2r')
B = 'gs://' + os.environ.get('GCS_BUCKET', 'set-GCS_BUCKET') + '/v2'
SCENES = ['panama', 'melbourne', 'iowa']
BUILDS = [('baseline', None), ('pallas', 'pallas'), ('pallas2', 'pallas2'), ('pallas3', 'pallas3'), ('pallas4', 'pallas4'), ('pallas5', 'pallas5'),
          ('cuda', 'cuda'), ('cuda2', 'cuda2'), ('cuda3', 'cuda3'), ('cuda4', 'cuda4'), ('cuda5', 'cuda5'), ('cuda6', 'cuda6')]
FINAL = {'tpu-v6e': 'pallas4', 'tpu-v5e': 'pallas4', 'gpu-l4': 'cuda6'}
LABELS = ['tpu-v6e', 'tpu-v5e', 'gpu-l4']


def fetch(path):
    try:
        out = subprocess.run(['gcloud', 'storage', 'cat', path], capture_output=True, text=True, check=True).stdout
        return json.loads(out)
    except Exception:
        return None


def main():
    kernels = {}
    for build, d in BUILDS:
        if d is None:
            continue
        for lab in LABELS:
            t = fetch(f'{B}/{d}/panama/{lab}/timing.json')
            if not t:
                continue
            for tag, rec in t.items():
                if tag.startswith('_') or not isinstance(rec, dict) or 'run_s' not in rec:
                    continue
                st = rec.get('stream') or {}
                kernels.setdefault(lab, {}).setdefault(build, {})[tag] = dict(run_s=rec['run_s'], s_per_image=st.get('s_per_image'))
    json.dump(kernels, open(f'{R}/kernels.json', 'w'), indent=1)
    for lab, d in FINAL.items():
        for s in SCENES:
            t = fetch(f'{B}/{d}/{s}/{lab}/timing.json')
            if not t:
                print('missing', d, s, lab)
                continue
            p = f'{R}/timing/{s}_{lab}.json'
            cur = json.load(open(p)) if os.path.exists(p) else {}
            n = 0
            for tag, rec in t.items():
                if tag.startswith('_') or not isinstance(rec, dict):
                    continue
                if 'pallas2' in tag or 'cuda' in tag:
                    cur[tag] = rec
                    n += 1
            json.dump(cur, open(p, 'w'), indent=1)
            print(s, lab, d, n, 'records merged')


if __name__ == '__main__':
    main()
