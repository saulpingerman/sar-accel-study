"""Records of the released FastSAR (public API only) on one Umbra collection and one device, in the format of the
study's timing and metrics inputs: for each configuration the first call, a warm single image (host memory to host
memory) and a back-to-back loop of at least four images and 60 s (on JAX and TPU the next history is staged on a
worker thread while the current image forms, as form_mosaic does), with utilization and power sampled where the
device reports them; the warm image saved as <out>/<tag with _ for />.npy and <out>/timing.json with every record
and the API call that produced it.
   python -I release_record.py <scene .npz> <out dir> <label> <configs, comma list> [no-loop configs]
configs: ffbp_cpu, ffbp_cuda32, ffbp_cuda16, ffbp_tpu3, ffbp_tpu1, ffbp_jax, pfa, bp_cubic, bp_linear, bp_linear16"""
import os, sys, json, time, threading, subprocess, platform
import numpy as np
import fastsar

data, outd, label, names = sys.argv[1:5]
noloop = set(sys.argv[5].split(',')) if len(sys.argv) > 5 else set()
os.makedirs(outd, exist_ok=True)
d = np.load(data)
S, ant, fmin, df = d['S'], d['ant'], float(d['fmin']), float(d['df'])
nx, ny, spx, spy, e1, e2 = int(d['nx']), int(d['ny']), float(d['spx']), float(d['spy']), d['e1'], d['e2']
P, K = S.shape
scene = os.path.basename(data).split('.')[0]
G = dict(spx=spx, spy=spy, e1=e1, e2=e2)

# configuration -> (tag in the study's tables, API call text, builder)
CONFIGS = {
    'ffbp_cpu': ('ffbp/fp32_cpp', "ImageFormer(..., backend='cpu')", lambda: fastsar.ImageFormer(ant, fmin, df, K, nx, ny, backend='cpu', **G)),
    'ffbp_cuda32': ('ffbp/fp32_cuda', "ImageFormer(..., backend='cuda')", lambda: fastsar.ImageFormer(ant, fmin, df, K, nx, ny, backend='cuda', **G)),
    'ffbp_cuda16': ('ffbp/f16_cuda', "ImageFormer(..., backend='cuda', precision='float16')",
                    lambda: fastsar.ImageFormer(ant, fmin, df, K, nx, ny, backend='cuda', precision='float16', **G)),
    'ffbp_tpu3': ('ffbp/fp32_high_pallas2_direct', "ImageFormer(..., backend='tpu')", lambda: fastsar.ImageFormer(ant, fmin, df, K, nx, ny, backend='tpu', **G)),
    'ffbp_tpu1': ('ffbp/fp32_fast_pallas2_direct', "ImageFormer(..., backend='tpu', precision='single-pass')",
                  lambda: fastsar.ImageFormer(ant, fmin, df, K, nx, ny, backend='tpu', precision='single-pass', **G)),
    'ffbp_jax': ('ffbp/fp32', "ImageFormer(..., backend='jax')", lambda: fastsar.ImageFormer(ant, fmin, df, K, nx, ny, backend='jax', **G)),
    'bp_cubic': ('bp/cubic', "ExactFormer(...)", lambda: fastsar.ExactFormer(ant, fmin, df, K, nx, ny, **G)),
    'bp_linear': ('bp/linear', "ExactFormer(..., interp='linear')", lambda: fastsar.ExactFormer(ant, fmin, df, K, nx, ny, interp='linear', **G)),
    'bp_linear16': ('bp/linear_ov16', "ExactFormer(..., interp='linear', upsample=16)",
                    lambda: fastsar.ExactFormer(ant, fmin, df, K, nx, ny, interp='linear', upsample=16, **G)),
}


class Monitor:
    """nvidia-smi utilization and power every 0.5 s on a GPU host; process CPU utilization everywhere."""
    def __init__(self):
        self.s, self.stop, self.c0, self.t0 = [], False, os.times(), time.perf_counter()
        self.gpu = subprocess.run(['which', 'nvidia-smi'], capture_output=True).returncode == 0
        if self.gpu:
            self.th = threading.Thread(target=self.run, daemon=True); self.th.start()

    def run(self):
        while not self.stop:
            r = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            try:
                self.s.append(tuple(float(v) for v in r.stdout.strip().split(',')))
            except ValueError:
                pass
            time.sleep(0.5)

    def done(self):
        self.stop = True
        c1, el = os.times(), time.perf_counter() - self.t0
        m = dict(cpu_util=((c1.user - self.c0.user) + (c1.system - self.c0.system)) / el / (os.cpu_count() or 1), ncpu=os.cpu_count())
        if self.s:
            a = np.array(self.s)
            m.update(gpu_util_mean=float(a[:, 0].mean()), gpu_watts_mean=float(a[:, 1].mean()), gpu_watts_max=float(a[:, 1].max()))
        return m


def pfa_call():
    return fastsar.form_image(S, ant, fmin, df, nx, ny, algorithm='pfa', **G)


tp = f'{outd}/timing.json'
rec = json.load(open(tp)) if os.path.exists(tp) else {}
commit = subprocess.run(['git', '-C', os.path.dirname(fastsar.__path__[0]), 'log', '-1', '--format=%H'], capture_output=True, text=True).stdout.strip()
rec['_meta'] = dict(label=label, name=scene, nx=nx, ny=ny, spx=spx, spy=spy, e1=list(map(float, e1)), e2=list(map(float, e2)),
                    pulses=int(P), samples=int(K), crop='', fastsar=commit, fastsar_version=fastsar.__version__,
                    host=platform.node(), protocol='host memory to host memory, warm; loop back to back (JAX/TPU: next history staged on a worker thread)')
for name in names.split(','):
    if name == 'pfa':
        tag, call = 'pfa/fp32_taps_corr', "form_image(..., algorithm='pfa')"
        f = pfa_call
        stage = None
        t = time.perf_counter(); img = f(); first = time.perf_counter() - t
    else:
        tag, call, build = CONFIGS[name]
        t = time.perf_counter(); former = build(); setup = time.perf_counter() - t
        f = lambda: former(S)
        stage = getattr(former, 'stage', None) if name in ('ffbp_tpu3', 'ffbp_tpu1', 'ffbp_jax') else None
        t = time.perf_counter(); img = f(); first = time.perf_counter() - t
    mon = Monitor()
    runs = []
    for _ in range(2):                 # the better of two warm calls (one-time work can fall on the second call)
        t = time.perf_counter(); img = f(); runs.append(time.perf_counter() - t)
    run = min(runs)
    m1 = mon.done()
    np.save(f'{outd}/{tag.replace("/", "_", 1)}.npy', img)
    del img
    r = dict(api=call, first_s=first, run_s=run, runs=runs, h2h=True, h2h_stage='release', monitor=m1)
    if name not in noloop:
        mon = Monitor(); n = 0; t = time.perf_counter()
        if stage is not None:          # stage the next history while the current one forms
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(1) as ex:
                nxt = ex.submit(stage, S)
                while n < 4 or time.perf_counter() - t < 60:
                    cur = nxt.result(); nxt = ex.submit(stage, S)
                    former(cur); n += 1
                nxt.result()
        else:
            while n < 4 or time.perf_counter() - t < 60:
                f(); n += 1
        el = time.perf_counter() - t
        r['stream'] = dict(images=n, seconds=el, s_per_image=el / n, images_per_s=n / el, monitor=mon.done())
    rec[tag] = r
    print(scene, label, tag, f'first {first:.2f} s, run {run:.2f} s' + (f", loop {r['stream']['s_per_image']:.2f} s over {r['stream']['images']}" if 'stream' in r else ''),
          m1.get('gpu_watts_mean'), flush=True)
    json.dump(rec, open(tp, 'w'), indent=1)
    if name != 'pfa':
        del former
