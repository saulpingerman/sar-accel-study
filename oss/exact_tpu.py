"""FastSAR's exact backprojection (fastsar.ExactFormer) on one Umbra collection, recorded in the format of the study's
timing and metrics files: per configuration the first call, a warm single image (host memory to host memory) and a
pipelined loop of at least four images and 60 s, with GPU utilization and power sampled by nvidia-smi; the image of
the warm call saved as <out>/bp_<name>.npy for metrics.py, and <out>/timing.json with the run records.
   python -I exact_tpu.py <scene .npz> <out dir> <backend> <label> [configs: name:interp:upsample,...]"""
import os, sys, json, time, threading, subprocess
import numpy as np
import fastsar

data, outd, backend, label = sys.argv[1:5]
configs = [c.split(':') for c in (sys.argv[5] if len(sys.argv) > 5 else 'cubic:cubic:4,linear:linear:8,linear_ov16:linear:16').split(',')]
os.makedirs(outd, exist_ok=True)
d = np.load(data)
S, ant, fmin, df = d['S'], d['ant'], float(d['fmin']), float(d['df'])
nx, ny, spx, spy, e1, e2 = int(d['nx']), int(d['ny']), float(d['spx']), float(d['spy']), d['e1'], d['e2']
scene = os.path.basename(data).split('.')[0]


class Monitor:
    """nvidia-smi utilization and power every 0.5 s (GPU), or process CPU time (CPU)."""
    def __init__(self):
        self.s, self.stop, self.c0, self.t0 = [], False, os.times(), time.perf_counter()
        if backend == 'cuda':
            self.th = threading.Thread(target=self.run, daemon=True); self.th.start()

    def run(self):
        while not self.stop:
            r = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            try:
                u, w = (float(v) for v in r.stdout.strip().split(','))
                self.s.append((u, w))
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


tp = f'{outd}/timing.json'
rec = json.load(open(tp)) if os.path.exists(tp) else {}
rec['_meta'] = dict(label=label, name=scene, nx=nx, ny=ny, spx=spx, spy=spy, e1=list(map(float, e1)), e2=list(map(float, e2)),
                    pulses=int(S.shape[0]), samples=int(S.shape[1]), crop='',
                    fastsar=subprocess.run(['git', '-C', os.path.dirname(fastsar.__path__[0]), 'log', '-1', '--format=%H'], capture_output=True, text=True).stdout.strip())
for name, interp, up in configs:
    former = fastsar.ExactFormer(ant, fmin, df, S.shape[1], nx, ny, spx, spy, e1, e2, backend=backend, interp=interp, upsample=int(up))
    t = time.perf_counter(); img = former(S); first = time.perf_counter() - t
    mon = Monitor()
    t = time.perf_counter(); img = former(S); run = time.perf_counter() - t
    m1 = mon.done()
    np.save(f'{outd}/bp_{name}.npy', img)
    del img
    mon = Monitor(); n = 0; t = time.perf_counter()
    while n < 4 or time.perf_counter() - t < 60:
        former(S); n += 1
    el = time.perf_counter() - t
    m2 = mon.done()
    rec[f'bp/{name}'] = dict(first_s=first, run_s=run, h2h=True, h2h_stage='exact', interp=interp, upsample=int(up), monitor=m1,
                             stream=dict(images=n, seconds=el, s_per_image=el / n, images_per_s=n / el, monitor=m2))
    print(scene, name, f'first {first:.2f} s, run {run:.2f} s, loop {el / n:.2f} s per image over {n}', m1.get('gpu_watts_mean'), flush=True)
    json.dump(rec, open(tp, 'w'), indent=1)
    del former
