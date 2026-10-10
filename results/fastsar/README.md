# Release records (paper inputs)

Every time and image of the paper's results comes from FastSAR's public interface (`ImageFormer`, `ExactFormer`,
`form_image`) at a release-candidate tag of the FastSAR repository. The library code of each tag differs from
release 0.1.0 only where noted, and each record names its commit.

| Records | Device | Tag | Differs from 0.1.0 in |
|---|---|---|---|
| `timing/*_cpu-c4d16.json`, `timing/*_tpu-v6e.json`, `timing/*_tpu-v5e.json` | c4d-highmem-16, TPU v6e, TPU v5e | `rel-rc3-20261009` | CUDA path, GPU memory settings |
| `timing/*_gpu-l4.json` | L4 (g2-standard-4) | `rel-rc5-20261009`; `bp/cubic` rows `rel-rc6-20261009` (d20d650) | one docstring; rc6: `ExactFormer` default `upsample=8` |
| `timing/panama_cpu-c4d16.json` `bp/cubic` row | c4d-highmem-16 | `rel-rc7-20261009` (4f14e4b) | `upsample=8` default, CPHD reading fixes (changelog 0.1.1) |
| Capella `../oss/modes/all/c*X`, `*X` (CPU, v6e) | c4d-highmem-16, v6e | `rel-rc2-20261009` | CUDA path, GPU memory settings |
| Capella `../oss/modes/all/g*Y` (L4) | L4 (g2-standard-32) | `rel-rc5-20261009` | one docstring |
| `crossover/crossover_{cpu,gpu}.json` | c4d-highmem-16, L4 (g2-standard-4) | `rel-rc7-20261009` (4f14e4b) | `upsample=8` default, CPHD reading fixes |
| `exact_regions.json` | c4d-highmem-16 | `rel-rc7-20261009` (4f14e4b) | `upsample=8` default, CPHD reading fixes |
| `truth64/`, `tpu_exact/rc6` | c4d-highmem-16, v5e, v6e | `rel-rc6-20261009` (version 0.1.1) | `upsample=8` default |

`git diff <tag> v0.1.0 -- fastsar/` in the FastSAR repository shows each difference.

Produced by the scripts in `../../oss`:

- `timing/<scene>_<device>.json`: `release_record.py`, one process per configuration; the first call, the better of
  two warm calls (`runs` holds both), a back-to-back loop of at least four images and 60 s, the monitor samples.
- `<scene>_metrics.json`: `../../metrics.py` on the saved images against the float64 reference; the family gain
  is fitted on the device's float32-class image.
- `pareto_regions_v3.json`: `region_errors.py`, errors over the lock, port and ship regions.
- `exact_regions.json`: `exact_regions.py`, exact backprojection on the three regions (the open-source estimates'
  method applied to FastSAR).
- `fastsar_pfa_scores_v3.json`: `pfa_region_scores.py`.
- `crossover/`: `crossover.py`, exact against factorized backprojection on n by n grids at the scene center.
- `geometry_facts.json`: `geometry_facts.py`, the plan and polar-format geometry of the released code.
- `logs/`: test-suite and record logs of each device, metrics logs.

Historical files, kept from the development records and used only where the paper says so:
`kernels.json` (device times of the development builds), `bounds.json` and `profiles/` (stage bounds of the profiled
builds), `logs/profile_*.json` (stage profile of the JAX program), `edge_panama.json` and `ptaps_panama.json`
(polar-format resampling variants), `pfa_center_crop.json`, `timing/*_ref.json` (the float64 reference's own time),
`historical.json` (the reference's convergence), `*.info`, `sicd/`, `panama_sicd.xml` and `ship/` (reference
against the vendor image; ship motion), `figs/*_overview*` (images of the reference).

## Record formats

- `timing/<scene>_<device>.json`: `_meta` (scene, grid, pulse and sample counts, the FastSAR commit and version,
  the host label and the protocol), then one entry per configuration (`ffbp/fp32_cpp`, `bp/cubic`, ...) with
  `api` (the public call used), `first_s` (the compile call), `run_s` (the better of two warm calls, `runs` holds
  both), `h2h` and `h2h_stage` (whether the time is host memory to host memory, and how), `monitor` (CPU time and,
  on the L4, the `nvidia-smi` power samples) and `stream` (the back-to-back loop: images formed, elapsed seconds,
  seconds per image).
- `<scene>_metrics.json`: per configuration, the error in dB (`err_db`), the largest pixel difference
  (`max_db`), the 5 by 5 coherence (mean, 0.1 percentile, minimum), the amplitude ratio and phase difference
  (99th percentile, maximum) and the ring errors, as `metrics.py` writes them.
- `pareto_regions_v3.json`, `truth64/score_truth64.json`: errors per region (lock, port, ship) against the
  reference and against the 64-times computation; `exact_regions.json`: the region times and the full-image
  estimate of exact backprojection; `crossover/crossover_<device>.json`: times by grid size for both algorithms;
  `tpu_exact/rc6/tpu_exact_<device>.json`: reads per second, grid error and full-image estimate on the TPUs.
- `logs/tests_<device>[_rcN].log`: the FastSAR test suite on each device at the candidate named in the file name;
  `logs/rec_<device>.log`: the record run's log; `logs/xover_*.log`: an earlier crossover run superseded by the
  JSON files.
