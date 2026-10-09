# Release records (paper inputs)

Every time and image of the paper's results comes from FastSAR's public interface (`ImageFormer`, `ExactFormer`,
`form_image`) at a release-candidate tag of the FastSAR repository. The library code of each tag differs from
release 0.1.0 only where noted, and each record names its commit.

| Records | Device | Tag | Differs from 0.1.0 in |
|---|---|---|---|
| `timing/*_cpu-c4d16.json`, `timing/*_tpu-v6e.json`, `timing/*_tpu-v5e.json` | c4d-highmem-16, TPU v6e, TPU v5e | `rel-rc3-20261009` | CUDA path, GPU memory settings |
| `timing/*_gpu-l4.json` | L4 (g2-standard-4) | `rel-rc5-20261009` | one docstring |
| Capella `../oss/modes/all/c*X`, `*X` (CPU, v6e) | c4d-highmem-16, v6e | `rel-rc2-20261009` | CUDA path, GPU memory settings |
| Capella `../oss/modes/all/g*Y` (L4) | L4 (g2-standard-32) | `rel-rc5-20261009` | one docstring |
| `crossover/crossover_{cpu,gpu}.json` | c4d-highmem-16, L4 (g2-standard-4) | `rel-rc5-20261009` | one docstring |
| `exact_regions.json` | c4d-highmem-16 | `rel-rc3-20261009` | CUDA path, GPU memory settings |

`git diff <tag> v0.1.0 -- fastsar/` in the FastSAR repository shows each difference.

Produced by the scripts in `../../oss`:

- `timing/<scene>_<device>.json`: `release_record.py`, one process per configuration; the first call, the better of
  two warm calls (`runs` holds both), a back-to-back loop of at least four images and 60 s, the monitor samples.
- `<scene>_metrics.json`: `../../v2_metrics.py` on the saved images against the float64 reference; the family gain
  is fitted on the device's float32-class image.
- `pareto_regions_v3.json`: `region_errors_v3.py`, errors over the lock, port and ship regions.
- `exact_regions.json`: `exact_regions.py`, exact backprojection on the three regions (the open-source estimates'
  method applied to FastSAR).
- `fastsar_pfa_scores_v3.json`: `pfa_region_scores.py`.
- `crossover/`: `crossover.py`, exact against factorized backprojection on n by n grids at the scene center.
- `geometry_facts.json`: `geometry_facts.py`, the plan and polar-format geometry of the released code.
- `logs/`: test-suite and record logs of each device, metrics logs.

Historical files, kept from the development records (`../v2r`) and used only where the paper says so:
`kernels.json` (device times of the development builds), `bounds.json` and `profiles/` (stage bounds of the profiled
builds), `logs/profile_*.json` (stage profile of the JAX program), `edge_panama.json` and `ptaps_panama.json`
(polar-format resampling variants), `pfa_center_crop.json`, `timing/*_ref.json` (the float64 reference's own time),
`historical.json` (the reference's convergence), `*.info`, `sicd/`, `panama_sicd.xml` and `ship/` (reference
against the vendor image; ship motion), `figs/*_overview*` (images of the reference).
