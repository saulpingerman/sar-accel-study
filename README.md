# FastSAR on CPUs, GPUs and TPUs: measurement records and paper build

Code, measurement records and paper build for *FastSAR: High-Throughput, High-Fidelity Open-Source SAR Image
Formation on CPUs, GPUs and TPUs* (Singerman and Braun, Penn State University, 2026). The paper is `paper/paper.pdf`; every
number in it is computed by `paper/build.py` from the records under `results/fastsar` (release records of the
[FastSAR](https://github.com/saulpingerman/FastSAR) library through its public interface) and `results/comparison`
(the open-source implementations and the Capella collections). Three Umbra spotlight collections are imaged at
native size (8,500 to 12,200 pixels on a side) on a Google Cloud TPU v5e, a TPU v6e, an NVIDIA L4 and a 16-vCPU
AMD instance by exact backprojection, factorized backprojection and polar format, and every image is compared
with a float64 exact backprojection; five Capella collections in three modes extend the comparison.

## What produces each table and figure

| Paper item | Script | Inputs | Device, time, cost |
|---|---|---|---|
| Table 1 (collections), Table 2 (platforms) | `paper/build.py` | `results/fastsar/*.info`, `pricing.py` | none |
| Figure 1 (cost against error), Table 4 and Table 15 (cost) | `oss/release_record.py` per scene and device; `oss/region_errors.py` | `results/fastsar/timing/*.json`, `results/fastsar/pareto_regions_v3.json`, `results/fastsar/truth64/` | one run per device: c4d-highmem-16 about 1 h, L4 and each TPU about 1 h; under $10 in all |
| Table 3 (open-source comparison), Table 12 (polar format) | `oss/*_panama.py`, `oss/nga_*.py`, `oss/isce3_compare.py`, `oss/torchbp_panama.py`, `oss/grdl_*.py`, `oss/exact_regions.py` | `results/comparison/*.json`, `results/fastsar/oss_times.json`, `results/fastsar/exact_regions.json` | c4d-highmem-16 and L4 (g2-standard-8); hours per implementation, see `results/comparison/PROVENANCE.txt` |
| Figures 3, 4 (image quality), Table 14 (precision) | `metrics.py`, `crops.py` | `results/fastsar/<scene>_metrics.json`, `results/fastsar/figs/panama_crops.npz` (release `figure-data`) | metrics on a CPU instance with the references, about 2 h |
| Table 5 (Capella collections) | `oss/modes_strip.py`, `oss/modes_common.py`, `oss/modes_table.py` | `results/comparison/modes/all/*/` (FastSAR tags `c*X`, `g*Y`, `*X`; ISCE3 `*e`, `ss2022ib`; GRDL `sp2024`, `sp2025`, `sm2021g`, `sm2025`), `results/comparison/modes/collections.json` | c4d-highmem-16, g2-standard-32, v6e; 1 to 6 h per collection |
| Tables 8, 9 (kernel bounds and builds) | `stage_profile.py`, `profile_cuda.py`, `ubench_*.py`, `paper/bounds.py` | `results/fastsar/bounds.json`, `results/fastsar/kernels.json`, `results/fastsar/profiles/` (development records, see the ledger) | development runs of September and October 2026 |
| Figure 5, Table 10 (polar format) | `form.py`, `metrics.py` | `results/fastsar/edge_panama.json`, `results/fastsar/ptaps_panama.json`, `<scene>_metrics.json` | CPU instance |
| Figure 6, Table 11 (vendor image) | `compare_sicd.py` | `results/fastsar/sicd/`, release `figure-data` crops | CPU instance, about 30 min |
| Appendix C (exact backprojection on the TPU) | `oss/exact_tpu.py` | `results/fastsar/tpu_exact/rc6/` | v5e and v6e, under 1 h each |
| Figure 7, Appendix I (ground truth, ship) | `geo_panama.py`, `eo.py`, `ship_refocus.py`, `ship_fig.py` | `results/fastsar/ship/`, release `figure-data` windows | CPU instance |

`results/fastsar/README.md` is the ledger: it names the FastSAR tag and commit of every record and describes the
record files. `results/comparison/README.md` indexes the open-source comparison scripts and records.

## Layout

| Path | Contents |
|---|---|
| `oss/` | The comparison scripts: FastSAR release records (`release_record.py`, `exact_record.py`, `exact_regions.py`, `crossover.py`, `region_errors.py`, `pfa_region_scores.py`, `geometry_facts.py`), the open-source implementations on Panama (RITSAR, the AFRL/NGA MATLAB toolbox in Octave, ISCE3, torchbp, GRDL) and the Capella modes comparison (`modes_*.py`, `grdl_strip*.py`) |
| `results/fastsar/` | Release records behind the paper (ledger in its README) |
| `results/comparison/` | Records of the open-source comparison and of the Capella collections (`modes/all/<tag>/`), with `DETAILS.md` (the long account) and `PROVENANCE.txt` (instances) |
| `dev/` | The development versions of the algorithms the kernels were built in (`ffbp2.py`, `pallas_ffbp.py`, `ffbp_cuda.py`, `pfa2.py`, `cpu_ref.py` the float64 reference, `ffbp_cpu.cpp`); the released code is the FastSAR package |
| `prep.py`, `form.py`, `metrics.py`, `crops.py`, `cpu_stream.py`, `stage_profile.py`, `profile_*.py`, `ubench_*.py` | Preparation of a collection from its CPHD and SICD, image formation and timing with the development code, image statistics against the reference, figure crops, CPU throughput, stage profiles and unit-rate microbenchmarks |
| `compare_sicd.py`, `geo_panama.py`, `eo.py`, `ship_refocus.py`, `ship_fig.py`, `umbra_find.py`, `pricing.py` | Reference against the vendor's image; pixel-to-ground mapping; Sentinel-2 and multi-pass checks; the ship refocus; the Umbra archive index; on-demand prices from a billing-catalog snapshot |
| `cloud/` | Instance setup, launch and run scripts for Google Cloud; they expect `GCP_PROJECT` and `GCS_BUCKET` in the environment |
| `paper/` | `paper.src.tex` with `@@token@@` placeholders, `build.py` (tables and numbers from the records, then pdflatex), `figs.py`, `bounds.py`, `fig/`, `numbers.json` (every number the text uses), the built `paper.pdf` and `supplement_tables.pdf` |
| `tests/` | Checks of the development kernels against the dense JAX image on a simulated scene |

## Data

The radar data are not redistributed. The three Umbra spotlight collections (Panama Canal 2023-07-18, Melbourne
2023-02-08, Johnston, Iowa 2023-10-20) come from the [Umbra Open Data Program](https://umbra.space/open-data)
(CC BY 4.0); `cloud/setup_cpu.sh` has the keys. The five Capella collections come from the
[Capella Space Open Dataset](https://registry.opendata.aws/capella_opendata/); `results/comparison/modes/collections.json`
names each by its core name and collector. The optical images are Copernicus Sentinel-2 Level-2A tiles read from
the AWS open-data mirror (`eo.py`).

The binary inputs of the figures (image crops of the release records, reference-against-vendor crops, downsampled
overviews, Sentinel-2 windows, geocoded windows of other Umbra passes, the refocused ship tiles; about 350 MB) are
attached to the GitHub release `figure-data`. `fetch_data.sh` downloads them into `results/fastsar/`.

## Reproducing the paper from the stored records

```
./fetch_data.sh                                 # figure inputs from the releases, into results/fastsar/
pip install -r requirements-report.txt          # numpy, scipy, matplotlib
python3 paper/figs.py                       # figures from results/fastsar (fastsar must be importable for two of them)
python3 paper/build.py                      # tables and numbers into paper.tex, then pdflatex
```

`build.py` needs `pdflatex` with `booktabs`, `placeins`, `microtype`, `subcaption` and `hyperref` (any TeX Live of
the last few years). `paper/numbers.json` lists every number the text uses, and `results/fastsar/README.md` and
`results/comparison/README.md` say which record each one comes from.

## Reproducing the measurements

The release records were made with FastSAR 0.1.0 and its release candidates `rel-rc2-20261009` to
`rel-rc7-20261009` (the ledger names the tag of each record and what differs from 0.1.0), JAX 0.11.2, CuPy 14.2.0
with CUDA 12 and Numba 0.68.0; `requirements-run.txt` pins them. Each cloud script sets up an instance, downloads
the prepared collections from your bucket, runs its configurations and uploads the records. In order:

1. `cloud/setup_cpu.sh` on a CPU instance with at least 64 GB of memory: downloads the CPHD and SICD files,
   runs `prep.py` for each collection, uploads the `.npz` files.
2. `cloud/ref.sh` on an n2-highmem-48 (384 GB): the float64 references with 16 times oversampled range
   profiles (`REF_OVERSAMPLE=16`); the 64 times computations of the three regions (`oss/fastsar_exact.py`).
3. `oss/release_record.py <scene>.npz <out> <label>` on each device with the FastSAR tag of the ledger installed:
   every configuration of the cost table, one process per configuration, best of two warm calls, the back-to-back
   loop; `oss/exact_record.py` and `oss/crossover.py` for exact backprojection and the crossover; `oss/exact_tpu.py`
   on the TPUs.
4. `cloud/metrics_job.sh` on a CPU instance with the references: `metrics.py` for every saved image,
   `oss/region_errors.py` for the region errors, `crops.py` for the figure crops, `compare_sicd.py` for the
   vendor comparison.
5. The open-source implementations: the scripts named in `results/comparison/README.md`, each on the instance type in
   `results/comparison/PROVENANCE.txt`.
6. The Capella collections: `oss/modes_strip.py` and `oss/modes_common.py` per collection and implementation,
   `oss/modes_table.py` to collect the records.
7. The kernel-build and bounds tables come from the development records kept in `results/fastsar` (`cloud/v2_pallas*_tpu.sh`,
   `cloud/cuda_gpu.sh`, `cloud/cpp_cpu.sh`, `ubench_*.py`, `paper/bounds.py`); `paper/merge_kernels.py`
   pulled them into the record files.

The cloud scripts use on-demand instances in us-central1, us-east1, us-east5 and us-west4 and delete them when
done. The whole study, including the development rounds and the open-source comparison, cost under 400 US dollars at
on-demand prices.

## Citation

`CITATION.cff` gives the software citation and the paper as the preferred citation. An arXiv identifier will be
added when the preprint is posted.

## License

MIT for the code, scripts and paper build (`LICENSE`); the measurement records and the release data under CC BY 4.0
(`LICENSE-DATA.md`). The Umbra, Capella, ICEYE and Copernicus data retain their own licenses.
