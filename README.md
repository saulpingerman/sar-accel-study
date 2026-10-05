# Cost and precision of spotlight SAR image formation on Google TPUs, an Nvidia GPU and a CPU

Code, measurement records and paper build for the study of that title (Singerman and Braun, Penn State University, 2026). Three Umbra spotlight collections are imaged at native size (8,500 to 12,200 pixels on a side) on a Google Cloud TPU v5e, a TPU v6e, an Nvidia L4 and a 16-vCPU AMD instance by exact backprojection, factorized backprojection expressed as matrix products, and polar format, and every image is compared with a float64 exact backprojection. The paper is `report/v2/sar_accel_v2.pdf`.

The image-formation kernels are also published on their own as [SARfocus](https://github.com/saulpingerman/SARfocus), one Python call with TPU, GPU and CPU backends.

## Layout

| Path | Contents |
|---|---|
| `sarbench/` | The algorithms: `ffbp2.py` (factorized backprojection, JAX, any image plane; filters `dense`, `conv`, `taps`, and `pallas2` for the fused TPU kernels), `pallas_ffbp.py` (the TPU kernels: fused level kernel and recurrence final stage, Pallas), `pallas_ffbp_gpu.py` (the Triton attempt on the GPU, kept for the record), `ffbp_cuda.py` (the CUDA factorized pipeline for the L4: shared-memory FIR levels, CUDA-core and tensor-core final stages), `gpu_bp2.py` (exact backprojection, CUDA kernel via CuPy), `cpu_ref.py` (float64 exact backprojection, Numba, the reference), `pfa2.py` (polar format with chirp-z resampling and the geometric correction), `bp.py` (shared geometry), `ffbp_cpu.cpp` and `ffbp_cpu.py` (the CPU kernels: C++ with OpenMP, compiled with g++ on first use and called through ctypes) |
| `v2_prep.py` | Reads an Umbra CPHD and SICD pair, trims invalid pulses, writes the phase history and the vendor's image grid to one `.npz` |
| `v2_form.py` | Forms images: `ref`, `ffbp`, `cuda`, `pfa` and `ffbpcuda` modes, every arithmetic configuration, timing and pipelined throughput (`--filters pallas2` selects the TPU kernels, `ffbpcuda --cuda-final fp32,f16tc,f16` the CUDA pipeline: float32 throughout, float32 with the tensor-core float16 final stage, and float16 storage throughout with float32 accumulation) |
| `v2_metrics.py` | Error, largest pixel difference, 5 by 5 coherence, amplitude and phase statistics, ring errors against the reference |
| `v2_profile.py`, `profile_level0.py`, `profile_cuda.py`, `ubench_tpu.py`, `ubench_gpu.py`, `ubench_cpu.py`, `v2_cpu_stream.py`, `v2_crops.py` | Per-stage profiles of the JAX program and of the kernel builds (TPU and GPU), the unit-rate microbenchmarks behind the bounds table (memory copy, the kernels' own instruction mixes on resident data, sines, matrix products), CPU multi-process throughput, the image crops of the figures |
| `compare_sicd.py` | The float64 reference against the vendor's SICD image: registration, phase-ramp removal, the displacement field in windows against the planar-wavefront model, and coherence after resampling through the model (the validation appendix) |
| `geo_panama.py`, `v2_eo.py`, `ship_refocus.py`, `v2_ship_fig.py` | Pixel-to-ground mapping from the SICD header, the Sentinel-2 and multi-pass checks of the Panama regions, the moving-target refocus of the ship |
| `cloud/` | Instance launch and run scripts for Google Cloud (`v2_*.sh`); they expect `GCP_PROJECT` and `GCS_BUCKET` in the environment |
| `results/v2r/` | Every measurement the paper uses: `timing/` (device times, pipelined loops, monitors), `*_metrics.json` (image statistics; `*_kern_*` for the kernel builds), `kernels.json` (Panama device time of every kernel build), `bounds.json` (stage bounds from the measured unit rates, made by `report/v2/bounds.py`), `profiles/` (stage profiles of the JAX program and the kernel builds, and the `ubench_*.json` unit rates), `logs/` (run logs), `ship/` (refocus sweeps), `overrides.json` (hand-entered reference-convergence and padding figures) |
| `report/v2/` | `paper.src.tex` with `@@token@@` placeholders, `build.py` (fills the tables and numbers from `results/v2r`, runs pdflatex), `figs.py` (figures), `fig/`, the built PDF and the plain-text abstract |
| `first_round/` | Scripts of the earlier internal round on simulated data and small images, kept for the record; they import `sarbench` from the repository root |
| `tests/` | Checks of the kernels against the dense JAX image on a simulated scene: `test_pallas_fused.py`, `test_pallas_final.py` and `test_pallas_e2e_tpu.py` (TPU kernels in interpret mode on the CPU), `test_pallas_e2e.py` (Triton kernels), `test_ffbp_cuda.py` (CUDA pipeline, needs a GPU); and a bounded-memory check for the factorized plan; `test_ffbp_cpu.py` (C++ kernels against the dense image) |

## Data

The radar data are three collections from the [Umbra Open Data Program](https://umbra.space/open-data) (CC BY 4.0), downloaded directly from the public bucket; `cloud/v2_setup_cpu.sh` has the exact keys. The optical images are Copernicus Sentinel-2 Level-2A tiles read from the AWS open-data mirror (`v2_eo.py`). Neither is redistributed here.

The binary inputs of the figures (image crops, downsampled overviews, Sentinel-2 windows, geocoded windows of other Umbra passes, the refocused ship tiles; 135 MB) are attached to the GitHub release `v2-data` rather than committed. `fetch_data.sh` downloads them into `results/v2r/`.

## Reproducing the paper from the stored measurements

```
./fetch_data.sh                      # figure inputs from the release
pip install numpy scipy matplotlib pymupdf rasterio   # or: uv run --with ... as in the scripts
python3 report/v2/figs.py            # figures from results/v2r
python3 report/v2/build.py           # tables and numbers into paper.tex, then pdflatex
```

`build.py` needs `pdflatex` with `booktabs`, `lscape`, `placeins`, `microtype` and `hyperref` (any TeX Live of the last few years). `report/v2/numbers.json` lists every number the text uses and `results/v2r` holds the records each one is computed from.

## Reproducing the measurements

Each device script sets up an instance, downloads the prepared collections from your bucket, runs every configuration and uploads `timing.json`, the images and logs. On-demand us-central1 prices are in `pricing.py` and in the paper's Table 2. In order:

1. `cloud/v2_setup_cpu.sh` on a CPU instance with at least 64 GB of memory: downloads the CPHD and SICD files, runs `v2_prep.py` for each collection, uploads the `.npz` files.
2. `cloud/v2_ref.sh` on an n2-highmem-48 (384 GB): the float64 references with 16 times oversampled range profiles (`REF_OVERSAMPLE=16`).
3. `cloud/v2_run_accel.sh` through `cloud/v2_launch_tpu.sh` (TPU v5e, v6e) and `cloud/launch_gpu.sh` (L4): every configuration, the profile and the pipelined loops; `cloud/v2_finish_*.sh` are the later additions (direct-ramp three-pass rows, L4 pipelined loops).
4. `cloud/v2_run_cpu.sh` on a c4d-highmem-16: the CPU configurations and `v2_cpu_stream.py`; `cloud/v2_cpp_cpu.sh [dest]` the C++ kernel build (the build of record is `cpp4`) and `cloud/v2_ubench_cpu.sh` its unit-rate microbenchmark.
5. `cloud/v2_metrics_job.sh` on a CPU instance with the references: `v2_metrics.py` for every saved image; `v2_crops.py` for the figure crops; `cloud/v2_sicd_compare.sh` the comparison with the vendor's SICD images.
6. `v2_eo.py` and `ship_refocus.py` (`cloud/v2_ship.sh`) for the ground-truth appendix.
7. The kernel builds of the paper's kernels appendix: `cloud/v2_pallas_tpu.sh`, `v2_pallas2_tpu.sh` and `v2_pallas3_tpu.sh <label> <pb> <nc> <ng> [dest]` on the TPUs (the build of record is `v2_pallas8_tpu.sh`, that is `v2_pallas3_tpu.sh` with 256 8 16 and the generation-3 kernels; `v2_pallas5_tpu.sh` and `v2_pallas7_tpu.sh` are rejected variants kept for the ablation table), `cloud/v2_cuda_gpu.sh [dest]` on the L4 (installs `g++` for nvcc, which the deep-learning image lacks; the builds of record are `cuda6` for float32 and `cuda7`, run with `FINALS=fp32,f16tc,f16`, for float16), and `cloud/v2_metrics_pallas.sh <labels> <dests> <out>` to score the saved images with the family gain taken from the device's six-pass image. `report/v2/merge_kernels.py` pulls the records into `results/v2r`.

The whole study cost about 135 US dollars of cloud time at on-demand prices; the TPU v6e was intermittently unavailable in us-east5 and the scripts retry.

## Software

JAX 0.11 (TPU, GPU and CPU backends; its Pallas extension for the TPU kernels), CuPy 14 and CUDA 12 for the CUDA kernels, Numba 0.68 for the reference, sarpy for reading CPHD and SICD, rasterio for Sentinel-2, Python 3.12. `requirements.txt` lists the Python packages for the analysis and the report; the device backends are installed by the cloud scripts.

## Citation

See `CITATION.cff`. Please cite the arXiv version of the paper once it is posted.

## License

MIT (code, scripts and build). The measurement records in `results/` and the release data may be reused under CC BY 4.0 with attribution to the paper; the Umbra and Copernicus data retain their own licenses.
