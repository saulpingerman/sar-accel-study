# Open-source comparison and Capella collections: scripts and records

The long account is `DETAILS.md` (written 2026-10-07 from the development runs; the FastSAR numbers in it are superseded by `results/fastsar`, as its first paragraph says) and the instances are in `PROVENANCE.txt`. The scripts live in `../../oss/`; this index gives each one's purpose from its own docstring and the records it writes here.

## Scripts

| Script | Purpose |
|---|---|
| `bench_gpu.py` | FastSAR against torchbp on one simulated spotlight scene (broadside track along y, looking along +x). |
| `crossover.py` | Exact against factorized backprojection by image size, on one device, with the released FastSAR's public API |
| `exact_record.py` | FastSAR's exact backprojection (fastsar.ExactFormer) on the full Panama image, as recorded for the paper: the |
| `exact_regions.py` | FastSAR's exact backprojection (ExactFormer, public API, default cubic interpolation) on the three Panama regions |
| `exact_tpu.py` | FastSAR's exact backprojection (fastsar.ExactFormer) on one Umbra collection, recorded in the format of the study's |
| `fastsar_exact.py` | FastSAR's exact backprojection (fastsar.backproject, cpu backend) on the Panama collection: the three 512 by 512 |
| `geometry_facts.py` | Geometry facts of the released FastSAR for the three Umbra collections, from its public modules (no device): the |
| `grdl_panama.py` | GRDL (github.com/GEOINT/grdl) polar format and fast factorized backprojection on the Umbra Panama CPHD, run as |
| `grdl_score.py` | GRDL polar-format and FFBP images against the reference on the three Panama regions, scored as the NGA pfa_mem image |
| `grdl_spot_score.py` | GRDL spotlight images (polar format, FFBP) of a modes-comparison collection against the float64 reference on the three |
| `grdl_strip.py` | GRDL (github.com/GEOINT/grdl) stripmap algorithms on a Capella stripmap CPHD, run as published (default parameters) |
| `grdl_strip_score.py` | GRDL stripmap images (range-Doppler, FFBP) against the float64 reference on the three regions of the other-modes |
| `isce3_compare.py` | ISCE3's backprojection (isce3.focus.backproject) on the Umbra Panama CPHD, against our float64 exact |
| `modes_common.py` | Shared pieces of the other-modes comparison (stripmap and further spotlight collections): every implementation gets |
| `modes_diag.py` | Diagnostic of the ISCE3 error on one real collection: one small patch at region 1, ISCE3 at two range |
| `modes_isce3_ramp.py` | ISCE3's region images of the modes comparison scored as on Panama: after removal of a fitted linear phase ramp (rad |
| `modes_strip.py` | Stripmap comparison on one Capella collection: FastSAR's mosaic over the whole vendor footprint (timed) and ISCE3's |
| `modes_table.py` | Collect the other-modes records (results/comparison/modes/<tag>/*.json) into one table: per collection and implementation |
| `nga_compare.py` | Score NGA's bpBasic images against our float64 reference (sim: computed here; Panama: the study's reference). |
| `nga_pfa_prep.py` | Inputs for the NGA polar format (pfa_mem) from the Panama CPHD: the phase history with the reference's Taylor |
| `nga_pfa_score2.py` | NGA pfa_mem image against the reference on the three Panama regions, as the vendor image of Appendix F: each |
| `nga_prep.py` | Write inputs for NGA's bpBasic.m (Octave): a simulated scene and three Panama regions, as raw binary files. |
| `pfa_region_scores.py` | FastSAR's polar format (release record, CPU) on the three Panama regions: amplitude and log-amplitude correlation |
| `range_sweep.py` |  |
| `region_errors.py` | Error of each configuration of the cost figure on the three Panama regions (lock, port, ship), per region and |
| `release_record.py` | Records of the released FastSAR (public API only) on one Umbra collection and one device, in the format of the |
| `ritsar_even.py` | RITSAR (range axis fixed) against our float64 backprojection on identical inputs: the Panama phase history with |
| `ritsar_panama.py` | RITSAR's backprojection (CPU, float64 NumPy) on three Panama regions of our pixel grid, against our float64 |
| `tbp_emulate.py` |  |
| `tbp_ffbp_check.py` |  |
| `test_ritsar_sim.py` | RITSAR backprojection (CPU, float64 NumPy) against our float64 reference and our C++ kernels, simulated scene. |
| `torchbp_panama.py` | torchbp on the Umbra Panama collection: exact backprojection of the three 512 by 512 pixel regions against the |
| `window_ab.py` | Does the moving GPU window of range profiles cost speed? A fifth of the 2025 Capella stripmap (small enough that its |
| `nga_run.m`, `nga_pfa_run.m`, `octshim/` | The AFRL/NGA toolbox's `bpBasic` and `pfa_mem` driven in GNU Octave, with a compatibility `cross.m` |

## Records

| File or directory | Produced by | Contents |
|---|---|---|
| `c4d_ritsar.json`, `c4d_hw_ritsar.log` | `ritsar_panama.py` | RITSAR as published on the lock region of the c4d-highmem-16 instance: time and error against the reference |
| `ritsar_panama.json`, `ritsar_even.json` | `ritsar_panama.py`, `ritsar_even.py` | RITSAR errors on the three regions as published and with its two indexing defects worked around |
| `nga_results.json`, `c4d_hw.log` | `nga_prep.py`, `nga_run.m`, `nga_compare.py` | `bpBasic` errors by FFT length and its time on the lock region |
| `c4d_isce3_*.json`, `isce3_*.json`, `isce3gpu_*.json`, `isce3_ramp_removed.json`, `isce3gpu_version.txt` | `isce3_compare.py` | ISCE3 CPU and CUDA backprojection on three 256 by 256 patches: times, errors before and after removal of the fitted linear phase ramp |
| `torchbp_panama.json`, `torchbp_ffbp_regions.json`, `torchbp_checks_cpu.json`, `bench*_*.json` | `torchbp_panama.py`, `tbp_emulate.py`, `tbp_ffbp_check.py`, `bench_gpu.py` | torchbp on the L4: the float32 range failure, the float64 emulation, the memory of its factorized form, and simulated-scene timings against FastSAR |
| `grdl_*.json` (if present), `fastsar_pfa_scores.json`, `nga_pfa_scores.json`, `c4d_nga_pfa.log` | `grdl_panama.py`, `grdl_score.py`, `pfa_region_scores.py`, `nga_pfa_prep.py`, `nga_pfa_score2.py` | Polar-format implementations on the three regions: log-amplitude correlation, coherence and time |
| `c4d_fastsar_exact.*`, `l4_fastsar_exact.*` | `fastsar_exact.py`, `exact_record.py` | FastSAR's exact backprojection on the three regions and the full image (development runs; the release records are in `../v3`) |
| `modes/collections.json` | `modes_common.py` | Core name, collector and mode of the five Capella collections |
| `modes/all/<tag>/` | `modes_strip.py`, `modes_common.py`, `grdl_strip.py`, `grdl_strip_score.py`, `modes_isce3_ramp.py` | One directory per run: `*.json` records and `strip_*.log` logs. Tags: `c<coll>X` FastSAR on the CPU instance, `g<coll>Y` FastSAR on the L4, `<coll>X` FastSAR on the v6e, `<coll>e` and `ss2022ib` ISCE3, `sp2024`, `sp2025`, `sm2021g`, `sm2025` GRDL; other suffixes are earlier or diagnostic runs that `modes_table.py` does not read |
| `fastsar_tests.log`, `gpu.txt` | FastSAR's test suite on the comparison instance; the L4 host's `nvidia-smi` |
