# Held-out round 2: selection and pass criteria, fixed before the round-1 fixes

Written 2026-10-09, after held-out round 1 found bugs and before any fix for them. The collections of
`heldout_round2.txt` were drawn by `oss/heldout_round2_select.py` (seed 20261009) from the open-data bucket listings
in this directory, excluding every collection used in development or in round 1. They are not downloaded or looked
at until the fixes of round 1 are released as a candidate; then they are run once, as drawn, with `oss/heldout.py`
(the same harness as round 1, with only the round-1 harness fixes listed below), and every result is reported.

Collections: Capella spotlight, sliding spotlight and stripmap, two each (satellites C05, C06, C13, none used
before); Umbra, four sites and satellites not used before (UMBRA-04, -05, -06, -08); ICEYE, the two remaining
collections of satellites not used before (X47 dwell, X56 spot).

Pass criteria, per collection:

1. `io.read_cphd` reads the file, or refuses it with a ValueError that names an unsupported feature.
2. Factorized against exact backprojection on the same grid (1024 by 1024 pixels, or the largest the collection's
   range ambiguity allows), all pulses: difference at most -50 dB.
3. The vendor's image (SICD of the same collection): amplitude correlation of FastSAR's exact backprojection on the
   vendor's pixels at least 0.5 in the center window, and at least 0.3 above the correlation with the phase sign
   reversed.
4. `form_cphd(cphd)` with defaults finishes with a finite image, or raises MemoryError or ValueError with a message
   that names what to change; never a kill by the operating system or another exception.

A failure is reported as a failure. A fix found from a round-2 failure is checked again on a third set chosen the
same way, not on round 2.

Harness fixes allowed before round 2 (found in round 1, not library changes): read Capella's SICD (.ntf) instead of
its SLC GeoTIFF; per-collection logs kept when a run is killed.

Amendment, 2026-10-09, before any round-2 collection was downloaded: 0.1.1 removes the troposphere delay by default
when the CPHD gives one (Capella's SICD images remove it, Umbra's keep it). The harness reads with
`troposphere=False`, which is what round 1's default did, so criterion 3 keeps round 1's meaning; the harness also
reports, as a diagnostic and not a criterion, the correlation peak over shifts of up to 32 pixels and its position.

Amendment, 2026-10-10, before any round-2 collection was downloaded: the harness's vendor check copied the phase
history twice (`S[sel]`, then its conjugate), which killed the 320,360-pulse Capella C11 spotlight of round 1 at
128 GB after the library fixes; it now passes a slice and conjugates in place. No criterion changes.
