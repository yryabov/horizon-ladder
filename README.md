# horizon-ladder

Scripts and verdict files for

> Y. Ryabov, *A ladder of conclusions on the history of dark energy:
> an inception at z ≃ 4 preferred by Planck, DESI DR2, and Pantheon+* (2026).

Every number in the paper traces to one script here and one verdict file
(JSON) under `results/`. The full map from paper objects (tables, figures,
sections) to scripts and verdicts is in [`MANIFEST.md`](MANIFEST.md).

## Layout

| Path | Contents |
|---|---|
| `battery/` | The three fitting batteries: every model (Avrami, Front/clock, Thermal GC, GC-p, CPL, ΛCDM) on CMB+BAO, +Lyman-α update, +Pantheon+ |
| `controls/` | External controls: eternal screen (HDE), no-dark-energy |
| `uncertainty/` | Profile likelihoods (z_c, p), MCMC validation, Fisher check |
| `aux/` | Solver integrity checks, cross-pipeline PR4 refits, kinetic arbitration, derived-quantity evals |
| `nulltests/` | Sky and supernova anisotropy null tests (seams, strips) |
| `figures/` | Scripts for every figure of the paper |
| `patchers/` | Data-convention patches (Lyman-α row, SN covariance) |
| `results/` | Verdict files (JSON) behind every quoted number; `results/figures/` holds the rendered figures exactly as compiled |
| `nucleation_camb.py` | Shared background solver + CAMB interface |

## Reproducing

Scripts are single-file and self-validating: each battery script evaluates
the official Planck 2018 best-fit anchor before fitting and refuses to run
unless the likelihood reproduces it; verdict JSONs carry the full parameter
vectors and per-block χ². Pure-background checks (`aux/attractor_check_v1.py`,
`aux/kinetics_arbitration_v1.py`, `aux/gc_ratio_eval_v1.py`) need only
numpy/scipy and run in about a minute each.

External data (not redistributed here):
- Planck 2018 `plik_lite` bandpowers and covariance (PLA);
- DESI DR2 BAO distance measurements (embedded as published numbers; the
  2026 Lyman-α full-shape row in both conventions);
- Pantheon+ light curves and STAT+SYS covariance (Pantheon+ release);
- PR4/NPIPE CamSpec likelihood via `cobaya` for the cross-pipeline refits;
- for `figures/fig_cmb_v1.py`, a clone of `planck-lite-py` (see MANIFEST).

Paths to these inputs are set in the CONFIG block at the top of each script.

## Requirements

See `requirements.txt` (numpy, scipy, matplotlib; camb and cobaya for the
likelihood-level scripts; emcee for the MCMC run).

## License

MIT (see `LICENSE`).
