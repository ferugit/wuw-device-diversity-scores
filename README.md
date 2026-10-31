# WuW device-diversity scores

Per-sample output scores of every system evaluated in our study on device-label-agnostic
training techniques for device-robust wake-up word detection, and the script that computes
all reported metrics, 95% confidence intervals and significance tests from them.

The paper is not published yet; its reference will be added here.

## Systems

All systems except TR are SGRU wake-up word models (`--arc sgru`) trained on the same data,
differing in the training technique (see `experiments.tsv` for the exact configurations).
The techniques correspond to the following components of the training code in
[wake-up-word-detection](https://github.com/Telefonica-Scientific-Research/wake-up-word-detection):

| Systems | Technique | Training code |
|---|---|---|
| BL | Baseline | `src/train_and_test.py --arc sgru` |
| DA-1, DA-2 | Device impulse responses (DIR) | `--dir_prob`, with impulse responses as `dir` rows of the partition |
| DA-3, DA-4 | FilterAugment | `--filteraugment_prob`, `--filteraugment_min_db`, `--filteraugment_max_db` |
| DA-6 | FreqMixStyle (label-wise) | `--freqmixstyle_prob`, `--freqmixstyle_alpha` |
| DA-7 – DA-10 | FreqMasking | `--freqmask_prob`, `--freqmask_max_len` |
| DA-11, DA-12 | TimeMasking | `--timemask_prob`, `--timemask_max_len` |

The SSL teacher model (TR, w2v-BERT 2.0), knowledge distillation (KD-1 – KD-5) and the
label-agnostic FreqMixStyle (DA-5) are not included in that training code, and the MixUp
systems (DA-13 and its combinations) were trained with an earlier MixUp implementation than
the one it provides. The scores of all systems are released so that all the reported
statistics can be reproduced.

## Contents

```
experiments.tsv                       system IDs used in the paper (BL, TR, DA-1 … CMB-11)
scores/wuw_device_diversity/<ID>.tsv  one file per system: 589 test samples
scores/parallel_tau/<ID>.tsv          one file per system: 1,404 test samples
evaluate.py                           metrics, bootstrap CIs, paired tests vs BL, threshold optimisation
results/                              output of evaluate.py (as reported in the paper)
results/sensitivity/                  same, excluding the two devices with 9 samples
requirements.txt
```

## Score files

Each row is one test sample.

| Column | Description |
|---|---|
| `sample_id` | Sample identifier |
| `speaker_id` (WuW DD) / `clip_id` (TAU) | Resampling cluster: speaker, or audio clip recorded on all devices |
| `device` | Recording device |
| `is_wuw` | 1 if the sample contains the wake-up word, 0 otherwise |
| `score_wuw`, `score_non_wuw` | Model output probabilities for the two classes |

A sample is predicted as WuW when `score_wuw > score_non_wuw`.

**WuW Device Diversity** recordings are not publicly available. Speaker and sample identifiers
are anonymised (`S01`–`S24`, `dd_0001`–`dd_0589`), and no speaker metadata is included. The
identifiers are assigned in the order of the original ones, so the bootstrap resamples, and
therefore every reported number, are reproduced exactly.

**Parallel TAU** is derived from the public TAU Urban Acoustic Scenes 2020 Mobile data
(Heittola et al., DCASE 2020; <https://doi.org/10.5281/zenodo.3819968>), using the clips
recorded in parallel on devices a, b and c; sample identifiers are kept as in the original
dataset. Only identifiers and model scores are included, not the audio.

## Reproducing the results

The evaluation has its own requirements (Python ≥ 3.12), separate from the training code, so
use a dedicated environment. From this folder:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python evaluate.py                       # writes results/ (about 4 min with 8 workers)
python evaluate.py --output-dir results/sensitivity \
                   --exclude-devices HUAWEI_BXX-W09 samsung_SM-S721B
```

The script's docstring (`python evaluate.py --help`) describes the metrics and the bootstrap
procedure. In short:

- **Metrics:** WuW F1-score and device-balanced accuracy (BA) on WuW Device Diversity; mean
  (WDM) and standard deviation (WDS) of the pairwise Wasserstein distances between per-device
  score distributions on both test sets.
- **Resampling:** one-stage cluster bootstrap, 10,000 resamples, seed 42. Speakers (WuW Device
  Diversity) or clips (Parallel TAU) are drawn with replacement and keep all their samples.
- **Comparisons:** all systems share the same resamples, so each system's difference to BL is
  paired. `significance.csv` gives the difference, its 95% CI and a two-sided p-value;
  p < 0.05 exactly when the CI excludes 0. No correction for multiple comparisons is applied.
- **Threshold optimisation** (BL and DA-2): leave-one-device-out search over thresholds
  0.01–0.99; the final threshold is the median of the per-round picks and is applied to all
  devices. The procedure is re-run on every resample, and the final threshold is compared with
  0.5 within each system (paired).

## Output columns

`ci_results.csv`: for each system and metric `<M>`: `<M>` (point estimate), `<M>_lo`, `<M>_hi`
(95% CI), `<M>_ci_valid` (CI contains the point estimate), `<M>_boot_mean` (mean over
resamples).

`significance.csv`: for each system except BL and each metric: `<M>_diff` (system − BL),
`<M>_lo`, `<M>_hi` (95% CI of the difference), `<M>_p`, `<M>_sig` (p < 0.05).

`threshold_optimization.csv`: for BL and DA-2: `Threshold` (final threshold) and, for `<M>` in
`BA` and `BF1` (mean per-device WuW F1-score): `<M>_default`, `<M>_opt` (at 0.5 and at the
final threshold) with `_lo`/`_hi` CIs; `<M>_diff` (final − default) with `<M>_diff_lo`,
`<M>_diff_hi`, `<M>_p`, `<M>_sig`.

Wasserstein metrics are on the probability scale; the paper reports them in units of 10⁻².

## License

- **Code** (`evaluate.py`): [MIT License](LICENSE).
- **WuW Device Diversity scores** (`scores/wuw_device_diversity/`), `experiments.tsv` and
  `results/`: [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-DATA).
- **Parallel TAU scores** (`scores/parallel_tau/`) and the Parallel TAU metrics in `results/`
  (columns containing `_TAU`): these are results obtained using the TAU Urban Acoustic Scenes
  2020 Mobile dataset and are distributed under its terms, for experimental and
  non-commercial purposes only. The dataset license is available with the dataset at
  <https://doi.org/10.5281/zenodo.3819968> and copied in
  [`scores/parallel_tau/LICENSE`](scores/parallel_tau/LICENSE).

© 2026 Fernando López Gavilánez.
