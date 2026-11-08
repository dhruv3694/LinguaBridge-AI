# Dataset Documentation

## `data/translation_benchmark.csv`
- Notebook Module 4 benchmark (20 sentences in notebook construction; current file may contain extended rows).
- Expected columns for evaluation: `id`, `english`, and language reference columns (e.g., `reference_hindi`, `reference_marathi`).
- **BLEU/chrF:** computed only when references are valid/non-empty. Do not fabricate references.

## `data/bias_test_dataset.csv`
- Notebook Module 7 profession/gender dataset.
- Expected columns: `id`, `gender_condition` (`neutral`/`masculine`/`feminine`), `profession`, `english`.
- Used for heuristic grammatical marker screening in Hindi/Marathi translations; requires human review.

## Generated outputs
- `outputs/evaluations/`: comparison and metric CSVs.
- `outputs/bias/`: results, summaries, neutral/pair analysis, human review worksheet, chart.
- `outputs/audio/`: generated WAV files (gitignored by default).
