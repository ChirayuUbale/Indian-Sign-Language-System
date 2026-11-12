# Repository guidance

## Project scope

- This repository currently covers image collection, hand cropping, and visual comparison of preprocessing methods. Do not describe it as a trained or evaluated recognition system unless that work is added.
- The current proposal direction is noisy training labels (0%, 10%, 20%), not noise added to image pixels. Keep validation and test labels clean, and hold preprocessing fixed across noise conditions.
- Keep two tracks distinct: reproduce the reference paper's preprocessing for the replication baseline, and use fixed `plain` grayscale preprocessing across 0%, 10%, and 20% label-noise conditions. Keep other preprocessing comparisons exploratory and outside the noisy-label comparison.
- Read the README and relevant script before changing workflow or data assumptions.
- Run commands from the repository root unless the script explicitly requires another working directory.

## Python environment

- Use Python 3.11 and install the pinned packages from `requirements.txt` in `.venv`.
- Use the project virtual environment for MediaPipe and OpenCV commands. Global Python installations may have incompatible protobuf packages.
- Keep direct runtime dependencies in `requirements.txt`; pin versions when compatibility depends on them.

## Dataset handling

- Preserve the `data/<signer>/<label>/` and `data_cropped/<signer>/<label>/` layout.
- Do not delete, rename, overwrite, or bulk-stage image datasets unless the task explicitly calls for it. Cropping writes to `data_cropped/` and can overwrite matching output paths.
- Inspect `git status` before staging. The repository has ignored data paths and previously committed image assets, so `.gitignore` alone does not tell you whether an existing file is tracked.
- Treat captures as potentially identifiable personal data. Avoid copying them into unrelated documentation or external services.

## Preprocessing experiments

- Each method lives in its own module under `experiment/methods/`. Functions accept OpenCV BGR images and return `(image, ok)`. Keep the 100 by 100 grayscale contract unless the task changes the interface.
- The latest visual run uses `python experiment/run_experiments.py --count 30 --balance-by-signer --rows-per-sheet 10 --output experiment/results/expanded_30`: 15 images per signer, three pages per method, no label stratification. The default command still samples 10 unbalanced images.
- A zero fallback count is not a recognition accuracy result. State the sample size, method names, observed failures, and limitations; distinguish visual observations from measured model performance.
- The 30-image visual rubric scores in `experiment/ratings.csv` are preliminary Codex visual ratings pending Chirayu review. They are subjective judgments, not ground truth or accuracy results; preserve their pending status in charts and documentation. The previously approved 10-image ratings are archived in `experiment/ratings_10_chirayu_approved.csv`.
- Regenerate the current 30-image plot with `python experiment/plot_ratings.py`; it writes `experiment/results/visual_ratings.png`.
- `src/CmparePreprocess.py` is a legacy combined comparison script. It is intentionally retained and can recreate `review/compare.png` if run.
- Update `docs/PREPROCESSING_COMPARISON.md` when recording new experiment runs.

## Documentation

- Keep setup instructions consistent with `requirements.txt` and actual script arguments.
- Record verified dataset counts and run outputs with a date or clear snapshot context. Include sample size, seed, signer/class distribution when known, per-method fallback counts, and visual failures. Do not claim a count is permanent when the dataset can change.
