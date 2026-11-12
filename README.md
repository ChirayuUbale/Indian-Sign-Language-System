# Indian Sign Language Image Dataset and Preprocessing

This repository contains a small image collection and scripts for collecting sign images, cropping them around detected hands, and comparing image preprocessing methods. The current work is an image preparation and comparison pipeline; it does not yet train or evaluate a sign recognition model.

## Dataset

Images are organized by signer and sign label:

```text
data/<signer>/<label>/<image>
data_cropped/<signer>/<label>/<image>
```

The collection script captures the 26 letters and 10 digits. At the time of the latest comparison, `data_cropped/` contained 1,372 images across 36 labels for two signers:

| Signer | Labels | Cropped images |
| --- | ---: | ---: |
| Chirayu | 36 | 708 |
| S04 | 36 | 664 |

Image counts are not balanced across signers or necessarily across labels. See [the project journey](docs/PROJECT_JOURNEY.md) for collection and cropping history. Raw and cropped captures may contain identifiable people; review dataset contents before sharing or adding more images to a commit.

## Setup

The pinned dependencies were used with Python 3.11 on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use the same virtual environment to run all scripts. The protobuf pin keeps the installed MediaPipe version on a compatible protobuf major version.

## Scripts

Run commands from the repository root.

### Collect camera images

```powershell
python src/collection.py
```

The script asks for a signer name and saves webcam images in `data/<signer>/<label>/`. It targets 20 images per label, with a 15-second setup interval and 2 seconds between captures. Press `s` to capture sooner or `q`/Escape to stop.

### Crop images around detected hands

```powershell
python src/crop_hands.py --input data --output data_cropped
```

MediaPipe detects up to two hands. The script combines their padded landmark boxes, saves PNG crops while preserving signer/label folders, and writes `data_cropped/crop_failures.csv`. The input tree is left untouched. Optional flags include `--padding`, `--size` (square resize; `0` preserves crop dimensions), and `--min-confidence`.

### Run separate preprocessing experiments

**Proposed research question:** On our self-collected 36-class ISL dataset, how do 10% and 20% symmetric label-noise rates in the training set affect sign-recognition performance compared with clean-label training (0%)?

Here, noise means deliberately incorrect training labels, not visual noise added to image pixels. Select 10% or 20% of training examples and replace each selected label with a different class label. Keep validation and test labels clean. Hold the images, model setup, and preprocessing constant across all three noise conditions so the label-noise rate is the only changing factor. This question follows the course's noisy-label project category; review related work before claiming a literature gap.

Keep two experiment tracks distinct. For the replication baseline, reproduce the paper's preprocessing faithfully. For the noisy-label experiment, use **plain grayscale as the fixed preprocessing** for all 0%, 10%, and 20% label-noise conditions. This keeps the noise rate as the main changing factor. The other preprocessing comparisons remain exploratory; they are not part of the noisy-label comparison.

```powershell
python experiment/run_experiments.py
```

The runner sorts image paths and samples 10 total uniformly from the full `data_cropped/` tree with seed 42. Sampling is not balanced by signer or label. It applies each method to the same sample and saves one input/output sheet per method plus `sampled_images.csv` in `experiment/results/`. Pass `--count 20` or `--seed 7` to change the sample size or select a different repeatable sample. The exact latest sample and per-method results are recorded in [the experiment notes](docs/PREPROCESSING_COMPARISON.md).

The recorded pilot used 10 images (8 Chirayu, 2 S04); all methods reported 0 fallbacks. Visual review still found HSV masks that missed much of the hand and GrabCut masks that included sleeves or background. The fallback count is not a visual quality score. The five pilot method sheets were deleted; the 10-image manifest remains, and the approved 10-image rating data and chart are archived separately. The expanded 30-image method sheets are linked below.

### Latest visual ratings: 30 images

A preliminary visual rubric was applied to each method output for the signer-balanced 30-image sample. Scores range from 1 (poor) to 5 (strong) for hand visibility, background suppression, and detail preservation. The chart shows the mean rating and standard deviation across 30 scores per method:

![Grouped bar chart of preliminary visual ratings on 30 images, pending Chirayu review](experiment/results/visual_ratings.png)

| Method | Hand visibility | Background suppression | Detail preservation |
| --- | ---: | ---: | ---: |
| `baseline_hsv` | 3.00 | 3.90 | 2.50 |
| `baseline_canny` | 2.40 | 5.00 | 2.10 |
| `grabcut` | 4.40 | 3.03 | 3.97 |
| `clahe` | 4.50 | 1.00 | 4.47 |
| `plain` | 4.50 | 1.00 | 4.47 |

These are initial Codex visual scores and are pending Chirayu's review. They are subjective judgments, not recognition accuracy or automated image-quality measurements. The per-image scores are in [ratings.csv](experiment/ratings.csv). The previously approved 10-image scores and chart are archived as [ratings_10_chirayu_approved.csv](experiment/ratings_10_chirayu_approved.csv) and [visual_ratings_10_chirayu_approved.png](experiment/results/visual_ratings_10_chirayu_approved.png). The rubric and limitations are documented in [the experiment report](docs/PREPROCESSING_COMPARISON.md).

Regenerate the chart with `python experiment/plot_ratings.py`.

The chart above now uses the same 30-image sample as the expanded visual pages below. The earlier 10-image rating data and chart are archived separately.

### Expanded visual sample: 30 images

To inspect more examples while keeping the sheets readable, sample 15 images per signer and split each method's outputs into pages of 10:

```powershell
python experiment/run_experiments.py --count 30 --balance-by-signer --rows-per-sheet 10 --output experiment/results/expanded_30
```

This run sampled 30 images (15 per signer) and reported zero fallbacks for all five methods. The expanded sample is listed in [sampled_images.csv](experiment/results/expanded_30/sampled_images.csv).

Each collapsible section below embeds all three 10-image pages for one method.

<details>
<summary>Baseline HSV: 30 images</summary>

![Baseline HSV, page 1 of 3](experiment/results/expanded_30/baseline_hsv_page_01.png)
![Baseline HSV, page 2 of 3](experiment/results/expanded_30/baseline_hsv_page_02.png)
![Baseline HSV, page 3 of 3](experiment/results/expanded_30/baseline_hsv_page_03.png)
</details>

<details>
<summary>Baseline Canny: 30 images</summary>

![Baseline Canny, page 1 of 3](experiment/results/expanded_30/baseline_canny_page_01.png)
![Baseline Canny, page 2 of 3](experiment/results/expanded_30/baseline_canny_page_02.png)
![Baseline Canny, page 3 of 3](experiment/results/expanded_30/baseline_canny_page_03.png)
</details>

<details>
<summary>GrabCut: 30 images</summary>

![GrabCut, page 1 of 3](experiment/results/expanded_30/grabcut_page_01.png)
![GrabCut, page 2 of 3](experiment/results/expanded_30/grabcut_page_02.png)
![GrabCut, page 3 of 3](experiment/results/expanded_30/grabcut_page_03.png)
</details>

<details>
<summary>CLAHE: 30 images</summary>

![CLAHE, page 1 of 3](experiment/results/expanded_30/clahe_page_01.png)
![CLAHE, page 2 of 3](experiment/results/expanded_30/clahe_page_02.png)
![CLAHE, page 3 of 3](experiment/results/expanded_30/clahe_page_03.png)
</details>

<details>
<summary>Plain grayscale: 30 images</summary>

![Plain grayscale, page 1 of 3](experiment/results/expanded_30/plain_page_01.png)
![Plain grayscale, page 2 of 3](experiment/results/expanded_30/plain_page_02.png)
![Plain grayscale, page 3 of 3](experiment/results/expanded_30/plain_page_03.png)
</details>

The individual methods are in `experiment/methods/`. The older `src/CmparePreprocess.py` script is retained as a combined comparison option; it can recreate `review/compare.png` if run. The historical result of its eight-image run is described in [the preprocessing comparison notes](docs/PREPROCESSING_COMPARISON.md).

## Repository map

```text
src/collection.py          Webcam image collection
src/crop_hands.py          MediaPipe hand crop generation
experiment/methods/        One implementation file per preprocessing method
experiment/run_experiments.py  Sample images and save one result sheet per method
experiment/plot_ratings.py  Plot the rubric ratings from experiment/ratings.csv
experiment/ratings_10_chirayu_approved.csv  Archived 10-image ratings approved by Chirayu
src/CmparePreprocess.py    Legacy combined comparison script
docs/                      Project history and comparison notes
experiment/results/        Current 30-image rating chart, 10-image manifest, and archived chart
experiment/results/expanded_30/  Expanded 30-image sheets and sample manifest
```

See [AGENTS.md](AGENTS.md) for repository-specific guidance for contributors and coding agents.
