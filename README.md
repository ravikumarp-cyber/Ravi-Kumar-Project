# Ravi Kumar Project — PGRFA continual-learning experiments

This repository contains the supplied CIFAR-100 continual-learning notebooks, ablation source, and saved experimental results. Original included files are preserved byte for byte. This is the PGRFA project.

## Contents

- `notebooks/notebook9f4545a7fd.ipynb`: original full-method and matched replay-only experiment, including saved notebook outputs.
- `notebooks/notebook3ea6290484.ipynb`: component-ablation orchestration and comparison notebook, including saved outputs.
- `src/pgrfa_ablation.py`: supplied generated ablation training script.
- `results/main/`: full-method and replay-only metrics, configuration, dataset manifest, status and PDF report.
- `results/ablation/`: no-feature, no-prototype and no-logit metrics, configuration, comparison table, status and PDF report.

## Experimental setup and saved results

The supplied manifest describes original CIFAR-100: 50,000 training images, 10,000 test images, 100 classes, 20 tasks with 5 classes each. Evaluation is task-incremental with known task ID. Seeds are 42–46. Raw CIFAR-100 images are not included.

Saved `method_summary.csv` files report mean accuracy (%) of 82.008 for improved, 77.626 for replay-only, 81.09 for no-feature, 79.17 for no-logit and 81.69 for no-prototype. These are supplied experimental records; training was not rerun or independently reproduced during repository preparation.

## Running the supplied code

The code targets a Kaggle GPU environment with internet enabled and uses `/kaggle/working` and `/kaggle/input`. It explicitly requires CUDA. Install the dependencies in `requirements.txt` with a CUDA-compatible PyTorch/torchvision pair appropriate for the environment. Dependency versions were not supplied and are intentionally unpinned.

Open the main notebook in a Kaggle GPU session to run the full method and replay baseline. The ablation script can be run with `python src/pgrfa_ablation.py` in the expected environment. Read configuration constants before running: these are substantial multi-seed training jobs.

The ablation notebook attempts to retrieve its original source using an existing Kaggle kernel reference, then a Google Drive file reference. Those references are preserved in the original notebook; their current availability is not verified. The notebook also accepts a notebook already placed in `/kaggle/working/PGRFA_ABLATION_SOURCE`. The supplied main notebook can be placed there before running to avoid external source retrieval. The saved Python ablation script is already included separately.

## Preservation and exclusions

ZIP copies are omitted because every archived entry matches the corresponding extracted file. The backup's source notebook has identical cell source to the main notebook but differs in saved outputs; the downloaded main notebook is retained. Private Kaggle deployment metadata (`kernel-metadata.json`) is omitted. Fifteen resumable `.pt` checkpoints (about 758 MB total) are excluded from this source-and-results proposal and remain in the original Downloads backup. No checkpoints are required to inspect the saved metrics; resuming a saved run requires the corresponding original checkpoint. No dataset, code or results have been invented.

The original downloads remain unchanged. No license was supplied; none is added.
