# Visual Anomaly Inspection

A college project comparing PatchCore, PaDiM and ViT-Patch with an adaptive hybrid. The Streamlit app displays experiment analytics, anomaly heatmaps, image and heatmap weights, and model contributions.

The evaluated categories are **bottle, carpet, leather, screw and transistor**, with runs using seeds **0, 1 and 2**. Models are fitted separately for each category.

## Setup

Use **Python 3.10.20**. Open a terminal in the `visual-inspection` project folder and run:

```powershell
conda create -n visual-inspection python=3.10.20 pip -y
conda activate visual-inspection
python -m pip install -r requirements.txt
```

If using an existing working environment, activate it instead of creating another one. For NVIDIA GPU use, install a matching CUDA-enabled PyTorch/torchvision build using the [official PyTorch instructions](https://pytorch.org/get-started/locally/). Pretrained backbone weights may download on first use.

## Fitted models and results

Place these folders next to `app.py`, preserving their original names and internal paths:

| Folder | Contents | Needed for |
| --- | --- | --- |
| `results_lightweight/` | Fitted PatchCore, PaDiM and ViT-Patch checkpoints and saved data splits | Inspecting uploaded images and rerunning comparisons |
| `results_hybrid_active/` | Hybrid configurations, comparison metrics and hybrid heatmaps | Analytics and hybrid inspection |
| `results_hybrid/` | Cached baseline heatmaps and predictions | Saved test-image inspection |

If model/result files are distributed separately, extract them into these locations before running the app. The hybrid configurations reference baseline checkpoints using relative paths; keep the folder layout intact.

With the fitted files supplied, **users do not need to refit the models**. The dataset is needed for saved test examples and experiment evaluation; it is not needed to inspect a user-uploaded image with the supplied models.

The project `.gitignore` excludes the dataset, environments and caches. It does **not** exclude these result folders, a `checkpoints/` folder, or `.pt` checkpoint files. Large artifacts can be distributed separately through GitHub Releases or Git LFS when appropriate.

## Download the dataset

This project uses the **original MVTec AD dataset (CVPR 2019)**. The download below contains all **15 categories**, including the five used by this project, with the original training/test folders and ground-truth masks. It does not download fitted model checkpoints.

- [Official MVTec AD dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [Download-link update from the Anomalib maintainers](https://github.com/open-edge-platform/anomalib/discussions/3246)

The archive is roughly 5 GB; allow additional disk space for extraction. If you already have the dataset in `datasets/MVTecAD/`, skip this step.

Run these commands in **Windows PowerShell** from the project folder:

```powershell
New-Item -ItemType Directory -Force -Path .\datasets\MVTecAD | Out-Null

curl.exe --location --fail --retry 3 --output .\datasets\mvtec_anomaly_detection.tar.xz "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"
if ($LASTEXITCODE -ne 0) { throw "Dataset download failed. Use the official dataset page linked above." }

tar.exe -xf .\datasets\mvtec_anomaly_detection.tar.xz -C .\datasets\MVTecAD
if ($LASTEXITCODE -ne 0) { throw "Dataset extraction failed. Check the archive and available disk space." }
```

If the download link changes, download `mvtec_anomaly_detection.tar.xz` from the official dataset page, place it in `datasets/`, and run the `tar.exe` extraction command above. No `.ps1` file is required.

After extraction, paths should include:

```text
datasets/MVTecAD/bottle/train/good/
datasets/MVTecAD/bottle/test/
datasets/MVTecAD/bottle/ground_truth/
datasets/MVTecAD/screw/train/good/
datasets/MVTecAD/transistor/train/good/
```

The other categories follow the same layout. Point the app's dataset field to `datasets/MVTecAD`.

## Run the app

```powershell
conda activate visual-inspection
python -m streamlit run app.py
```

The default results folder is `results_hybrid_active`. Select a category and inspection seed to use that saved run. Upload an image of the selected product category, or choose a saved dataset example.

## Optional: rerun the comparison

This requires the dataset **and** the existing fitted baseline checkpoints and `split.json` files. The current `experiments.py` reuses those baselines; it does not train them from scratch.

```powershell
python experiments.py --root datasets/MVTecAD --categories bottle carpet leather screw transistor --seeds 0 1 2
```

## Dataset attribution

MVTec AD: Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger, *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019. Dataset license: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), as stated on the official dataset page.
