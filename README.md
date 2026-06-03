# BIKE (Local)

This repo contains my local modifications and experiments based on the BIKE project.

## Environment

- Linux
- Python 3.8+
- PyTorch 1.8+ with CUDA
- NVIDIA GPU with CUDA drivers

## Start Command

```sh
sh /home/neimedia/gmk/BIKE/scripts/run_train_23.sh /home/neimedia/gmk/BIKE/configs/hmdb51/hmdb_CLIP_fix_L_0_11_fusion.yaml
```

## Project Structure

- `.git/`: Git metadata.
- `Coviar/`: COViar-related code and assets (video representation utilities).
- `X_CLIP/`: X-CLIP submodule code and dependencies.
- `attributes/`: Attribute vocabularies and auxiliary attribute resources.
- `clip/`: CLIP model definitions, tokenizers, and fusion variants.
- `configs/`: Dataset-specific configs (e.g., hmdb51/ucf101/k400) for training and evaluation.
- `datasets/`: Dataset loaders and video/LMDB pipelines (`video.py`, `video_lmdb.py`, `video_attr.py`).
- `docs/`: Documentation assets (figures, diagrams).
- `exps/`: Experiment outputs, checkpoints, logs, and analysis artifacts.
- `exps_few/`: Few-shot experiment outputs and checkpoints.
- `lists/`: Dataset list files and label/annotation indexes.
- `modules/`: Core model modules (fusion, temporal, prompt, and compression blocks).
- `outputs/`: Default output directory for runs.
- `outputs-hmdb51/`: HMDB51-specific outputs.
- `outputs——dataname/`: Outputs grouped by dataset name.
- `qwen/`: Qwen-related scripts and resources.
- `retrieval_results/`: Retrieval evaluation result files.
- `scripts/`: Run scripts, e.g., `run_train_23.sh`, `run_test.sh`, `run_test_zeroshot.sh`.
- `utils/`: Shared utilities (logging, metrics, IO helpers).
- `video_sentence_fusion/`: Video-text fusion modules and helpers.
- `visualization_comparison/`: Visualization assets and comparison reports.

## Key Files

- `train.py` / `train3.py`: Main training entry points.
- `train_comp_CoAPT.py`: CoAPT training pipeline.
- `train_comp_CoAPT_fusion.py`: CoAPT training with fusion variants.
- `train_comp_CoAPT_lmdb.py`: CoAPT training with LMDB-backed data loading.
- `train_attributes.py`: Attribute branch training.
- `train_attribute_qwen.py`: Attribute training with Qwen inputs.
- `train_qwen_description.py`: Qwen description generation/training.
- `test.py`: Standard evaluation entry point.
- `test_zeroshot.py`: Zero-shot evaluation.
- `test_charades.py`: Charades evaluation.
- `test_comp_CoAPT.py`: CoAPT evaluation.
- `retrieval_eval.py`: Retrieval evaluation.
- `attribute_search.py`: Attribute search/selection utilities.
- `grad_cam.py`: Grad-CAM visualization.
- `dataloader_test.py`: Data loader sanity checks.
- `BIKE_inference.ipynb`: Notebook for inference demos.

## Notes

- Training configs live under `configs/`.
- Experiments and outputs are saved under `exps/`.
