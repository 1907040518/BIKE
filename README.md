# BIKE / TextComp Local

This repository contains local modifications and experiments based on the BIKE project, mainly for **compressed video action recognition** with **text-guided semantic fusion**.

The current codebase supports experiments on compressed-domain video representations, including:

- I-frames
- Motion Vectors
- Residuals
- Text / attribute-guided fusion

The main experimental framework is related to **TextComp: Text-Guided Learning for Compressed Video Action Recognition**.

---

## 1. Overview

Compressed video action recognition operates directly on codec-domain representations such as **I-frames**, **motion vectors**, and **residuals**, avoiding full video decoding and reducing preprocessing cost.

This repository implements and experiments with compressed video recognition pipelines based on CLIP-style visual/textual encoders and fusion modules. In particular, the TextComp-style framework introduces textual semantic attributes generated from action labels and uses them to guide modality fusion.

Main ideas:

- Use compressed video signals directly:
  - I-frame
  - Motion Vector
  - Residual
- Use CLIP visual/text encoders for representation learning.
- Use action-related semantic attributes as textual guidance.
- Fuse compressed modalities dynamically with text-guided fusion modules.
- Support fully-supervised, zero-shot, and few-shot evaluation settings.

---

## 2. Environment

### 2.1 System

The code was tested under the following environment:

- OS: Linux
- Python: 3.8+
- GPU: NVIDIA GPU with CUDA drivers
- CUDA-enabled PyTorch
- Video codec: MPEG-4 compressed videos
- Main compressed-video dependency: `coviar`

### 2.2 Important Notes about MPEG-4 / COViar

This project relies on compressed-domain video representations. Therefore, videos should be encoded in a format that allows extracting:

- I-frames
- Motion vectors
- Residuals

Following common compressed video action recognition settings, videos are expected to be **MPEG-4 encoded**, typically with GOP structures containing one I-frame followed by multiple P-frames.

If the dataset videos are not in the expected MPEG-4 format, they may need to be re-encoded before training/testing.

Example using `ffmpeg`:

```sh
ffmpeg -i input.mp4 \
  -vcodec mpeg4 \
  -q:v 3 \
  -g 12 \
  output_mpeg4.mp4
```

Typical setting used in the paper:

- MPEG-4 encoded videos
- Average of 11 P-frames per I-frame
- Video resolution resized to `340 x 256`

### 2.3 Python Packages

The following packages were installed in the local environment:

```txt
Package            Version
------------------ ----------------
annotated-doc      0.0.4
anyio              4.13.0
audioread          3.1.0
brotlicffi         1.2.0.0
certifi            2026.1.4
cffi               2.0.0
charset-normalizer 3.4.4
click              8.3.1
contourpy          1.3.3
coviar             0.1
cycler             0.12.1
decorator          5.3.0
decord             0.6.0
dotmap             1.3.30
filelock           3.20.3
fonttools          4.61.1
fsspec             2026.2.0
ftfy               6.3.1
gmpy2              2.2.2
h11                0.16.0
hf-xet             1.5.0
httpcore           1.0.9
httpx              0.28.1
huggingface_hub    1.14.0
idna               3.11
Jinja2             3.1.6
joblib             1.5.3
jsonpatch          1.33
jsonpointer        3.0.0
kiwisolver         1.4.9
lazy-loader        0.5
librosa            0.11.0
llvmlite           0.47.0
lmdb               1.7.5
markdown-it-py     4.2.0
MarkupSafe         3.0.2
matplotlib         3.10.8
mdurl              0.1.2
mkl_fft            1.3.11
mkl_random         1.2.8
mkl-service        2.4.0
mpmath             1.3.0
msgpack            1.1.2
networkx           3.6.1
nltk               3.9.3
numba              0.65.1
numpy              2.0.1
opencv-python      4.13.0.92
packaging          25.0
pandas             3.0.1
pillow             12.1.0
pip                26.0.1
platformdirs       4.9.6
pooch              1.9.0
psutil             7.2.2
pycparser          2.23
Pygments           2.20.0
pyparsing          3.3.2
PySocks            1.7.1
python-dateutil    2.9.0.post0
PyYAML             6.0.3
regex              2026.2.19
requests           2.32.5
rich               15.0.0
safetensors        0.7.0
scikit-learn       1.8.0
scipy              1.17.1
seaborn            0.13.2
setuptools         69.5.1
shellingham        1.5.4
six                1.17.0
soundfile          0.13.1
soxr               1.1.0
sympy              1.13.1
termcolor          3.3.0
thop               0.1.1-2209072238
threadpoolctl      3.6.0
timm               1.0.26
torch              2.5.1
torchaudio         2.5.1
torchinfo          1.8.0
torchnet           0.0.4
torchsummary       1.5.1
torchvision        0.20.1
tornado            6.5.4
tqdm               4.67.3
triton             3.1.0
typer              0.25.1
typing_extensions  4.15.0
urllib3            2.6.3
visdom             0.2.4
wcwidth            0.6.0
websocket-client   1.9.0
wheel              0.46.3
```

### 2.4 Minimal Recommended Dependencies

If creating a new environment, the following packages are important:

```sh
pip install torch torchvision torchaudio
pip install numpy pandas scipy scikit-learn
pip install opencv-python decord lmdb
pip install ftfy regex tqdm
pip install timm thop
pip install matplotlib seaborn
pip install PyYAML dotmap
pip install visdom
```

`coviar` should be installed separately according to the local COViar build instructions.

---

## 3. Quick Start

### 3.1 Training

Example training command:

```sh
sh /home/neimedia/gmk/BIKE/scripts/run_train_23.sh \
  /home/neimedia/gmk/BIKE/configs/hmdb51/hmdb_CLIP_fix_L_0_11_fusion.yaml
```

Or from the repository root:

```sh
sh scripts/run_train_23.sh configs/hmdb51/hmdb_CLIP_fix_L_0_11_fusion.yaml
```

### 3.2 SSv2 semantic-attribute ablation (Correct / Dummy / Shuffled + IADF)

This controlled ablation tests whether SSv2 gains come from correct class-to-attribute semantics rather than merely longer prompts or arbitrary attribute text. All three configurations retain the same IADF settings, action-prompt template, training schedule, data lists, and seed index. Only `network.action_prompt.dataset_key` changes.

The loader in [`train_comp_CoAPT.py`](train_comp_CoAPT.py) resolves an attribute file as `vocab_root / f"{dataset_key.upper()}_{seed_index}.json"`; therefore the runtime aliases use uppercase suffixes. The original source is [`SOMETHING_SOMETHING_V2_1.json`](attributes/llama3.1:8b/SOMETHING_SOMETHING_V2_1.json), a 174-key JSON object mapping each SSv2 class name to one whitespace-separated attribute string. At runtime the first 16 tokens are used.

| Condition | Config | `dataset_key` | Resolved runtime attribute file |
| --- | --- | --- | --- |
| Correct Attributes + IADF | [`sthv2_pre_fix_B16_subset10k_val4k_fusion_on_correct_rerun.yaml`](configs/sthv2/sthv2_pre_fix_B16_subset10k_val4k_fusion_on_correct_rerun.yaml) | `SOMETHING_SOMETHING_V2` | [`SOMETHING_SOMETHING_V2_1.json`](attributes/llama3.1:8b/SOMETHING_SOMETHING_V2_1.json) |
| Dummy Attributes + IADF | [`sthv2_pre_fix_B16_subset10k_val4k_fusion_on_dummy.yaml`](configs/sthv2/sthv2_pre_fix_B16_subset10k_val4k_fusion_on_dummy.yaml) | `SOMETHING_SOMETHING_V2_dummy` | [`SOMETHING_SOMETHING_V2_DUMMY_1.json`](attributes/llama3.1:8b/SOMETHING_SOMETHING_V2_DUMMY_1.json) |
| Shuffled Attributes + IADF | [`sthv2_pre_fix_B16_subset10k_val4k_fusion_on_shuffled_seed0.yaml`](configs/sthv2/sthv2_pre_fix_B16_subset10k_val4k_fusion_on_shuffled_seed0.yaml) | `SOMETHING_SOMETHING_V2_shuffled_seed0` | [`SOMETHING_SOMETHING_V2_SHUFFLED_SEED0_1.json`](attributes/llama3.1:8b/SOMETHING_SOMETHING_V2_SHUFFLED_SEED0_1.json) |

Generated source files are [`SOMETHING_SOMETHING_V2_dummy.json`](attributes/llama3.1:8b/SOMETHING_SOMETHING_V2_dummy.json) and [`SOMETHING_SOMETHING_V2_shuffled_seed0.json`](attributes/llama3.1:8b/SOMETHING_SOMETHING_V2_shuffled_seed0.json), with lowercase `_1` copies retained for traceability. The Dummy condition assigns the same 16 generic tokens to every class. The Shuffled condition deterministically permutes whole class attribute strings with seed 0 and verifies zero unchanged class-to-attribute assignments, preserving the original attribute-length distribution.

Regenerate either controlled vocabulary without changing the source file:

```sh
python tools/make_ssv2_attribute_ablation.py \
  --input attributes/llama3.1:8b/SOMETHING_SOMETHING_V2_1.json \
  --output attributes/llama3.1:8b/SOMETHING_SOMETHING_V2_dummy.json \
  --mode dummy --seed 0 --num_attributes 16

python tools/make_ssv2_attribute_ablation.py \
  --input attributes/llama3.1:8b/SOMETHING_SOMETHING_V2_1.json \
  --output attributes/llama3.1:8b/SOMETHING_SOMETHING_V2_shuffled_seed0.json \
  --mode shuffled --seed 0 --num_attributes 16
```

Run the three experiments from the repository root. Use the same visible GPUs and training precision for all three runs; only config and log tag vary.

```sh
CUDA_VISIBLE_DEVICES=1,5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n mpeg4 torchrun --nproc_per_node=2 --master_port=12691 \
  train_comp_CoAPT.py \
  --config configs/sthv2/sthv2_pre_fix_B16_subset10k_val4k_fusion_on_correct_rerun.yaml \
  --log_time ssv2_attr_correct_$(date +%Y%m%d_%H%M%S) --precision amp

CUDA_VISIBLE_DEVICES=1,5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n mpeg4 torchrun --nproc_per_node=2 --master_port=12692 \
  train_comp_CoAPT.py \
  --config configs/sthv2/sthv2_pre_fix_B16_subset10k_val4k_fusion_on_dummy.yaml \
  --log_time ssv2_attr_dummy_$(date +%Y%m%d_%H%M%S) --precision amp

CUDA_VISIBLE_DEVICES=1,5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n mpeg4 torchrun --nproc_per_node=2 --master_port=12693 \
  train_comp_CoAPT.py \
  --config configs/sthv2/sthv2_pre_fix_B16_subset10k_val4k_fusion_on_shuffled_seed0.yaml \
  --log_time ssv2_attr_shuffled_seed0_$(date +%Y%m%d_%H%M%S) --precision amp
```

Validation logs contain final Top-1 / Top-5 in the `Testing Results: Prec@1 ... Prec@5 ...` line. [`train_comp_CoAPT.py`](train_comp_CoAPT.py) additionally logs `IADF mean modality weights (GT-conditioned; IFrame/MV/Residual): ...` once per validation. These values are averages over each sample's ground-truth-class-conditioned IADF route, so they are diagnostic only and are not used to compute predictions. To inspect a completed experiment:

```sh
grep -E 'Testing Results: Prec@1|IADF mean modality weights' exps/sthv2/ViT-B/16/<run_name>/log.txt
```

### 3.3 Testing

Standard testing entry:

```sh
python test.py --config configs/hmdb51/hmdb_CLIP_fix_L_0_11_fusion.yaml
```

If using the provided shell script:

```sh
sh scripts/run_test.sh configs/hmdb51/hmdb_CLIP_fix_L_0_11_fusion.yaml
```

### 3.4 Zero-shot Testing

```sh
sh scripts/run_test_zeroshot.sh configs/hmdb51/hmdb_CLIP_fix_L_0_11_fusion.yaml
```

or

```sh
python test_zeroshot.py --config configs/hmdb51/hmdb_CLIP_fix_L_0_11_fusion.yaml
```

---

## 4. Project Structure

```txt
BIKE/
├── .git/                         # Git metadata
├── Coviar/                       # COViar-related code and compressed video utilities
├── X_CLIP/                       # X-CLIP submodule code and dependencies
├── attributes/                   # Attribute vocabularies and auxiliary attribute resources
├── clip/                         # CLIP model definitions, tokenizers, and fusion variants
├── configs/                      # Dataset-specific configs
│   ├── hmdb51/
│   ├── ucf101/
│   └── k400/
├── datasets/                     # Dataset loaders and video / LMDB pipelines
│   ├── video.py
│   ├── video_lmdb.py
│   └── video_attr.py
├── docs/                         # Documentation assets, figures, and diagrams
├── exps/                         # Experiment outputs, checkpoints, logs, analysis artifacts
├── exps_few/                     # Few-shot experiment outputs and checkpoints
├── lists/                        # Dataset list files and label/annotation indexes
├── modules/                      # Core model modules
│   ├── fusion modules
│   ├── temporal modules
│   ├── prompt modules
│   └── compression blocks
├── outputs/                      # Default output directory
├── outputs-hmdb51/               # HMDB51-specific outputs
├── outputs——dataname/            # Outputs grouped by dataset name
├── qwen/                         # Qwen-related scripts and resources
├── retrieval_results/            # Retrieval evaluation result files
├── scripts/                      # Run scripts
│   ├── run_train_23.sh
│   ├── run_test.sh
│   └── run_test_zeroshot.sh
├── utils/                        # Shared utilities
├── video_sentence_fusion/        # Video-text fusion modules and helpers
├── visualization_comparison/     # Visualization assets and comparison reports
├── train.py                      # Main training entry
├── train3.py                     # Alternative training entry
├── train_comp_CoAPT.py           # CoAPT training pipeline
├── train_comp_CoAPT_fusion.py    # CoAPT training with fusion variants
├── train_comp_CoAPT_lmdb.py      # CoAPT training with LMDB-backed data
├── train_attributes.py           # Attribute branch training
├── train_attribute_qwen.py       # Attribute training with Qwen inputs
├── train_qwen_description.py     # Qwen description generation/training
├── test.py                       # Standard evaluation
├── test_zeroshot.py              # Zero-shot evaluation
├── test_charades.py              # Charades evaluation
├── test_comp_CoAPT.py            # CoAPT evaluation
├── retrieval_eval.py             # Retrieval evaluation
├── attribute_search.py           # Attribute search/selection utilities
├── grad_cam.py                   # Grad-CAM visualization
├── dataloader_test.py            # Data loader sanity checks
└── BIKE_inference.ipynb          # Notebook for inference demos
```

---

## 5. Key Files

### Training

- `train.py`  
  Main training entry point.

- `train3.py`  
  Alternative training script.

- `train_comp_CoAPT.py`  
  Training script for CoAPT-style compressed video recognition.

- `train_comp_CoAPT_fusion.py`  
  Training with fusion variants.

- `train_comp_CoAPT_lmdb.py`  
  Training with LMDB-backed video loading.

- `train_attributes.py`  
  Attribute branch training.

- `train_attribute_qwen.py`  
  Attribute-related training with Qwen-generated inputs.

- `train_qwen_description.py`  
  Qwen description generation/training script.

### Testing

- `test.py`  
  Standard evaluation entry.

- `test_zeroshot.py`  
  Zero-shot evaluation.

- `test_charades.py`  
  Charades evaluation.

- `test_comp_CoAPT.py`  
  CoAPT evaluation.

### Analysis and Visualization

- `retrieval_eval.py`  
  Retrieval evaluation.

- `attribute_search.py`  
  Attribute search and selection utilities.

- `grad_cam.py`  
  Grad-CAM visualization.

- `dataloader_test.py`  
  Data loader sanity check.

---

## 6. Method Summary

The main TextComp-style framework contains two branches:

### 6.1 Visual Branch

The visual branch extracts compressed-domain representations:

- I-frame features
- Motion Vector features
- Residual features

The I-frame stream is encoded using a CLIP visual encoder. Motion vectors and residuals are processed with lightweight visual encoders initialized from CLIP visual layers.

### 6.2 Textual Branch

Instead of using only action class names, the method uses LLM-generated action attributes.

Example prompt for attribute generation:

```txt
Describe the visual characteristics of the action "{class name}" from the {dataset name} video action recognition dataset. Provide {n words} single descriptive words (NOT phrases) about the motion, posture, and movement patterns.
```

The generated attributes are inserted into a text prompt such as:

```txt
a video about {class name} {attribute 1} {attribute 2} ... {attribute L}
```

The textual prompt is then encoded by the CLIP text encoder.

### 6.3 Instance-Aware Dynamic Fusion

The Instance-Aware Dynamic Fusion module uses textual attributes to guide the fusion of compressed-domain modalities.

It dynamically assigns frame-level modality weights to:

- I-frame
- Motion Vector
- Residual

This allows the model to adaptively focus on the most reliable or discriminative modality for each video instance.

---

## 7. Datasets

Experiments are conducted on the following datasets:

| Dataset | Description |
|---|---|
| HMDB-51 | 51 action classes, 6,766 videos |
| UCF-101 | 101 action classes, 13,320 videos |
| Something-Something v2 | 174 fine-grained action classes |
| Kinetics-400 | 400 action classes, large-scale video benchmark |

For HMDB-51 and UCF-101, results are usually reported as the average over the three official splits.

---

## 8. Implementation Details

The main experimental settings are as follows:

| Setting | Value |
|---|---|
| Video codec | MPEG-4 |
| Average GOP setting | 1 I-frame + about 11 P-frames |
| Input resolution | `340 x 256` |
| Number of sampled clips | 16 |
| I-frame encoder | CLIP ViT-B/16 |
| Text encoder | CLIP ViT-B/32 |
| Base learning rate | `5e-5` |
| Weight decay | `2e-2` |
| Batch size | 32 |
| Training epochs | 20 |
| Warmup epochs | 5 |
| GPUs | 2 × NVIDIA 4090D |

---

## 9. Experimental Results

### 9.1 Fully-supervised Results on Kinetics-400

| Method | Modality | Backbone | GFLOPs | Top-1 | Top-5 | Zero-shot |
|---|---|---:|---:|---:|---:|---|
| MVFNet | RGB | ResNet50 | 1974.0 | 77.0 | 92.8 | No |
| TEA | RGB | ResNet50 | 2100.0 | 75.0 | 92.8 | No |
| TDN | RGB | ResNet50 | 3240.0 | 76.6 | 92.8 | No |
| VideoMAE | RGB | ViT-B | 1080.0 | 81.5 | 95.1 | No |
| ActionCLIP | RGB + Text | ViT-B | 563 | 83.8 | 96.2 | Yes |
| X-CLIP | RGB + Text | ViT-B | 145 | 83.8 | 96.7 | Yes |
| M2-CLIP | RGB + Text | ViT-B | 422 | 83.4 | 96.7 | Yes |
| Vita-CLIP | RGB + Text | ViT-B | 190 | 82.9 | 96.3 | Yes |
| MFCD-Net | I + M + R | MFNet3D | 1300.0 | 68.3 | - | No |
| MEACI-Net | I + M + R | I3D | 269.4 | 71.5 | - | No |
| DSDMTR | I + M + R | ResNet101 | 141 | 74.1 | - | No |
| CoViFocus | I + M | ResNet50 | 296 | 72.0 | - | No |
| TextComp | I + M + R + Text | ViT-B | 145.8 | 80.3 | 94.5 | Yes |

TextComp achieves strong compressed-domain performance on Kinetics-400 with only `145.8` GFLOPs.

---

### 9.2 Fully-supervised Results on HMDB-51, UCF-101, and SSv2

| Method | Modality | Backbone | GFLOPs | HMDB-51 | UCF-101 | SSv2 | Zero-shot |
|---|---|---:|---:|---:|---:|---:|---|
| MVFNet | RGB | ResNet50 | 1974.0 | 75.7 | 96.6 | 63.5 | No |
| TEA | RGB | ResNet50 | 2100.0 | 73.3 | 96.9 | 48.9 | No |
| TDN | RGB | ResNet50 | 3240.0 | 76.3 | 97.4 | 65.3 | No |
| VideoMAE | RGB | ViT-B | 1080.0 | 73.3 | 96.1 | 69.9 | No |
| Vita-CLIP | RGB + Text | ViT-B | 190 | - | - | 48.7 | Yes |
| M2-CLIP | RGB + Text | ViT-B | 422 | - | - | 66.9 | Yes |
| IPTSN | I + M + R | ResNet152 | 215.0 | 69.1 | 93.4 | - | No |
| CoViAR | I + M + R | ResNet152 | 1222.0 | 59.1 | 90.4 | - | No |
| CoViAR | I + M + R + Flow | ResNet152 | 3970.0 | 70.2 | 94.9 | - | No |
| DMC-Net | I + M + R + Flow | I3D | 401.0 | 71.8 | 92.3 | - | No |
| SIFT-Net | I + M + R | I3D | 1971.0 | 72.3 | 94.0 | - | No |
| MFCD-Net | I + M + R | MFNet3D | 1300.0 | 66.9 | 93.2 | - | No |
| MEACI-Net | I + M + R | I3D | 269.4 | 74.4 | 96.4 | - | No |
| DSDMTR | I + M + R | ResNet101 | 141 | 74.9 | 95.8 | - | No |
| CoViFocus | I + M | ResNet50 | 296 | 74.8 | 95.8 | - | No |
| MussNet | I + M + R | ResNet50-TSM | 38.7 | 63.7 | 89.1 | - | No |
| MussNet-LF | I + M + R | ResNet50 | 33.6 | 62.9 | 89.2 | - | No |
| PKD + WISE | I + M + R | ResNet152 | - | 68.2 | 92.9 | - | No |
| CFM-Net | I + M + R | UniFormer | 48 | 75.7 | 94.8 | - | No |
| MM-ViT | I + M + R | ViT-B | 820 | - | 93.3 | 64.9 | No |
| CVPT | I + M + R | ViT-B | 772.2 | 69.7 | 95.5 | - | No |
| CVPT | I + M + R | ViT-L | 2569.6 | 81.5 | 98.0 | 65.5 | No |
| TextComp | I + M + R + Text | ViT-B | 145.8 | 76.2 | 96.7 | 60.4 | Yes |

TextComp achieves competitive or superior performance among compressed-domain methods while maintaining relatively low computational cost.

---

### 9.3 Zero-shot Results

Zero-shot evaluation is conducted on HMDB-51 and UCF-101 using models pre-trained on Kinetics-400.

| Method | Modality | HMDB-51 | UCF-101 |
|---|---|---:|---:|
| ActionCLIP | RGB + Text | 40.8 ± 5.4 | 58.3 ± 3.4 |
| X-CLIP | RGB + Text | 44.6 ± 5.2 | 72.0 ± 2.3 |
| ViFi-CLIP | RGB + Text | 51.3 ± 0.6 | 76.8 ± 0.7 |
| Vita-CLIP | RGB + Text | 48.6 ± 0.6 | 75.0 ± 0.6 |
| M2-CLIP | RGB + Text | 47.1 ± 0.4 | 78.7 ± 1.2 |
| TextComp | I + M + R + Text | 49.4 ± 1.1 | 73.6 ± 1.0 |

TextComp shows strong zero-shot generalization despite using compressed-domain visual inputs.

---

### 9.4 Few-shot Results

Few-shot results on HMDB-51 and UCF-101.

| Method | Modality | HMDB K=2 | HMDB K=4 | HMDB K=8 | HMDB K=16 | UCF K=2 | UCF K=4 | UCF K=8 | UCF K=16 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ActionCLIP | RGB + Text | 55.0 | 56.0 | 58.0 | - | 80.0 | 85.0 | 89.0 | - |
| X-CLIP | RGB + Text | 53.0 | 57.3 | 62.8 | 64.0 | 76.4 | 83.4 | 88.3 | 91.4 |
| X-Florence | RGB + Text | 51.6 | 57.8 | 64.1 | 64.2 | 84.0 | 88.5 | 92.5 | 94.8 |
| ViFi-CLIP | RGB + Text | 57.2 | 62.7 | 64.5 | 66.8 | 80.7 | 85.1 | 90.0 | 92.7 |
| TextComp | I + M + R + Text | 65.2 | 65.3 | 67.2 | 70.9 | 91.1 | 92.9 | 94.6 | 95.4 |

TextComp performs especially well in few-shot scenarios, indicating that attribute-guided semantic supervision helps the model adapt to new classes with limited samples.

---

## 10. Ablation Studies

### 10.1 Attribute Generator

| Vocabulary / LLM | HMDB-51 | UCF-101 |
|---|---:|---:|
| Qwen3:8B | 76.0 | 96.8 |
| Llama3.1:8B | 76.2 | 96.7 |

Both LLMs provide effective semantic attributes. Llama3.1:8B is slightly better on HMDB-51, while Qwen3:8B is slightly better on UCF-101.

---

### 10.2 Attribute Prompt Design

| Prompt | HMDB-51 | UCF-101 |
|---|---:|---:|
| `a video about {class}` | 74.4 | 95.2 |
| `Train token {class}` | 75.2 | 96.1 |
| `a video about {class} {Attr}` | 75.9 | 96.4 |

Attribute-enhanced prompts provide better supervision than class-name-only prompts or learnable prompt tokens.

---

### 10.3 Fusion Weight Module

| Fusion Weight Strategy | HMDB-51 | UCF-101 |
|---|---:|---:|
| Fixed | 75.9 | 96.4 |
| Trainable | 76.0 | 96.5 |
| IADF | 76.2 | 96.7 |

The Instance-Aware Dynamic Fusion module achieves the best performance.

---

### 10.4 Dataset Name in Prompt

| Strategy | HMDB-51 | UCF-101 |
|---|---:|---:|
| Without dataset name | 75.4 | 96.5 |
| With dataset name | 76.2 | 96.7 |

Adding the dataset name to the LLM prompt helps generate more domain-specific attributes.

---

### 10.5 Inference Efficiency

| Method | Preprocess Time / video | Inference Time / video | Full Pipeline |
|---|---:|---:|---:|
| ActionCLIP | 41.51 ms | 11.52 ms | 53.03 ms |
| TextComp | 6.11 ms | 14.23 ms | 20.34 ms |

TextComp reduces preprocessing time by directly using compressed video streams and achieves a faster full pipeline.

---

## 11. Common Commands

### Train on HMDB-51

```sh
sh scripts/run_train_23.sh configs/hmdb51/hmdb_CLIP_fix_L_0_11_fusion.yaml
```

### Test on HMDB-51

```sh
sh scripts/run_test.sh configs/hmdb51/hmdb_CLIP_fix_L_0_11_fusion.yaml
```

### Zero-shot Evaluation

```sh
sh scripts/run_test_zeroshot.sh configs/hmdb51/hmdb_CLIP_fix_L_0_11_fusion.yaml
```

### Estimate TextComp FLOPs / Views

```sh
python scripts/estimate_textcomp_flops.py --config configs/sthv2/sthv2_pre_fix_B16.yaml
python scripts/estimate_textcomp_flops.py --config configs/sthv2/sthv2_pre_fix_B16.yaml --class-conditioned
```

The first command reports the standard per-view paper-table cost. The second command also counts the current class-conditioned validation path where fusion and temporal head are repeated for every class.

### Check Data Loader

```sh
python dataloader_test.py
```

### Run Grad-CAM Visualization

```sh
python grad_cam.py
```

---

## 12. Output Directories

Experiment results and checkpoints are saved under:

```txt
exps/
exps_few/
outputs/
outputs-hmdb51/
outputs——dataname/
retrieval_results/
visualization_comparison/
```

The exact output path depends on the config file and training script.

---

## 13. Notes

1. Make sure the dataset videos are encoded in MPEG-4 format before training.
2. Check the dataset paths in the corresponding YAML config files under `configs/`.
3. If using LMDB-backed loading, prepare LMDB files before running `train_comp_CoAPT_lmdb.py`.
4. Attribute files should be placed under `attributes/`.
5. For zero-shot or few-shot experiments, ensure the corresponding split files and class names are correctly configured.
6. COViar installation and video codec compatibility are important for extracting I-frames, motion vectors, and residuals.

---

## 14. Citation / Reference

This repository is used for experiments related to compressed video action recognition and text-guided semantic fusion.

If you use this code or results, please cite the corresponding compressed video action recognition and vision-language learning works, such as:

- CoViAR
- CLIP
- X-CLIP
- CVPT
- TextComp

---

## 15. Contact

This is a local research repository. Please check the configuration files and scripts for dataset-specific paths before running experiments.
