# Comparing GAN-Based and Diffusion-Based Synthetic Image Augmentation for Class-Imbalanced Melanoma Classification

Bachelor's thesis project by **Matei-Cristian Baciu** at **SRH University of Applied Sciences**.

This repository contains the notebooks and evaluation code used to compare traditional augmentation, class weighting, GAN-generated images, and diffusion-generated images for binary melanoma classification on the HAM10000 collection. All classifier configurations use EfficientNet-B0 and are evaluated on the same unchanged test set containing only real dermoscopic images.

## Links

- **HAM10000 collection:** [ISIC Archive](https://api.isic-archive.com/collections/212/)

Large checkpoints, generated image collections, and the original dataset are intentionally excluded from Git because of their size. The repository contains the code required to reproduce the data preparation, training, and evaluation workflow.

## Research objective

The project investigates whether synthetic melanoma images can improve classification performance when the training data are strongly imbalanced. It addresses three questions:

1. How does an EfficientNet-B0 baseline perform when trained only on the original imbalanced data?
2. How do traditional augmentation, class weighting, GAN-based augmentation, and diffusion-based augmentation affect minority-class performance?
3. Which strategy provides the strongest observed balance between precision and recall when every model is evaluated on the same real test set?

## Dataset and split

The study uses the HAM10000 collection downloaded from the ISIC Archive. Images were divided at the **lesion level** so that images belonging to the same lesion could not appear in more than one subset.

| Split | Total images | Melanoma | Non-melanoma | Melanoma proportion |
|---|---:|---:|---:|---:|
| Training | 8,204 | 943 | 7,261 | 11.49% |
| Validation | 1,742 | 193 | 1,549 | 11.08% |
| Test | 1,774 | 169 | 1,605 | 9.53% |

The split was created with `GroupShuffleSplit`, grouped by `lesion_id`, using random state 42. The validation and test sets remained unchanged throughout the project and contained no augmented or generated images.

## Experiments

Eight classifier configurations were evaluated:

| ID | Configuration | Training images | Melanoma images | Non-melanoma images |
|---|---|---:|---:|---:|
| E1 | Baseline | 8,204 | 943 | 7,261 |
| E2a | Traditional augmentation — doubled | 9,147 | 1,886 | 7,261 |
| E2b | Traditional augmentation — balanced | 14,522 | 7,261 | 7,261 |
| E3 | Class weighting | 8,204 | 943 | 7,261 |
| E4a | GAN augmentation — doubled | 9,147 | 1,886 | 7,261 |
| E4b | GAN augmentation — balanced | 14,522 | 7,261 | 7,261 |
| E5a | Diffusion augmentation — doubled | 9,147 | 1,886 | 7,261 |
| E5b | Diffusion augmentation — balanced | 14,522 | 7,261 | 7,261 |

### Traditional augmentation

Melanoma entries were duplicated or sampled with replacement. Random crops, flips, rotations, and color transformations were applied to the added entries when they were loaded during training. The doubled configuration added 943 entries; the balanced configuration added 6,318 entries.

### Class weighting

The original imbalanced training set was retained. A positive-class weight derived from the training distribution was supplied to `BCEWithLogitsLoss`, increasing the penalty for errors on melanoma images.

### GAN augmentation

A publicly available melanoma-generation GAN checkpoint was fine-tuned using only the 943 melanoma images in the training split. Checkpoints at 200 kimg and 500 kimg were compared through classifiers trained under the same conditions. The 500-kimg checkpoint was selected using validation F1-score and was used to generate the final augmentation sets.

### Diffusion augmentation

Stable Diffusion XL 1.0 was adapted to the dermoscopic melanoma domain using DreamBooth with LoRA. Fine-tuning used the 943 training-set melanoma images for 2,000 optimization steps, with checkpoints saved every 500 steps. The predetermined 2,000-step checkpoint was used to generate the final synthetic datasets. Full prompts and generation parameters are documented in the diffusion notebooks.

## Classifier training

All classifier experiments use the same general setup:

- EfficientNet-B0 initialized with ImageNet-pretrained weights
- complete network fine-tuning
- binary output with `BCEWithLogitsLoss`
- Adam optimizer
- learning rate: `1e-4`
- batch size: 32
- maximum of 20 epochs for the main experiments
- checkpoint selected by validation F1-score
- classification threshold: 0.5
- random seed: 42
- image input size: 224 × 224 pixels

Most classifier runs were performed on an NVIDIA L4 GPU. GAN and diffusion fine-tuning were performed on an NVIDIA A100-SXM4-40GB GPU.

## Test results

| Method | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | **92.22%** | **60.99%** | 50.89% | 55.48% | 87.72% |
| Traditional augmentation — doubled | **92.22%** | **60.99%** | 50.89% | 55.48% | 89.61% |
| Traditional augmentation — balanced | 90.76% | 51.34% | 56.80% | 53.93% | 89.86% |
| Class weighting | 89.35% | 45.90% | **66.27%** | 54.24% | 89.27% |
| GAN augmentation — doubled | 91.26% | 53.89% | 57.40% | 55.59% | 89.51% |
| GAN augmentation — balanced | 91.54% | 55.90% | 53.25% | 54.55% | 89.29% |
| Diffusion augmentation — doubled | 90.81% | 51.46% | 62.72% | **56.53%** | 89.32% |
| Diffusion augmentation — balanced | 90.98% | 52.26% | 61.54% | 56.52% | **90.00%** |

The doubled diffusion configuration achieved the highest observed F1-score. Class weighting achieved the highest recall and fewest false negatives, but with substantially lower precision. All imbalance-handling configurations achieved a higher ROC-AUC than the baseline, although only doubled GAN augmentation and the two diffusion configurations achieved a higher test F1-score.

These are results from one dataset split and one main random seed. Small differences should not be interpreted as definitive rankings of the methods.

## Repository structure

```text
.
├── data/                              # Local metadata and split files (large data excluded)
├── notebooks/
│   ├── 00_eda_and_dataset_setup.ipynb
│   ├── 01_baseline_efficientnet.ipynb
│   ├── 02_traditional_augmentation/   # Doubled and balanced experiments
│   ├── 03_class_weighting.ipynb
│   ├── 04_gan/                        # Fine-tuning, generation and classifiers
│   ├── 05_diffusion/                  # LoRA fine-tuning, generation and classifiers
│   └── roc_accuracy_precision_curves.ipynb
├── results/
│   ├── confusion_matrices/
│   └── figures/
├── .gitignore
└── README.md
```

Some filenames may include experiment-specific suffixes. The numerical prefixes indicate the intended execution order.

## Running the project

### 1. Clone the repository

```bash
git clone https://github.com/zone2233/bachelor-thesis-melanoma-augmentation.git
cd bachelor-thesis-melanoma-augmentation
```

### 2. Obtain the data

Download the HAM10000 collection from the [ISIC Archive](https://api.isic-archive.com/collections/212/) and place the images and metadata in your chosen data directory.

The notebooks were primarily developed in Google Colab and use Google Drive paths. Update the path-configuration cells to match your local or Drive directory before running them.

### 3. Install dependencies

The exact setup cells are included in the relevant notebooks. The main Python dependencies include:

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn pillow
pip install diffusers transformers accelerate peft safetensors
```

GAN notebooks may require additional dependencies included by their setup cells. A CUDA-capable GPU is strongly recommended for training and image generation.

### 4. Run the notebooks in order

1. `00_eda_and_dataset_setup.ipynb`
2. `01_baseline_efficientnet.ipynb`
3. notebooks in `02_traditional_augmentation/`
4. `03_class_weighting.ipynb`
5. notebooks in `04_gan/`
6. notebooks in `05_diffusion/`
7. `roc_accuracy_precision_curves.ipynb`

The GAN and diffusion generation notebooks create new training-split CSV files for the doubled and balanced configurations. They do not modify the validation or test splits.

## Reproducibility notes

- The split is grouped by lesion identifier to reduce data leakage.
- Validation and test images are always real and unchanged.
- Random seed 42 is used where supported.
- Model selection is performed on validation F1-score, never on the test set.
- Accuracy, precision, recall, F1-score and confusion matrices use a threshold of 0.5.
- ROC-AUC and average precision are calculated from predicted probabilities.
- Large artifacts are stored separately and are not tracked by Git.

GPU kernels, data-loader ordering, software versions, and other sources of randomness may still produce small differences when the experiments are repeated.

## Limitations

- Each classifier configuration was evaluated using one main seed and one lesion-level split.
- Only EfficientNet-B0 was evaluated as the classifier.
- Generated images were assessed mainly through visual inspection and their effect on classification results; they were not evaluated by dermatologists.
- No external dataset was used for independent validation.
- The dataset contains imaging artifacts, and no manual artifact-removal procedure was applied before training.

Additional limitations and a complete discussion are provided in the thesis report.

## Author

**Matei-Cristian Baciu**  
Bachelor of Computer Science: Healthcare Management  
SRH University of Applied Sciences, Berlin
