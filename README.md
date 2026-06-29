# Bachelor Thesis

## Title
Comparing GAN-Based and Diffusion-Based Synthetic Image Augmentation for Class-Imbalanced Melanoma Classification

## Description
This repository contains the code, experiment logs, notebooks, and results for my bachelor thesis. The project compares traditional augmentation, class weighting, GAN-based synthetic image augmentation, and diffusion-based synthetic image augmentation for binary melanoma classification.

## Core Experiments

| ID | Training Setup |
|---|---|
| E1 | Real imbalanced training data only |
| E2 | Real data + traditional augmentation |
| E3 | Real data + class weighting or weighted sampler |
| E4 | Real data + GAN-generated melanoma images |
| E5 | Real data + diffusion-generated melanoma images |
| E6 | Real data + traditional augmentation + GAN images |
| E7 | Real data + traditional augmentation + diffusion images |

## Project Structure

```text
data/
notebooks/
src/
results/
generated_images/
thesis/