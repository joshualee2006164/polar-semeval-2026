# Multilingual Polarization Detection – SemEval POLAR

This repository contains the training script used for our submission to POLAR Subtask 1.

## Model
FacebookAI/xlm-roberta-large

## Training Setup
- Batch size: 16
- Learning rate: 1e-5
- Epochs: 8
- Max length: 256
- Early stopping patience: 3

## Data Format

The script expects CSV files structured as:

train/
  en.csv
  de.csv
  ...

Each CSV must contain:
- id
- text
- polarization

## Run Training

python subtask_1_XLM-Roberta_2_git.py

## Notes
The training script combines all languages into a single dataset and applies language-aware sampling using a WeightedRandomSampler.
