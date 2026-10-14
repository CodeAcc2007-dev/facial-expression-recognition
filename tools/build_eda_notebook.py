from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_eda.ipynb"


def build() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "FER V2 Python", "language": "python", "name": "fer-v2"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# FER Challenge V2 — Data Audit and EDA

This notebook verifies the official manifests and summarizes the immutable audit produced by
`tools/run_audit.py`. The development split is grouped by decoded-pixel SHA-256 so an exact
duplicate face cannot appear in both training and validation. The hidden test set is inspected
only for file integrity and duplicate-overlap disclosure; it is never assigned labels or used
for training/model selection."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path.cwd().resolve()
DATA = ROOT / 'data' / 'raw' / 'fer_challenge'
EDA = ROOT / 'outputs' / 'eda'
summary = json.loads((EDA / 'audit_summary.json').read_text())
train = pd.read_csv(DATA / 'train.csv')
development_train = pd.read_csv(EDA / 'train_split.csv')
validation = pd.read_csv(EDA / 'validation_split.csv')
summary"""
        ),
        nbf.v4.new_markdown_cell(
            """## Class balance

Accuracy is the competition's primary metric, but the severe class imbalance makes per-class
recall and Macro F1 necessary diagnostics. They help reveal whether a gain in overall accuracy
comes from ignoring rare expressions."""
        ),
        nbf.v4.new_code_cell(
            """class_order = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
counts = train['label'].value_counts().reindex(class_order)
percentages = 100 * counts / len(train)
display(pd.DataFrame({'count': counts, 'percent': percentages.round(2)}))

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(class_order, counts.values, color='#6C8CF5')
ax.set(title='Training class distribution', ylabel='Images')
ax.grid(axis='y', alpha=.2)
for bar, value in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, value + 90, f'{value:,}', ha='center', fontsize=8)
fig.tight_layout()
fig.savefig(EDA / 'class_distribution.png', dpi=180)
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Representative samples

These grids support visual understanding only. They do not change any organizer-provided label."""
        ),
        nbf.v4.new_code_cell(
            """rng = np.random.default_rng(42)
fig, axes = plt.subplots(7, 5, figsize=(10, 13))
for row_index, label in enumerate(class_order):
    choices = train.loc[train.label == label, 'id'].to_numpy()
    selected = rng.choice(choices, size=5, replace=False)
    for column_index, identifier in enumerate(selected):
        with Image.open(DATA / 'train' / identifier) as image:
            axes[row_index, column_index].imshow(image, cmap='gray', vmin=0, vmax=255)
        axes[row_index, column_index].axis('off')
        if column_index == 0:
            axes[row_index, column_index].set_title(label, loc='left', fontweight='bold')
fig.suptitle('Five organizer-labelled samples per class', y=.995)
fig.tight_layout()
fig.savefig(EDA / 'class_sample_grid.png', dpi=180)
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Ambiguity review sheet

The following reproducible sample is for the participant's own review. A low-resolution facial
crop may look ambiguous, but this notebook does **not** assert that any label is wrong and does
not modify labels. Descriptions such as *possible annotation issue* remain hypotheses."""
        ),
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(7, 20, figsize=(24, 9))
for row_index, label in enumerate(class_order):
    choices = validation.loc[validation.label == label, 'id'].to_numpy()
    selected = rng.choice(choices, size=20, replace=False)
    for column_index, identifier in enumerate(selected):
        with Image.open(DATA / 'train' / identifier) as image:
            axes[row_index, column_index].imshow(image, cmap='gray', vmin=0, vmax=255)
        axes[row_index, column_index].axis('off')
        if column_index == 0:
            axes[row_index, column_index].set_title(label, loc='left', fontsize=9, fontweight='bold')
fig.suptitle('Participant review sheet — descriptive inspection only', y=.995)
fig.tight_layout()
fig.savefig(EDA / 'ambiguity_review_grid.png', dpi=160)
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Leakage, duplicates, and normalization

Pixel statistics are calculated from the development-training partition only. Exact duplicates
are grouped before splitting; conflicting-label duplicate groups remain outside validation.
Train/test overlap is disclosed but never converted into test labels or prediction overrides."""
        ),
        nbf.v4.new_code_cell(
            """report = {
    'split_strategy': summary['split']['strategy'],
    'split_hash': summary['split_hash'],
    'train_rows': summary['split']['train_rows'],
    'validation_rows': summary['split']['validation_rows'],
    'exact_hash_overlap': summary['split']['exact_hash_overlap'],
    **summary['duplicates'],
    'training_only_mean': summary['normalization']['mean'],
    'training_only_std': summary['normalization']['std'],
}
pd.Series(report, name='value').to_frame()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Decision carried into modelling

- Keep one locked split for every deadline-gated ablation.
- Select checkpoints by validation accuracy first and Macro F1 second.
- Use ImageNet normalization for the first RGB-replicated pretrained reference; treat any
  alternative normalization as its own experiment.
- Keep the hidden test set outside training and model selection."""
        ),
    ]
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    notebook = build()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    if arguments.execute:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/fer-v2-matplotlib")
        client = NotebookClient(notebook, timeout=600, kernel_name="fer-v2", resources={"metadata": {"path": str(PROJECT_ROOT)}})
        client.execute()
        nbf.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
