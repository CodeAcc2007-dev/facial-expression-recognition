from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def _notebook(title: str, cells: list[nbf.NotebookNode]) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "FER V2 Python", "language": "python", "name": "fer-v2"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [nbf.v4.new_markdown_cell(f"# {title}")] + cells
    return notebook


def build_error_analysis() -> nbf.NotebookNode:
    return _notebook(
        "FER V2 — Error Analysis",
        [
            nbf.v4.new_markdown_cell(
                """This notebook analyzes the frozen best-by-accuracy candidate on the same locked,
duplicate-safe validation holdout used for all comparisons. Descriptions of ambiguous examples are
review prompts only; they do not override or modify organizer labels."""
            ),
            nbf.v4.new_code_cell(
                """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image as DisplayImage, display
ROOT = Path.cwd().resolve()
selection = json.loads((ROOT / 'outputs' / 'selected_candidate.json').read_text())
analysis = json.loads((ROOT / 'outputs' / 'final_analysis' / 'summary.json').read_text())
selection['selected']"""
            ),
            nbf.v4.new_code_cell(
                """metrics = analysis['metrics']
display(pd.DataFrame([{'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1']}]))
display(pd.DataFrame(metrics['per_class']).T)
display(pd.DataFrame(analysis['top_confusions']))"""
            ),
            nbf.v4.new_markdown_cell(
                """The left matrix is row-normalized and therefore emphasizes recall. The right
matrix is column-normalized and therefore emphasizes precision. The gallery samples the three
largest off-diagonal confusion pairs without making a definitive annotation judgment."""
            ),
            nbf.v4.new_code_cell(
                """display(DisplayImage(filename=str(ROOT / 'outputs' / 'final_analysis' / 'confusion_matrices.png'), width=950))
display(DisplayImage(filename=str(ROOT / 'outputs' / 'final_analysis' / 'misclassification_gallery.png'), width=950))
pd.read_csv(ROOT / 'outputs' / 'final_analysis' / 'misclassification_gallery_manifest.csv').head(30)"""
            ),
        ],
    )


def build_interpretability() -> nbf.NotebookNode:
    return _notebook(
        "FER V2 — Interpretability and Calibration",
        [
            nbf.v4.new_markdown_cell(
                """Grad-CAM highlights spatial regions associated with a predicted-class logit. It
is qualitative evidence, not a causal explanation. Reliability compares model confidence with
observed validation accuracy; Expected Calibration Error (ECE) summarizes their bin-wise gap."""
            ),
            nbf.v4.new_code_cell(
                """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image as DisplayImage, display
ROOT = Path.cwd().resolve()
candidate = json.loads((ROOT / 'outputs' / 'selected_candidate.json').read_text())['selected']
analysis = json.loads((ROOT / 'outputs' / 'final_analysis' / 'summary.json').read_text())
member_summaries = {
    run_id: json.loads((ROOT / 'outputs' / 'checkpoints' / run_id / 'summary.json').read_text())
    for run_id in candidate['members']
}
gradcam_run = max(
    candidate['members'],
    key=lambda run_id: (
        member_summaries[run_id]['best_metrics']['accuracy'],
        member_summaries[run_id]['best_metrics']['macro_f1'],
    ),
)
print({'candidate': candidate['candidate_id'], 'Grad-CAM representative': gradcam_run,
       'ECE_10_bins': analysis['expected_calibration_error_10_bins']})"""
            ),
            nbf.v4.new_code_cell(
                """display(pd.read_csv(ROOT / 'outputs' / 'final_analysis' / 'calibration_bins.csv'))
display(DisplayImage(filename=str(ROOT / 'outputs' / 'final_analysis' / 'reliability_diagram.png'), width=600))
display(DisplayImage(filename=str(ROOT / 'outputs' / 'analysis' / gradcam_run / 'gradcam_grid.png'), width=850))
json.loads((ROOT / 'outputs' / 'analysis' / gradcam_run / 'gradcam_summary.json').read_text())"""
            ),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    notebooks = {
        ROOT / "notebooks" / "02_error_analysis.ipynb": build_error_analysis(),
        ROOT / "notebooks" / "03_interpretability.ipynb": build_interpretability(),
    }
    for path, notebook in notebooks.items():
        nbf.write(notebook, path)
        if arguments.execute:
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/fer-v2-matplotlib")
            NotebookClient(
                notebook,
                timeout=300,
                kernel_name="fer-v2",
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
            nbf.write(notebook, path)
        print(path)


if __name__ == "__main__":
    main()
