# Early Warning System for Patient Deterioration

This repository contains the codebase and technical report for a deep learning-based early warning system designed to predict patient deterioration in a critical care setting.

The project models the progression of patient deterioration through three architectural generations, moving from basic tabular data processing to advanced temporal sequence modeling, and finally to unstructured clinical language processing using Transformers.

## Project Structure

```text
DL_assigment/
├── README.md                 # Project documentation
├── Report.pdf                # Compiled final technical report
├── notebook.ipynb            # Master Directory Notebook linking to individual generations
├── scripts/                  # Pre-executed notebooks and generated PDFs
│   ├── generation1.ipynb / .pdf
│   ├── generation2.ipynb / .pdf
│   └── generation3.ipynb / .pdf
└── src/                      # Source code and assignment assets
    ├── pyproject.toml / uv.lock # Python environment configuration
    ├── convert_to_pdf.py     # Utility to convert notebooks to PDF
    ├── merge_notebooks.py    # Utility to generate the master notebook index
    ├── assignment.txt        # Original assignment prompt and rubrics
    ├── notebook.ipynb        # Source for Master Directory Notebook
    ├── data/                 # Raw and processed datasets
    ├── Part_A/               # Core implementation scripts for Generations 1-3
    │   ├── generation_1.py   # Baseline DNN training script
    │   ├── generation_2.ipynb # Recurrent sequence models (LSTM, GRU, BiLSTM)
    │   ├── generation_3.ipynb # ClinicalBERT Fine-tuning
    │   ├── unified_eval.py   # Unified metric aggregation script
    │   └── extended_eval.py  # Script to generate extensive tradeoff visualizations
    ├── Part_B/               # LaTeX Report files
    │   ├── report.tex        # Final technical report source
    │   └── *.png             # Visualizations and confusion matrices
    └── assets/               # Evaluation results and model metric logs
        ├── generation_1/     # DNN metrics
        ├── generation_2/     # RNN metrics
        ├── generation_3/     # ClinicalBERT metrics
        └── evaluation/       # Unified tables and plots
```

## Model Generations

The project investigates three paradigms of patient deterioration prediction:

### 1. Generation 1: Baseline DNN
A deep neural network (DNN) trained purely on static snapshots of patient vital signs. This established a critical baseline for analyzing the highly imbalanced nature of clinical data and comparing optimizers (Adam vs. SGD).

### 2. Generation 2: Recurrent Sequence Models
This generation shifted the paradigm to sequential modeling. Using 24-hour windows of vital signs, we implemented and compared **LSTM**, **GRU**, and **BiLSTM** architectures. The models successfully captured the temporal trajectory of deterioration, significantly outperforming the static DNN.

### 3. Generation 3: ClinicalBERT (NLP)
Vital signs are often trailing indicators. To capture leading indicators of deterioration, we incorporated unstructured nursing and physician notes using **ClinicalBERT**. This generation leveraged the self-attention mechanism of Transformers to highlight crucial clinical warnings (e.g., "hypotension", "desaturating"). We empirically compared Frozen base fine-tuning against Full fine-tuning.

## Unified Evaluation
A comprehensive evaluation of all six architectures is documented in the final report. This includes:
- Confusion Matrices (highlighting the critical balance between Recall and Alarm Fatigue)
- Training Latency vs. Recall Trade-off plots
- F1 Score comparative charts

## How to Navigate
Due to the massive computational time required for Generation 3 (Transformers), the codebase is pre-executed. 
1. **Start** by opening `notebook.ipynb` in the root directory. It serves as a master directory with links to the executed PDFs/Notebooks for each generation.
2. **Read** the final technical report by compiling `Part_B/report.tex`.

## Dependencies
This project uses the modern `uv` Python package manager. The environment is locked in `uv.lock` and configured in `pyproject.toml`.
To run any of the evaluation scripts manually:
```bash
uv run python Part_A/extended_eval.py
```
