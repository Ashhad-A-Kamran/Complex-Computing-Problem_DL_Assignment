# Note to Sir Hamza

The files you should view are:
1. **[notebook.ipynb](https://github.com/Ashhad-A-Kamran/Complex-Computing-Problem_DL_Assignment/blob/main/notebook.ipynb)** — the original Jupyter Notebook.
2. **[Report.pdf](https://github.com/Ashhad-A-Kamran/Complex-Computing-Problem_DL_Assignment/blob/main/Report.pdf)** — the final technical report and evaluation dashboard.

*(Note: GitHub's PDF preview renderer is sometimes fragile and disables internal links. For direct access to the pre-executed Generation PDFs, please use the links below!)*

**Direct Links to Executed Generations:**
- [Generation 1: Baseline DNN](https://github.com/Ashhad-A-Kamran/Complex-Computing-Problem_DL_Assignment/blob/main/scripts/generation1.pdf)
- [Generation 2: Recurrent Sequence Models](https://github.com/Ashhad-A-Kamran/Complex-Computing-Problem_DL_Assignment/blob/main/scripts/generation2.pdf)
- [Generation 3: ClinicalBERT](https://github.com/Ashhad-A-Kamran/Complex-Computing-Problem_DL_Assignment/blob/main/scripts/generation3.pdf)


# Early Warning System for Patient Deterioration

This repository contains the codebase and technical report for a deep learning-based early warning system designed to predict patient deterioration in a critical care setting.

The project models the progression of patient deterioration through three architectural generations, moving from basic tabular data processing to advanced temporal sequence modeling, and finally to unstructured clinical language processing using Transformers.

## Project Structure

```text
DL_assigment/
├── README.md                 # Project documentation
├── Report.pdf                # Compiled final technical report
├── notebook.ipynb            # Master Directory Notebook linking to generations
├── notebook.pdf              # PDF version of Master Directory
├── scripts/                  # Pre-executed notebooks, PDFs, and code logic
│   ├── generation1.ipynb / .pdf
│   ├── generation2.ipynb / .pdf
│   ├── generation3.ipynb / .pdf
│   ├── generation_1.py       # Baseline DNN training script
│   ├── unified_eval.py       # Unified metric aggregation script
│   ├── extended_eval.py      # Script to generate extensive tradeoff visualizations
│   ├── convert_to_pdf.py     # Utility to convert notebooks to PDF
│   └── merge_notebooks.py    # Utility to generate the master notebook index
├── data/                     # Raw and processed datasets
└── assets/                   # Evaluation results, charts, and confusion matrices
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
