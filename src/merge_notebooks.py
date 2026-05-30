import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

# Add Title
nb.cells.append(new_markdown_cell("# Early Warning System for Patient Deterioration\n\nThis notebook unifies the code from all three model generations: Baseline DNN, Recurrent Sequence Models, and Transformer NLP Models."))

# --- Generation 1 ---
nb.cells.append(new_markdown_cell("## Generation 1: Baseline DNN (Tabular Vital Signs)"))
with open("Part_A/generation_1.py", "r") as f:
    gen1_code = f.read()
nb.cells.append(new_code_cell(gen1_code))

# --- Generation 2 ---
nb.cells.append(new_markdown_cell("## Generation 2: Recurrent Sequence Models (LSTM, GRU, BiLSTM)"))
with open("Part_A/generation_2.ipynb", "r") as f:
    gen2_nb = nbformat.read(f, as_version=4)
    
# Extract code cells from gen2
for cell in gen2_nb.cells:
    if cell.cell_type == "code":
        # Ignore empty cells or simple imports if you want, but appending all code cells is safest
        nb.cells.append(new_code_cell(cell.source))
    elif cell.cell_type == "markdown":
        nb.cells.append(new_markdown_cell(cell.source))

# --- Generation 3 ---
nb.cells.append(new_markdown_cell("## Generation 3: ClinicalBERT NLP (Unstructured Text Notes)"))
with open("generate_gen3.py", "r") as f:
    gen3_code = f.read()
nb.cells.append(new_code_cell(gen3_code))

# Write unified notebook
with open("notebook.ipynb", "w") as f:
    nbformat.write(nb, f)

print("Successfully merged generations into notebook.ipynb")
