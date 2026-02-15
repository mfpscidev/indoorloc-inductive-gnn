# Inductive learning scheme for WiFi fingerprinting-based indoor positioning systems


## Repository structure

* **notebooks/:** \
    Jupyter notebooks used as the working environment.

    - `evaluation.ipynb`: Optimization, training and evaluation of graph neural networks.

* **src/:** \
    Python modules containing the core classes and functions used throughout the notebooks.

   - `indoorloc_data.py`: Manages data loading and processing, including graph construction.
   - `indoorloc_trainer.py`: Manages model optimization, training and evaluation.
   - `indoorloc_models.py`: Contains the GNN models.
   - `indoorloc_viz.py`: Manages the generation of plots.  
   - `indoorloc_enums.py`: Contains constants and enums used in the other modules.

---

## Requirements

Install all dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt