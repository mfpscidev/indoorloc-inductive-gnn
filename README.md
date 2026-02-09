# Indoor positioning systems under the inductive learning scheme


## Repository structure

* **notebooks/:** \
    Jupyter notebooks used as the working environment.

    - `evaluation.ipynb`: Optimization, training and evaluation of graph neural networks.

* **src/:** \
    Python modules containing the core classes and functions used throughout the notebooks.

   - `indoorloc_data.py`: Manages data loading and processing, including graph construction.
   - `indoorloc_models.py`: Manages model training and evaluation.
   - `indoorloc_vizs.py`: Manages the generation of plots.  
   - `indoorloc_enums.py`: Contains constants and enums used in the other modules.

---

## Requirements

Install all dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt