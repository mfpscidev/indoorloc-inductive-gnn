# Inductive learning scheme for WiFi fingerprinting-based indoor positioning systems

### Transductive learning scheme

<table>
  <tr>
    <td align="center">
      <img src="images/transductive_class_tut5.png" width="400">
    </td>
    <td align="center">
      <img src="images/transductive_split_tut5.png" width="400">
    </td>
  </tr>
</table>

### Inductive learning scheme

<table>
  <tr>
    <td colspan="2" align="center">
      <img src="images/inductive_tut5.png" width="1000">
    </td>
  </tr>
</table>

## Repository structure

* **notebooks/:** \
    Jupyter notebooks used as the working environment.

    - `evaluation.ipynb`: Optimization, training and evaluation of graph neural networks.

* **src/:** \
    Python modules containing the core classes and functions used throughout the notebooks.

   - `indoorloc_data.py`: Manages data loading and processing, including graph construction.
   - `indoorloc_optimizer.py`: Manages model optimization.
   - `indoorloc_trainer.py`: Manages model training and evaluation.
   - `indoorloc_models.py`: Contains the GNN models.
   - `indoorloc_viz.py`: Manages the generation of plots.  
   - `indoorloc_enums.py`: Contains constants and enums used in the other modules.

---

## Requirements

Install all dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt