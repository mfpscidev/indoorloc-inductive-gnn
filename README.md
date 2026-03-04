# Inductive learning scheme for WiFi fingerprinting-based indoor positioning systems

Traditionally, indoor positioning systems based on WiFi fingerprinting have relied on k-NN algorithms and, more recently, on deep learning models.

This repository contains the code for a framework developed to use **Graph Neural Networks (GNNs)** with both inductive and transductive learning schemes.

In the **transductive scheme**, a single graph is created that includes all dataset splits (train, validation, and test), whereas in the **inductive scheme**, separate graphs are constructed for each split to prevent phenomena such as *data leakage* and to enable evaluation on unseen graphs without retraining the model each time.

The framework supports preprocessing of the original datasets using linear and power normalization, as well as dimensionality reduction via PCA. It also provides flexibility in constructing k-NN graphs, allowing selection of different distance metrics (Manhattan and Cosine) and the number of nearest neighbors *k*.


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

**The graphs were generated from the TUT5 dataset.*

### Inductive learning scheme

<table>
  <tr>
    <td colspan="2" align="center">
      <img src="images/inductive_tut5.png" width="800">
    </td>
  </tr>
</table>

**The graphs were generated from the TUT5 dataset.*

## Summary of results

Results obtained after 10 runs using a random split of the training dataset each time (80/20).

### Regression (Prediction of coordinates)

<table border="1" style="border-collapse: collapse; text-align:center;">
    <thead>
        <tr>
            <th colspan="4" style="text-align:center;">Mean Position Error (m)</td>
        </tr>
        <tr>
            <th>Dataset</th>
            <th>Best k-NN <br> (Optimized)</th>
            <th>GraphSAGE <br>(Transductive)</th>
            <th>GraphSAGE <br> (Inductive)</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>UJI1</td><td>7.33</td><td>7.36 ± 0.12</td><td>8.72 ± 0.20</td></tr>
        <tr><td>UTS1</td><td>6.50</td><td>7.10 ± 0.12</td><td>7.39 ± 0.23</td></tr>
        <tr><td>SAH1</td><td>5.93</td><td>5.85 ± 0.56</td><td>5.79 ± 0.97</td></tr>
        <tr><td>TIE1</td><td>2.36</td><td>3.14 ± 0.85</td><td>3.34 ± 0.63</td></tr>
        <tr><td>TUT1</td><td>4.43</td><td>6.46 ± 0.19</td><td>6.80 ± 0.21</td></tr>
        <tr><td>TUT2</td><td>8.37</td><td>9.46 ± 0.62</td><td>9.46 ± 0.39</td></tr>
        <tr><td>TUT3</td><td>7.76</td><td>7.65 ± 0.13</td><td>7.71 ± 0.25</td></tr>
        <tr><td>TUT4</td><td>5.20</td><td>5.38 ± 0.09</td><td>5.71 ± 0.09</td></tr>
        <tr><td>TUT5</td><td>5.22</td><td>6.19 ± 0.22</td><td>6.43 ± 0.26</td></tr>
        <tr><td>SOD1</td><td>2.43</td><td>2.56 ± 0.10</td><td>2.59 ± 0.07</td></tr>
        <tr><td>SOD2</td><td>1.54</td><td>1.62 ± 0.08</td><td>1.60 ± 0.11</td></tr>
        <tr><td>SOD6</td><td>3.47</td><td>3.52 ± 0.11</td><td>3.32 ± 0.13</td></tr>
    </tbody>
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