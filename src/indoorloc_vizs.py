import platform
import psutil
import cpuinfo
from typing import Dict, List
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from collections import Counter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib.collections import PolyCollection
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import linregress
import statsmodels.api as sm
import networkx as nx
import torch
from torch_geometric.utils.convert import to_networkx

from indoorloc_enums import (
    Devices,
    Subsets,
    PlotLabels,
)

# Constants
SEED = 42
HEADER_SEPARATOR = "\n" + 50 * "#" + "\n"
SEPARATOR = 50 * "-"


class EnvironmentInfo:
    """
    Mostra informació del sistema i la GPU.
    """
    def __init__(self) -> None:
        self.system = platform.system()
        self.cpu = cpuinfo.get_cpu_info()['brand_raw']
        self.ram = psutil.virtual_memory().total / (1024**3)
        self.gpu = None
   
    def _show_gpu_info(self) -> None:
        print(f"Selected device {self.gpu}")
        print("CUDA version:", torch.version.cuda)
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        num_gpus = torch.torch.cuda.device_count()
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    def show(self) -> None:
        print(f"{HEADER_SEPARATOR}\tENVIRONMENT INFORMATION{HEADER_SEPARATOR}")
        print(f"Operating System: {self.system}")
        print(f"CPU: {self.cpu}")
        print(f"RAM: {self.ram:.2f} GB")
        print(SEPARATOR)

        if torch.cuda.is_available():
            self.gpu = torch.device(Devices.cuda.value)
            self._show_gpu_info()
        else:
            self.gpu = torch.device(Devices.cpu.value)
            print("CUDA is not available. The CPU will be used.")
            print(f"Selected device: {self.gpu}")
        print(SEPARATOR)


class TrainingVisualizer:
    """
    Gestiona la creació dels gràfics de les mètriques d'entrenament i validació.
    """
    def __init__(self) -> None:
        pass

    def plot_metrics(
        self,
        train: Dict[str, List[float]],
        validation: Dict[str, List[float]]
    ) -> None:
        if train.keys() != validation.keys():
            raise ValueError("Keys in `train` and `validation` dictionaries must match.")

        plot_size = [4, 3]
        metrics = list(train.keys())
        if len(metrics) == 1:
            fig = plt.figure(figsize=(plot_size[0], plot_size[1]))
            plt.plot(train[metrics[0]], label=Subsets.train.value)
            plt.plot(validation[metrics[0]], label=Subsets.validation.value)
            plt.ylabel('Pèrdua')
            plt.yscale('log')  
            plt.xlabel(PlotLabels.epoch.value)
            plt.legend()
        else:
            fig, ax = plt.subplots(
                1, len(metrics),
                figsize=(len(metrics) * plot_size[0], plot_size[1])
            )
            for i in range(len(metrics)):
                ax[i].plot(train[metrics[i]], label=Subsets.train.value)
                ax[i].plot(validation[metrics[i]], label=Subsets.validation.value)
                ax[i].set_ylabel('Pèrdua')
                ax[i].set_yscale('log')  
                ax[i].set_xlabel(PlotLabels.epoch.value)
                ax[i].legend()

        plt.tight_layout()
        plt.show()


class GraphVisualizer:
    """
    Gestiona el dibuix dels grafs.
    """
    def __init__(self):
        pass

    def _assign_node_colors_by_class(self, graphdata):
        """
        Assigna un color als nodes de cada classe.
        """
        colors = {i: plt.cm.tab20(i) for i in range(graphdata.num_classes)}
        unique_classes = sorted(graphdata.y.unique().tolist())
        class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
        node_colors = []
        
        for class_id in graphdata.y:
            class_id = class_id.item()
            index = class_to_index[class_id]
            node_colors.append(colors[index])
        
        return node_colors, colors, class_to_index

    def _assign_node_colors_by_split(self, graphdata):
        split_colors = {
            "train": "tab:blue",
            "val": "tab:orange",
            "test": "tab:green"
        }

        split_to_index = {split: idx for idx, split in enumerate(split_colors)}

        train_nodes = graphdata.train_mask.nonzero(as_tuple=True)[0].tolist()
        val_nodes = graphdata.val_mask.nonzero(as_tuple=True)[0].tolist()
        test_nodes = graphdata.test_mask.nonzero(as_tuple=True)[0].tolist()

        node_colors = []
        for idx in range(graphdata.num_nodes):
            if idx in train_nodes:
                node_colors.append(split_colors["train"])
            elif idx in val_nodes:
                node_colors.append(split_colors["val"])
            elif idx in test_nodes:
                node_colors.append(split_colors["test"])
            else:
                node_colors.append("lightgray")

        return node_colors, split_colors, split_to_index

    def _add_edge_weights_to_nx(self, G, graphdata):
        """
        Assigna els pesos de les arestes del graf si estan disponibles
        """
        if hasattr(graphdata, "edge_weight") and graphdata.edge_weight is not None:
            weights = graphdata.edge_weight.cpu().tolist()
        else:
            weights = [1.0] * G.number_of_edges()   

        for (u, v), w in zip(G.edges(), weights):
            G[u][v]["weight"] = float(w)

        return G
    
    def compact_cluster_layout_from_pyg(
        self, 
        data, 
        k=0.5, 
        iterations=300, 
        spacing=5.0,
        mode="grid"    
    ):
        """
        Calcula els clusters i assigna la nova posició als nodes.
        """
        G = to_networkx(data, to_undirected=True)
        G = self._add_edge_weights_to_nx(G, data)

        labels = data.y.cpu().tolist()
        classes = sorted(set(labels))

        pos = {}

        cluster_subpos = {}
        for cls in classes:
            nodes = [i for i, l in enumerate(labels) if l == cls]
            subG = G.subgraph(nodes)

            sub_pos = nx.spring_layout(subG)    
            cluster_subpos[cls] = sub_pos

        num_clusters = len(classes)

        if mode == "grid":
            grid_size = int(np.ceil(np.sqrt(num_clusters)))
            cluster_centers = []
            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx < num_clusters:
                        cluster_centers.append(np.array([i * spacing, j * spacing]))
                        idx += 1

        elif mode == "circle":
            radius = spacing * num_clusters / (2 * np.pi)
            angles = np.linspace(0, 2*np.pi, num_clusters, endpoint=False)
            cluster_centers = [
                np.array([radius * np.cos(a), radius * np.sin(a)]) 
                for a in angles
            ]

        elif mode == "random":
            cluster_centers = [np.random.randn(2) * spacing for _ in classes]

        else:
            raise ValueError("mode must be 'grid', 'circle', or 'random'")

        if num_clusters == 1:
            pos = cluster_subpos[0]
        else:
            for cls, center in zip(classes, cluster_centers):
                sub_pos = cluster_subpos[cls]
                for n, (x, y) in sub_pos.items():
                    pos[n] = center + np.array([x, y])

        return pos

    def draw_graph(
            self, 
            graphdata, 
            k=0.2, 
            iterations=200, 
            spacing=3.0, 
            scheme="inductive",
            cluster="split", 
            mode=None, 
            ax=None, 
            title=None, 
            out_path=None
    ):
        """
        Dibuixa el graf G amb Networkx.
        """
        if cluster == "class":
            node_colors, colors, class_to_index = self._assign_node_colors_by_class(graphdata)
        elif cluster == "split":
            node_colors, colors, class_to_index = self._assign_node_colors_by_split(graphdata)
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))

        G = to_networkx(graphdata, to_undirected=True)
        G = self._add_edge_weights_to_nx(G, graphdata)
        if mode != None:
            pos = self.compact_cluster_layout_from_pyg(graphdata, k, iterations, spacing, mode)
        pos = nx.spring_layout(G, seed=SEED)    

        nx.draw(
            G, pos, ax=ax,
            with_labels=False,
            node_size=40,
            node_color=node_colors,
            width=0.05,
            edgecolors='black',
            edge_color="#474545BA",       
            linewidths=0.1,
            alpha=0.8
        )

        patches = []
        for split, idx in class_to_index.items():
            color = colors[split]  
            patch = mpatches.Patch(color=color, label=f"{split}")
            patches.append(patch)

        ax.legend(
            handles=patches,
            title="Floor" if cluster=="class" else "Data split",
            title_fontsize=12,
            loc="upper right",
            fontsize=10,
            bbox_to_anchor=(1.2, 0.85)
        )
        plt.title(title, fontsize=16)

        if out_path != None:
            plt.savefig(out_path, format="pdf", bbox_inches="tight")


class TableVisualizer:
    """
    Gestiona la creació de taules a partir d'un Dataframe.
    """
    def __init__(self):
        pass

    def set_style(
        self, 
        styler: pd.io.formats, 
        title
    ) -> pd.io.formats:
        """
        Estableix l'estil visual de la taula.
        """
        
        styler.set_caption(title)
        styler.hide(axis="index")
        caption = {
            "selector": "caption",
            "props": [("color", "black"), ("font-size", "16px"),
                      ("text-align", "left"), ("font-family", "Times New Roman"),
                      ("width", "750px")]
        }
        headers = {
            "selector": "th:not(.index_name)",
            "props": [("font-family", "Times New Roman"), ("font-size", "16px"),
                       ("text-align", "left")]
        }
        cells = {
            "selector": "td",
            "props": [("font-family", "Times New Roman"), ("font-size", "16px"),
                       ("text-align", "left")]
        }
        styler.set_table_styles([caption, headers, cells])
        return styler
    

def plot_2d_sample_distribution(dataset, title, out_path):
    """
    Genera el gràfic de la distribució bidimensional de les mostres.
    """
    sns.set(style="whitegrid")
    
    fig, ax = plt.subplots(figsize=(6, 5))

    #Train
    sns.scatterplot(
        x=dataset.train.y['LONGITUDE'],
        y=dataset.train.y['LATITUDE'],
        ax=ax,
        color='#1f77b4',
        marker='o',
        s=50,          
        edgecolor='k',
        linewidth=0.4,
        alpha=0.6,
        label='Train'
    )

    #Test
    sns.scatterplot(
        x=dataset.test.y['LONGITUDE'],
        y=dataset.test.y['LATITUDE'],
        ax=ax,
        color="#f7a259",
        marker='^',
        s=50,
        edgecolor='k',
        linewidth=0.4,
        alpha=0.6,
        label='Test'
    )

    ax.set_xlabel('Longitud', fontsize=10)
    ax.set_ylabel('Latitud', fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_title(title)

    ax.legend(loc='upper left', bbox_to_anchor=(0.6, 1.15), fontsize=12, markerscale=1, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.show()

def plot_3d_sample_distribution(dataset, elev, azim, out_path):
    """
    Genera el gràfic de la distribució tridimensional de les mostres.
    """
    xs_train = dataset.train.y['LONGITUDE']
    ys_train = dataset.train.y['LATITUDE']
    zs_train = dataset.train.y['FLOORID']

    xs_test = dataset.test.y['LONGITUDE']
    ys_test = dataset.test.y['LATITUDE']
    zs_test = dataset.test.y['FLOORID']

    fig = go.Figure()

    #Train
    fig.add_trace(go.Scatter3d(
        x=xs_train,
        y=ys_train,
        z=zs_train,
        mode='markers',
        name='Train',
        marker=dict(
            color='#1f77b4',
            size=2,
            symbol='circle',
            opacity=0.6
        )
    ))

    #Test 
    fig.add_trace(go.Scatter3d(
        x=xs_test,
        y=ys_test,
        z=zs_test,
        mode='markers',
        name='Test',
        marker=dict(
            color='#ff7f0e',
            size=2,
            symbol='diamond',
            opacity=0.6
        )
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Longitud",
            yaxis_title="Latitud",
            zaxis_title="Planta",
            camera=dict(
                eye=dict(
                    x=np.cos(np.radians(azim)),
                    y=np.sin(np.radians(azim)),
                    z=np.sin(np.radians(elev))
                )
            ),
        ),
        legend=dict(
            orientation="h",
            xanchor="center",
            x=0.5,
            y=1.05
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.write_html(out_path)
    fig.show()

def plot_rss_distribution(dataset, title, out_path):
    """
    Genera el gràfic de la distribució de valors RSSI.
    """

    X_train_stack = dataset.train.x.replace(to_replace=[-105, 100], value=np.nan).stack()
    X_test_stack = dataset.test.x.replace(to_replace=[-105, 100], value=np.nan).stack()

    plt.figure(figsize=(5, 4))
    sns.histplot(X_train_stack.dropna(), binwidth=2, stat='probability',
                 color='steelblue', alpha=0.5, label='Train')
    sns.histplot(X_test_stack.dropna(), binwidth=2, stat='probability',
                 color='orange', alpha=0.5, label='Test',
                 binrange=(X_train_stack.min(), X_train_stack.max()))

    plt.xlabel('Valor RSSI (dBm)')
    plt.ylabel('Probabilitat')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.show()

def plot_detected_waps_per_sample(dataset, title, out_path):
    """
    Genera el gràfic de la distribució de WAPs detectats en una sola captura.
    """

    X_train = dataset.train.x.replace(to_replace=[-105, 100], value=np.nan)
    X_test = dataset.test.x.replace(to_replace=[-105, 100], value=np.nan)
    train_detected = X_train.notna().sum(axis=1)
    test_detected = X_test.notna().sum(axis=1)
    
    total_waps = X_train.shape[1]  
    
    train_mode = train_detected.mode()[0]  
    test_mode = test_detected.mode()[0]
    
    train_min = train_detected.min()
    train_max = train_detected.max()
    test_min = test_detected.min()
    test_max = test_detected.max()
    
    train_mode_pct = (train_mode / total_waps) * 100
    test_mode_pct = (test_mode / total_waps) * 100
    train_max_pct = (train_max / total_waps) * 100
    test_max_pct = (test_max / total_waps) * 100
    
    print(f"\n=== {title} ===")
    print(f"Total WAPs: {total_waps}")
    print(f"\nTrain:")
    print(f"  Límit inferior: {train_min}")
    print(f"  Punt màxim (moda): {train_mode} ({train_mode_pct:.2f}% del total)")
    print(f"  Límit superior: {train_max} ({train_max_pct:.2f}% del total)")
    print(f"\nTest:")
    print(f"  Límit inferior: {test_min}")
    print(f"  Punt màxim (moda): {test_mode} ({test_mode_pct:.2f}% del total)")
    print(f"  Límit superior: {test_max} ({test_max_pct:.2f}% del total)")
    
    plt.figure(figsize=(5, 4))
    sns.histplot(train_detected, binwidth=2, color='steelblue', alpha=0.6, label='Train', stat='probability')
    sns.histplot(test_detected, binwidth=2, color='orange', alpha=0.6, label='Test', stat='probability')
    
    plt.axvline(train_mode, color='steelblue', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.axvline(test_mode, color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
    
    plt.xlabel('Nombre de WAPs detectats')
    plt.ylabel('Probabilitat')
    plt.legend()
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.show()
    
    return {
        'total_waps': total_waps,
        'train': {
            'min': train_min,
            'mode': train_mode,
            'mode_pct': train_mode_pct,
            'max': train_max,
            'max_pct': train_max_pct
        },
        'test': {
            'min': test_min,
            'mode': test_mode,
            'mode_pct': test_mode_pct,
            'max': test_max,
            'max_pct': test_max_pct
        }
    }

def plot_samples_by_floor(dataset, title, out_path):
    """
    Genera el gràfic de la freqüència de mostres per planta.
    """
        
    train_meta = dataset.train.y.copy()
    test_meta = dataset.test.y.copy()
    
    train_meta["Set"] = "Train"
    test_meta["Set"] = "Test"
    meta = pd.concat([train_meta, test_meta], axis=0)

    counts = meta.groupby(["BUILDING_FLOOR", "Set"]).size().reset_index(name="Nombre_de_mostres")
    
    floors = sorted(counts["BUILDING_FLOOR"].unique())
    x = np.arange(len(floors))
    
    train_counts = counts[counts["Set"] == "Train"].set_index("BUILDING_FLOOR")["Nombre_de_mostres"]
    test_counts  = counts[counts["Set"] == "Test"].set_index("BUILDING_FLOOR")["Nombre_de_mostres"]

    train_counts = train_counts.reindex(floors, fill_value=0)
    test_counts  = test_counts.reindex(floors, fill_value=0)

    bar_width = 0.6
    if title in ['SOD02', 'SOD06']:
        bar_width = 0.2

    x_train = x - bar_width/4
    x_test  = x + bar_width/4

    plt.figure(figsize=(5,4))

    # Train
    plt.bar(x_train, train_counts, 
            width=bar_width/2, 
            color="steelblue", 
            edgecolor="black", 
            label="Train", 
            alpha=0.8)

    # Test
    plt.bar(x_test, test_counts, 
            width=bar_width/2, 
            color="orange", 
            edgecolor="black", 
            label="Test", 
            alpha=0.7)

    plt.xticks(x, floors)
    plt.xlabel("Planta")
    plt.ylabel("Nombre de mostres")
    plt.title(title)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.show()

def plot_cdf(errors, dataset, save_path=None):
    """
    Genera el gràfic de la funció de distribució acumulada.
    """
    plt.figure(figsize=(7, 5))
    
    for key, info in errors.items():
        errors = np.load(info['path'])
        errors = np.abs(errors)
        mean_pos_error = errors.mean(axis=1)
        errors_sorted = np.sort(mean_pos_error)
        cdf = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
        plt.plot(errors_sorted, cdf, label=info['label'])
    
    plt.xlabel("Error de posicionament mitjà (m)")
    plt.ylabel("Probabilitat")
    #plt.yscale("log")
    plt.xscale("log")
    plt.title(dataset)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, format="svg")
    
    plt.show()

def plot_building_floor_distribution(df, save_path=None):
    """
    Genera el gràfic amb la distribució de mostres de les dues primeres plantes (TIE1).
    """
    pastel_colors = [
        sns.color_palette("Set2")[0], 
        sns.color_palette("Set2")[1],  
        sns.color_palette("Set2")[2]  
    ]
    
    train_df = df.train.y
    test_df = df.test.y
    
    train_class0 = train_df[train_df['BUILDING_FLOOR'] == 0]
    test_class0 = test_df[test_df['BUILDING_FLOOR'] == 0]
    train_class1 = train_df[train_df['BUILDING_FLOOR'] == 1]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(train_class1['LONGITUDE'], train_class1['LATITUDE'], 
                color=pastel_colors[0], label='Train - Planta 1', alpha=0.6)
    plt.scatter(train_class0['LONGITUDE'], train_class0['LATITUDE'], 
                color=pastel_colors[2], label='Train - Planta 0', alpha=1.)
    plt.scatter(test_class0['LONGITUDE'], test_class0['LATITUDE'], 
                color=pastel_colors[1], label='Test - Planta 0', alpha=1.)
    
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    
    plt.show()

def plot_confusion_matrix(cm, dataset, model, save_path=None):
    """
    Mostra la representació gràfica de la matriu de confusió.
    """

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    )
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        cbar=True,
        vmin=0,
        vmax=1
    )
    plt.xlabel("Predicció")
    plt.ylabel("Classe")
    plt.title(f"{dataset} - {model}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format="svg")
    
    plt.show()

def plot_time_correlation(df, save_path=None):
    """
    Genera el gràfic de la correlació entre temps d'entrenament i inferèrncia.
    """
    x_col = 't (test)'
    y_col = 't (train)'
    models = ['GCN', 'GAT', 'SAGE']
    colors = {
        'GCN': sns.color_palette("Set2")[0],
        'GAT': sns.color_palette("Set2")[1],
        'SAGE': sns.color_palette("Set2")[2]
    }
    markers = {
        'GCN': 'o',
        'GAT': 's',
        'SAGE': 'D'
    }
    
    plt.figure(figsize=(8, 6))
    
    for model in models:
        subset = df[df['Model'] == model]
        plt.scatter(
            subset[x_col],
            subset[y_col],
            label=model,
            marker=markers[model],
            color=colors[model],
            alpha=1,
            s=40
        )
    
    ax = sns.regplot(
        x=x_col,
        y=y_col,
        data=df,
        scatter=False,
        ci=95,
        color='black',
        line_kws={'lw': 1.0, 'ls': '--'}
    )
    
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            collection.set_alpha(0.35)
            collection.set_facecolor('lightgray')
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Temps d'inferència (s)")
    plt.ylabel("Temps d'entrenament (s)")
    plt.legend(title='Model', loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format="svg")
    
    plt.show()
    
    x = df[x_col]
    y = df[y_col]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    R2 = r_value**2
    print(f"R² = {R2:.3f}")
    
    return R2

def plot_partial_effects(df, save_path=None):
    """
    Genera els gràfic de les regressions per variable independent i arquitectura.
    """
    variables = ['Característiques', 'Nodes', 'Enllaços', 'Paràmetres entrenables']
    model_col='Model'
    y_col = 't (test)'
    models=['GCN', 'GAT', 'SAGE']   

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    pastel_colors = {
        'GCN': sns.color_palette("Set2")[0],
        'GAT': sns.color_palette("Set2")[1],
        'SAGE': sns.color_palette("Set2")[2]
    }
    
    markers = {
        'GCN': 'o',
        'GAT': 's',
        'SAGE': 'D'
    }
    
    for ax, var in zip(axes, variables):
        for model_name in models:
            df_m = df[df[model_col] == model_name]
            y = df_m[y_col]
            X = df_m[variables]
            X_rest = sm.add_constant(X.drop(columns=[var]))
            y_resid = sm.OLS(y, X_rest).fit().resid
            x_var = X[var]
            x_resid = sm.OLS(x_var, X_rest).fit().resid
            
            sns.regplot(
                x=x_resid,
                y=y_resid,
                ax=ax,
                marker=markers[model_name],
                scatter=True,
                ci=None,
                color=pastel_colors[model_name],
                scatter_kws={
                    's': 35,
                    'alpha': 1
                },
                line_kws={
                    'ls': '--',
                    'lw': 1.5
                },
                label=model_name
            )
        
        ax.set_title(f'Efecte parcial: {var}')
        ax.set_xlabel(f'{var} (residual)')
        ax.set_ylabel(r'$t_{train}$ (residual)')
        ax.grid(True, linestyle='--', alpha=0.5)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, format="svg")
    
    plt.show()

def plot_beta_coefficients(df, save_path=None):
    """
    Calcula els coeficients beta de la regressió lineal múltiple (t_inferencia)
    i genera gràfic de barres agrupant per variable independent i arquitectura.
    """
    betas = []
    variables = ['Característiques', 'Nodes', 'Enllaços', 'Paràmetres entrenables']
    model_col='Model'
    y_col = 't (test)'
    models=['GCN', 'GAT', 'SAGE']  

    for model in models:
        df_m = df[df[model_col] == model]
        
        X = df_m[variables]
        y = df_m[y_col]
        
        Xs = StandardScaler().fit_transform(X)
        ys = StandardScaler().fit_transform(y.values.reshape(-1, 1)).ravel()
        
        Xs = sm.add_constant(Xs)
        ols = sm.OLS(ys, Xs).fit()
        
        for var, coef in zip(X.columns, ols.params[1:]):
            betas.append([model, var, coef])
    
    df_beta = pd.DataFrame(betas, columns=['Model', 'Variable', 'Beta'])
    
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=df_beta,
        x='Variable',
        y='Beta',
        hue='Model',
        palette='Set2',
        alpha=0.8
    )
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('Coeficient de regressió')
    plt.xlabel('')
    plt.legend(title='Model')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format="svg")
    
    plt.show()

def plot_accuracy_by_k(ks, datasets, save_path=None):
    """
    Genera el gràfic de la exactitud en funció del paràmetre k.
    """
    colors = {
        'GCN': sns.color_palette("Set2")[0],
        'GAT': sns.color_palette("Set2")[1],
        'SAGE': sns.color_palette("Set2")[2]
    }

    plt.figure(figsize=(7, 6))
    plt.plot(ks, datasets[0]['mean_accuracy'], marker='o', color=colors['GCN'], label='GCN')
    plt.plot(ks, datasets[1]['mean_accuracy'], marker='o', color=colors['GAT'], label='GAT')
    plt.plot(ks, datasets[2]['mean_accuracy'], marker='o', color=colors['SAGE'], label='GraphSAGE')
    
    plt.xlabel("k")
    plt.ylabel("Exactitud (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    
    plt.show()

def plot_time_by_k(ks, datasets, save_path=None):
    """
    Genera el gràfic del temps d'entrenament i inferència en funció del paràmetre k.
    """
    colors = {
        'GCN': sns.color_palette("Set2")[0],
        'GAT': sns.color_palette("Set2")[1],
        'SAGE': sns.color_palette("Set2")[2]
    }
    
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    
    axes[0].plot(ks, datasets[0]["mean_train_time"], marker='o', color=colors['GCN'], label="GCN")
    axes[0].plot(ks, datasets[1]["mean_train_time"], marker='o', color=colors['GAT'], label="GAT")
    axes[0].plot(ks, datasets[2]["mean_train_time"], marker='o', color=colors['SAGE'], label="GraphSAGE")
    axes[0].set_ylabel("Temps d'entrenament (s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(ks, datasets[0]["mean_test_time"], marker='o', color=colors['GCN'], label="GCN")
    axes[1].plot(ks, datasets[1]["mean_test_time"], marker='o', color=colors['GAT'], label="GAT")
    axes[1].plot(ks, datasets[2]["mean_test_time"], marker='o', color=colors['SAGE'], label="GraphSAGE")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Temps d'inferència (s)")
    axes[1].grid(True, alpha=0.3)
    
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, -3))
    axes[1].yaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    
    plt.show()

def plot_computational_scaling(df, save_path=None):
    """
    Genera el gràfic amb la regressió lineal del temps d'inferencia en funció de N*F,
    per a cada arquitectura, i també la recta teòrica del cost computacional per k-NN.
    """
    
    model_col='Model'
    models=['GCN', 'GAT', 'SAGE'] 
    
    plt.figure(figsize=(11, 7))
    
    colors = {
        'GCN': sns.color_palette("Set2")[0],
        'GAT': sns.color_palette("Set2")[1],
        'SAGE': sns.color_palette("Set2")[2]
    }
    
    pastel_colors = {
        'GCN': sns.color_palette("Set2")[0],
        'GAT': sns.color_palette("Set2")[1],
        'SAGE': sns.color_palette("Set2")[2]
    }
    
    markers = {
        'GCN': 'o',
        'GAT': 's',
        'SAGE': 'D'
    }
    
    results = []
    
    for model in models:
        df_m = df[df[model_col] == model]
        
        complexity = df_m['Nodes'] * df_m['Característiques']
        time = df_m['t (test)']
        
        slope, intercept, r_value, _, _ = stats.linregress(
            np.log10(complexity),
            np.log10(time)
        )
        
        results.append({
            'Model': model,
            'Exponent': slope,
            'R²': r_value**2
        })
        
        plt.scatter(
            complexity, time,
            color=colors[model],
            label=model,
            marker=markers[model],
            alpha=0.7,
            s=100,
            zorder=3
        )
        
        x_fit = np.logspace(
            np.log10(complexity.min()),
            np.log10(complexity.max()),
            100
        )
        y_fit = 10**intercept * x_fit**slope
        
        plt.plot(
            x_fit, y_fit,
            color=pastel_colors[model],
            linestyle='--',
            linewidth=2.5,
            alpha=0.7,
            zorder=2
        )
    
    complexity_range = np.logspace(
        np.log10((df['Nodes'] * df['Característiques']).min()),
        np.log10((df['Nodes'] * df['Característiques']).max()),
        100
    )
    
    mean_time = df['t (test)'].mean()
    mean_degree = (df['Característiques'] / df['Nodes']).mean()
    k_knn = mean_time / (complexity_range.mean() / mean_degree)
    time_knn = k_knn * (complexity_range / mean_degree)
    
    plt.plot(
        complexity_range, time_knn,
        color='black',
        linestyle='-',
        lw=2.5,
        label='k-NN O(N·F)',
        alpha=0.6,
        zorder=1
    )
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$N \cdot F$', fontsize=13)
    plt.ylabel(r"Temps d'inferència (s)", fontsize=13)
    plt.legend(fontsize=9, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format="svg")
    
    plt.show()
    
    return pd.DataFrame(results)

def count_param(configs, ids, param):
    return Counter([configs[i][param] for i in ids])

def grouped_bars(ax, reg, clf, title):
    """
    Genera gràfics de barres per la distribució dels paràmetres òptims.
    """
    width = 0.35

    labels = sorted(set(reg.keys()) | set(clf.keys()))
    x = np.arange(len(labels))
    ax.bar(x - width/2,
            [reg.get(l, 0) for l in labels],
            width,
            label='Regressió')
    ax.bar(x + width/2,
            [clf.get(l, 0) for l in labels],
            width,
            label='Classificació')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
