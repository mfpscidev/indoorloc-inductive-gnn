import copy
from typing import Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

import indoorloc_enums as ilenums

# Constants
SEED = 42

# Enums
DEVICES_CUDA = ilenums.Devices.cuda.value
DEVICES_CPU = ilenums.Devices.cpu.value
TASKS_CLS = ilenums.Tasks.classification.value
TASKS_REG = ilenums.Tasks.regression.value
SUBSETS_TRAIN = ilenums.Subsets.train.value
SUBSETS_TEST = ilenums.Subsets.test.value
TARGETS_BUILDING_FLOOR = ilenums.Targets.building_floor.value
TARGETS_BUILDING = ilenums.Targets.building.value
TARGETS_FLOOR = ilenums.Targets.floor.value
TARGETS_LONGITUDE = ilenums.Targets.longitude.value
TARGETS_LATITUDE = ilenums.Targets.latitude.value
DATASETS_UJIINDOORLOC = ilenums.Datasets.ujiindoorloc.value
DATASETS_SODINDOORLOC = ilenums.Datasets.sodindoorloc.value

@dataclass
class Train:
    x: pd.DataFrame
    y: pd.DataFrame


@dataclass
class Test:
    x: pd.DataFrame
    y: pd.DataFrame


@dataclass
class Classification:
    dataset_name: str
    floorid: dict = field(default_factory=dict, init=False)
    buildingid: dict = field(default_factory=dict)
    building_floor: dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.floorid = self.floorid or {}
        self.buildingid = self.buildingid or {}
        self.building_floor = self.building_floor or {}

        if self.dataset_name == DATASETS_SODINDOORLOC:
            del self.buildingid
            del self.building_floor 
                    

@dataclass
class Regression:
    dataset_name: str
    allbuildings: dict = None
    building: dict = None

    def __post_init__(self):
        self.building = self.building or {}
        if self.dataset_name == DATASETS_UJIINDOORLOC:
            self.allbuildings = self.allbuildings or {}


class IndoorLocResults:
    """
    Agrupa els resultats de classificació i regressió.
    """
    def __init__(self, dataset_name):
        self.cls = Classification(dataset_name)
        self.reg = Regression(dataset_name)


class IndoorLocGraphDataLoader:
    """
    Agrupa objectes Data dels grafs per classificació i regressió.
    """
    cls: dict = None
    reg: dict = None
    def __post_init__(self):
        self.cls = self.cls or {}
        self.reg = self.reg or {}


class IndoorLocDataset:
    """
    Gestiona la carrega de dades.
    """
    def __init__(self, dataset_structure, path, header):
        self.target = None
        
        self.train = Train(
            self._load_data(path + "_trnrss.csv", header),
            self._load_data(path + "_trncrd.csv", header)
        )
        self.test = Test(
            self._load_data(path + "_tstrss.csv", header),
            self._load_data(path + "_tstcrd.csv", header)
        )

        self.train = self._add_headers(self.train, dataset_structure)
        self.test = self._add_headers(self.test, dataset_structure)

        self.features = self.train.x.columns
        self.buildings = self._get_buildings(self.train)

    def _load_data(self, path, header):
        """
        Carrega els fitxers de dades.
        """
        return pd.read_csv(filepath_or_buffer=path, header=header)
    
    def _add_headers(self, dataset, dataset_structure):
        """
        Assigna nom a les columnes del Datafrane
        """
        if dataset_structure == DATASETS_UJIINDOORLOC:
            dataset.y.drop(2, axis=1, inplace=True)
        if dataset_structure == DATASETS_SODINDOORLOC:
            dataset.y.drop(3, axis=1, inplace=True)

        dataset.x.columns = [f"WAP{id+1}" for id in range(len(dataset.x.columns))]
        dataset.y.columns = [TARGETS_LONGITUDE, TARGETS_LATITUDE, 
                             TARGETS_FLOOR, TARGETS_BUILDING]
        return dataset
    
    def _get_buildings(self, dataset):
        """
        Retorna identificador dels edificis d'un conjunt de dades.
        """
        return sorted(dataset.y[TARGETS_BUILDING].unique().tolist())


class IndoorLocPreprocessor:
    """
    Gestiona el preprocessament de les dades.
    """
    def preprocess_dataset(
        self, 
        data: IndoorLocDataset,  
        relace_missing_signals: bool = True,
        codification: bool = True,
        drop_unused_columns: bool = True,
        normalization: str = "lineal",
        pca_components: float = 0.9,
    ) -> Union[Train, Test]:
        
        if relace_missing_signals:
            data.train.x = self._replace_missing_signals(data.train.x)
            data.test.x = self._replace_missing_signals(data.test.x)
        
        if codification:
            data.train.y = self._create_target(data.train.y)
            data.test.y = self._create_target(data.test.y)

        if drop_unused_columns:
            data.train.y.drop(columns=["FLOORID", "BUILDINGID"], axis=1, inplace=True)
            data.test.y.drop(columns=["FLOORID", "BUILDINGID"], axis=1, inplace=True)

        if normalization == "lineal":
            data.train.x, data.test.x = self._normalize_zero_to_one(data.train.x, data.test.x)
        
        if normalization == "exponential":
            data.train.x, data.test.x = self._normalize_exponential(data.train.x, data.test.x)

        if normalization == "powed":
            data.train.x, data.test.x = self._normalize_power(data.train.x, data.test.x)
        
        if 0 < pca_components < 1:
            data = self._apply_pca(data, n_components=pca_components)
        
        return data

    def filter_building(self, data, target, building_id):
        """
        Filtra les dades corresponents a un edifici del Dataframe.
        """
        building_data = copy.deepcopy(data)

        train_x = building_data.train.x
        train_y = building_data.train.y
        test_x = building_data.test.x
        test_y = building_data.test.y

        idxs_trn = train_y[train_y[target] == building_id].index
        idxs_tst = test_y[test_y[target] == building_id].index

        building_data.train.x = train_x.loc[idxs_trn]
        building_data.test.x = test_x.loc[idxs_tst]
        building_data.train.y = train_y.loc[idxs_trn]
        building_data.test.y = test_y.loc[idxs_tst]

        return building_data

    def _apply_pca(self, data, n_components=0.95):
        """
        Aplica reducció de dimensionalitat amb la tècnica PCA.
        """
        pca = PCA(n_components=n_components)

        data.train.x = pd.DataFrame(
            pca.fit_transform(data.train.x),
            columns=[f"WAP{i+1}" for i in range(pca.n_components_)],
            index=data.train.x.index
        )
        
        data.test.x = pd.DataFrame(
            pca.transform(data.test.x),
            columns=[f"WAP{i+1}" for i in range(pca.n_components_)],
            index=data.test.x.index
        )
        data.features = data.train.x.columns

        return data

    def _replace_missing_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reemplaça el valor dels senyals dèbils o absents.
        """
        return df.where(df != 100, -105)

    def _normalize_zero_to_one(
            self, train_x: pd.DataFrame, test_x: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aplica normalització amb tranformació lineal.
        """
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_x)
        test_scaled = scaler.transform(test_x)
        
        train_df = pd.DataFrame(train_scaled, columns=train_x.columns, index=train_x.index)
        test_df = pd.DataFrame(test_scaled, columns=test_x.columns, index=test_x.index)
        
        return train_df, test_df
    
    def _normalize_exponential(
            self, train_x: pd.DataFrame, test_x: pd.DataFrame, alpha: float = np.e
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aplica normalització amb tranformació exponencial.
        """
        train_min = train_x.min().min()
        train_max = train_x.max().max()
        
        train_normalized = (train_x - train_min) / (train_max - train_min)
        test_normalized = (test_x - train_min) / (train_max - train_min)
        
        train_exp = np.power(alpha, train_normalized)
        test_exp = np.power(alpha, test_normalized)
        
        train_df = pd.DataFrame(train_exp, columns=train_x.columns, index=train_x.index)
        test_df = pd.DataFrame(test_exp, columns=test_x.columns, index=test_x.index)
        
        return train_df, test_df

    def _normalize_power(
            self, train_x: pd.DataFrame, test_x: pd.DataFrame, alpha: float = np.e
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aplica normalització amb tranformació de potència.
        """
        train_min = train_x.min().min()
        train_max = train_x.max().max()
        
        train_normalized = (train_x - train_min) / (train_max - train_min)
        test_normalized = (test_x - train_min) / (train_max - train_min)
        
        train_pow = np.power(train_normalized, alpha)
        test_pow = np.power(test_normalized, alpha)
        
        train_df = pd.DataFrame(train_pow, columns=train_x.columns, index=train_x.index)
        test_df = pd.DataFrame(test_pow, columns=test_x.columns, index=test_x.index)
        
        return train_df, test_df

    def _create_target(self, y: pd.DataFrame) -> pd.DataFrame:
        """
        Codifica variable building_floor
        """
        building_floor_pairs = list(zip(y[TARGETS_BUILDING], y[TARGETS_FLOOR]))
        y[TARGETS_BUILDING_FLOOR] = pd.Categorical(building_floor_pairs).codes
        return y    
    

class IndoorLocGraphData:
    """
    Gestiona la construcció dels grafs.
    """
    def __init__(self):
        self.y_scaler = MinMaxScaler()
        self.device = torch.device(
                DEVICES_CUDA if torch.cuda.is_available() else DEVICES_CPU
            )
    
    def _assign_nodeid(
        self, 
        data: IndoorLocDataset
    ) -> IndoorLocDataset:
        """
        Assigna identificador únic a tots els nodes.
        """
        data_copy = copy.deepcopy(data)
        train_len = len(data.train.x)
        nodeid = "nodeid"

        subsets = [SUBSETS_TRAIN, SUBSETS_TEST]
        attributes = ["x", "y"]

        for subset in subsets:
            for attr in attributes:
                df = getattr(getattr(data_copy, subset), attr)
                df = df.reset_index(drop=True)

                if subset == SUBSETS_TRAIN:
                    df[nodeid] = df.index
                else:  
                    df[nodeid] = df.index + train_len
                
                setattr(getattr(data_copy, subset), attr, df)

        return data_copy
        
    def _create_tensor_mask(
        self, 
        data: pd.DataFrame, 
        tensor_dim: int
    ) -> torch.Tensor:
        """
        Crea les màscares per un subconjunt.
        """
        mask = torch.zeros(tensor_dim, dtype=torch.bool)
        mask[data["nodeid"].values] = True

        return mask

    def _build_knn_graph(self, gdata, k, metric='euclidean'):
        """
        Construeix el graf k-veïns pròxims.
        """
        if k is not None and k > 0:
            X = gdata.x.cpu().numpy()
            
            knn_graph = kneighbors_graph(
                X, 
                n_neighbors=k, 
                metric=metric,
                mode='connectivity',
                include_self=False
            )
            
            knn_graph_coo = knn_graph.tocoo()
            edge_index = torch.from_numpy(
                np.stack([knn_graph_coo.row, knn_graph_coo.col])
            ).long().to(self.device)
            
            edge_index = to_undirected(edge_index)

            gdata.edge_index = edge_index.to(self.device)
            gdata.k = k
    
        return gdata
    
    def create_nodes(
        self, 
        gdata,
        dataset: IndoorLocDataset, 
        val_size: float = 0.1,
        n_split: int = 0
    ) -> Data:
        """
        Crea el tensor de les caracteristiques o empremtes.
        """
        dataset = self._assign_nodeid(dataset)
        
        X_train, X_val, _, _ = train_test_split(
            dataset.train.x, dataset.train.y, test_size=val_size, random_state=SEED + n_split
        )

        gdata.num_nodes = sum(len(df) for df in [X_train, X_val, dataset.test.x])

        gdata.val_mask = self._create_tensor_mask(X_val, gdata.num_nodes)
        gdata.train_mask = self._create_tensor_mask(X_train, gdata.num_nodes)
        gdata.test_mask = self._create_tensor_mask(dataset.test.x, gdata.num_nodes)

        gdata.val_size = val_size

        x_concat = pd.concat([dataset.train.x, dataset.test.x])
        gdata.x = torch.tensor(x_concat[dataset.features].values, dtype=torch.float)

        gdata.num_features = len(dataset.features)
        
        return gdata.to(self.device)

    def create_edges(self, gdata, graph_params):
        """
        Crea el tensor dels enllaços generats a partir del graf k-veïns pròxims.
        """

        k = graph_params.get('k', 15)
        metric = graph_params.get('metric', 'manhattan')
        self._build_knn_graph(gdata, k, metric)

        return gdata.to(self.device) 
    
    def create_node_labels(self, dataset, gdata, task):
        """
        Crea el tensor de les variables objectiu.
        """
        if task == TASKS_REG:
            train_y = dataset.train.y

            train_coords = np.column_stack((
                train_y[TARGETS_LONGITUDE].values,
                train_y[TARGETS_LATITUDE].values
            ))

            self.y_scaler.fit(train_coords)

            y_concat = pd.concat([dataset.train.y, dataset.test.y])

            all_coords = np.column_stack((
                y_concat[TARGETS_LONGITUDE].values,
                y_concat[TARGETS_LATITUDE].values
            ))

            coords_scaled = self.y_scaler.transform(all_coords)  
            
            gdata.y = torch.tensor(coords_scaled, dtype=torch.float)
            gdata.y_scaler = self.y_scaler
            gdata.num_classes = 0

        if task == TASKS_CLS:
            y_concat = pd.concat([dataset.train.y, dataset.test.y])

            gdata.y = torch.tensor(y_concat[dataset.target].values, dtype=torch.int64)
            gdata.num_classes = len(y_concat[dataset.target].unique())

        return gdata.to(self.device)

    def create_dataloader(
        self, 
        dataset: IndoorLocDataset, 
        val_size: float,
        graph_params: dict,
        n_split: int = 0,
    ) -> IndoorLocGraphDataLoader:
        """
        Crea l'objecte Data amb l'estructura del graf 
        per les tasques de regressió i classificació.
        """
    
        gdataloader = IndoorLocGraphDataLoader()
        tasks = [TASKS_CLS, TASKS_REG]

        graph_data = Data()
        graph_data = self.create_nodes(graph_data, dataset, val_size, n_split)
        graph_data = self.create_edges(graph_data, graph_params)

        for task in tasks:
            if task == TASKS_CLS:
                dataset.target = TARGETS_BUILDING_FLOOR
                gdata_cls = self.create_node_labels(
                    dataset=dataset,
                    gdata=copy.deepcopy(graph_data),
                    task=task,
                )
                gdataloader.cls = gdata_cls

            if task == TASKS_REG:
                dataset.target = [TARGETS_LONGITUDE, TARGETS_LATITUDE] 
                gdata_reg = self.create_node_labels(
                    dataset=dataset, 
                    gdata=copy.deepcopy(graph_data),
                    task=task,
                )
                gdataloader.reg = gdata_reg

        return gdataloader
    
