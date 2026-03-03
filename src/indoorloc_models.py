import torch
import torch.nn as nn
import torch.optim as optimizer
import torch_geometric
from torch_geometric.nn import (
    Sequential, 
    GraphNorm,
    GCNConv,
    GATConv,
    SAGEConv,
    MLP 
)

class GCNRegressor(nn.Module):
    """
    GCN model for regression tasks
    """
    def __init__(
        self,
        input_dim: int, 
        n_layers: int,
        hidden_dim: int, 
        output_dim: int,
        dropout: float, 
        learning_rate: float,
        optim_factor: float,
        weight_decay: float,
        mlp_layers: int
    ) -> None:
        super().__init__()

        layers = []

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim] * n_layers
        else:
            hidden_dims = hidden_dim[:n_layers]

        if isinstance(dropout, (float, int)):
            dropouts = [dropout] * n_layers
        else:
            dropouts = dropout[:n_layers]
        
        current_dim = input_dim

        for i in range(n_layers):
            layers.append((GCNConv(
                current_dim, 
                hidden_dims[i]
            ), "x, edge_index -> x"))

            layers.append(GraphNorm(hidden_dims[i])) 
            layers.append(nn.LeakyReLU())

            if i < n_layers - 1:  
                layers.append(nn.Dropout(p=dropouts[i]))
        
            current_dim = hidden_dims[i]

        mlp_layers_dims = [current_dim] + [current_dim // (2 ** (i + 1)) \
                                        for i in range(mlp_layers)] + [output_dim]

        layers.append(MLP(mlp_layers_dims))

        self.layers = Sequential("x, edge_index", layers) 
        
        self.criterion = nn.L1Loss()
        self.optimizer = optimizer.Adam(
            params=self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        
        self.scheduler = optimizer.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=optim_factor, patience=10
        )

    def get_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        return self.layers(data.x, data.edge_index,)
    

class GATRegressor(nn.Module):
    """
    GAT model for regression tasks.
    """
    def __init__(
        self,
        input_dim: int, 
        n_layers: int,
        hidden_dim: int, 
        output_dim: int,
        heads: int,
        dropout: float, 
        learning_rate: float,
        optim_factor: float,
        weight_decay: float,
        mlp_layers: int
    ) -> None:
        super().__init__()

        layers = []

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim] * n_layers
        else:
            hidden_dims = hidden_dim[:n_layers]

        if isinstance(dropout, (float, int)):
            dropouts = [dropout] * n_layers
        else:
            dropouts = dropout[:n_layers]
        
        current_dim = input_dim

        for i in range(n_layers):
            concat = i < n_layers - 1 
            layers.append((GATConv(
                current_dim, hidden_dims[i], heads=heads[i], concat=concat,
            fill_value="mean"), "x, edge_index -> x"))

            layers.append(GraphNorm(hidden_dims[i] * heads[i] if concat else hidden_dims[i])) 
            layers.append(nn.LeakyReLU())

            if i < n_layers - 1:  
                layers.append(nn.Dropout(p=dropouts[i]))
        
            current_dim = hidden_dims[i] * heads[i] if concat else hidden_dims[i]

        mlp_layers_dims = [current_dim] + [current_dim // (2 ** (i + 1)) \
                                        for i in range(mlp_layers)] + [output_dim]

        layers.append(MLP(mlp_layers_dims))

        self.layers = Sequential("x, edge_index", layers) 
        
        self.criterion = nn.L1Loss()
        self.optimizer = optimizer.Adam(
            params=self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        
        self.scheduler = optimizer.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=optim_factor, patience=10
        )
        
    def get_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        return self.layers(data.x, data.edge_index)
    

class SAGERegressor(nn.Module):
    """
    GraphSAGE model for regression tasks.
    """
    def __init__(
        self, 
        input_dim: int, 
        n_layers: int,
        hidden_dim: int, 
        output_dim: int,
        dropout: float, 
        learning_rate: float,
        optim_factor: float,
        weight_decay: float,
        mlp_layers: int
    ) -> None:
        super().__init__()
        
        layers = []
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim] * n_layers
        else:
            hidden_dims = hidden_dim[:n_layers]
        
        if isinstance(dropout, (float, int)):
            dropouts = [dropout] * n_layers
        else:
            dropouts = dropout[:n_layers]

        current_dim = input_dim
        for i in range(n_layers):
            layers.append((SAGEConv(current_dim, hidden_dims[i], aggr="mean"),
                            "x, edge_index -> x"))
            layers.append(GraphNorm(hidden_dims[i]))  
            layers.append(nn.LeakyReLU())

            if i < n_layers - 1:  
                layers.append(nn.Dropout(p=dropouts[i]))
            current_dim = hidden_dims[i]
        
        mlp_layers_dims = [current_dim] + [current_dim // (2 ** (i + 1)) \
                                        for i in range(mlp_layers)] + [output_dim]

        layers.append(MLP(mlp_layers_dims))

        self.layers = Sequential("x, edge_index", layers) 

        self.criterion = nn.L1Loss()
        self.optimizer = optimizer.Adam(
            params=self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        
        self.scheduler = optimizer.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=optim_factor, patience=10
        )

    def get_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        return self.layers(data.x, data.edge_index)
    

class SAGEClassifier(nn.Module):
    """
    GraphSAGE model for classification tasks.
    """
    def __init__(
        self, 
        input_dim: int, 
        n_layers: int,
        hidden_dim: list, 
        output_dim: int,
        dropout: list, 
        learning_rate: float,
        optim_factor: float,
        weight_decay: float,
        mlp_layers: int
    ) -> None:
        super().__init__()
        
        layers = []

        current_dim = input_dim
        for i in range(n_layers):
            layers.append((SAGEConv(current_dim, hidden_dim[i]),
                            "x, edge_index -> x"))
            
            layers.append(GraphNorm(hidden_dim[i]))  
            layers.append(nn.LeakyReLU())

            if i < n_layers - 1:  
                layers.append(nn.Dropout(p=dropout[i]))
            current_dim = hidden_dim[i]
        
        mlp_layers_dims = [current_dim] + [current_dim // (2 ** (i + 1)) \
                                        for i in range(mlp_layers)] + [output_dim]

        layers.append(MLP(mlp_layers_dims))

        self.layers = Sequential("x, edge_index", layers) 

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer.Adam(
            params=self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = optimizer.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=optim_factor, patience=10
        )

    def get_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        return self.layers(data.x, data.edge_index)