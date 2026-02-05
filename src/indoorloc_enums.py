from enum import Enum

class Devices(Enum):
    cpu = "cpu"
    cuda = "cuda"

class Tasks(Enum):
    classification = "classification"
    regression = "regression"

class Datasets(Enum):
    ujiindoorloc = "ujiindoorloc"
    sodindoorloc = "sodindoorloc"
    utsindoorloc = "utsindoorloc"

class Subsets(Enum):
    train = "train"
    validation = "validation"
    test = "test"

class Loaders(Enum):
    train = "loader_train"
    validation = "loader_val"
    test = "loader_test"

class Targets(Enum):
    building_floor = "BUILDING_FLOOR"
    floor = "FLOORID"
    building = "BUILDINGID"
    longitude = "LONGITUDE"
    latitude = "LATITUDE"

class Metrics(Enum):
    loss = "loss"
    accuracy = "accuracy"
    mean_accuracy = "mean_accuracy"
    accuracy_std = "accuracy_std"
    max_accuracy = "max_accuracy"
    precission_macro = "precision_macro"
    f1_macro = "f1_macro"
    recall_macro = "recall_macro"
    confusion_matrix = "confusion_matrix"
    mean_train_time = "mean_train_time"
    mean_train_time_std = "mean_train_time_std"
    mean_test_time = "mean_test_time"
    mean_test_time_std = "mean_test_time_std"
    mpe = "mpe"
    mpe_std = "mpe_std"
    mae = "mae"
    mae_std = "mae_std"
    mae_x = "mae_x"
    mae_x_std = "mae_x_std"
    mae_y = "mae_y"
    mae_y_std = "mae_y_std"
    mse = "mse"
    rmse = "rmse"
    elapsed_time = "elapsed_time"
    mean_elapsed_time = "mean_elapsed_time"
    num_tests = "num_tests"
    max_epochs = "max_epochs"
    patience = "patience"
    output_errors = "output_errors"

class Parameters(Enum):
    parameters = "parameters"
    best_parameters = "best_parameters"
    grid_parameters = "grid_parameters"
    n_parameters = "n_parameters"
    n_neighbors = "n_neighbors"

class Preprocessing(Enum):
    scaler = "scaler"

class Networks(Enum):
    nn = "nn"
    gnn = "gnn"

class PlotLabels(Enum):
    score = "Score"
    epoch = "Epoch"
