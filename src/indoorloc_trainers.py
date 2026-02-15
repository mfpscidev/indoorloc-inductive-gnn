import os
import time
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric
import optuna

import indoorloc_vizs as ilvizs
import indoorloc_enums as ilenums

# Numerical constants
SEED = 42
INF = float("inf")

# Messages
SEPARATOR = "-" * 30
RUN_TESTS = "Running tests"
RUN = "Running"
SEARCH_MODELS = "Searching best model"

# Enums
DEVICES_CUDA = ilenums.Devices.cuda.value
DEVICES_CPU = ilenums.Devices.cpu.value
TASKS_CLS = ilenums.Tasks.classification.value
TASKS_REG = ilenums.Tasks.regression.value
SUBSETS_TRAIN = ilenums.Subsets.train.value
SUBSETS_VAL = ilenums.Subsets.validation.value
SUBSETS_TEST = ilenums.Subsets.test.value
LOADERS_TRAIN = ilenums.Loaders.train.value
LOADERS_VAL = ilenums.Loaders.validation.value
LOADERS_TEST = ilenums.Loaders.test.value
METRICS_LOSS = ilenums.Metrics.loss.value
METRICS_ACCU = ilenums.Metrics.accuracy.value
METRICS_ACCU_STD = ilenums.Metrics.accuracy_std.value
METRICS_MEAN_ACCU = ilenums.Metrics.mean_accuracy.value
METRICS_MAX_ACCU = ilenums.Metrics.max_accuracy.value
METRICS_PRECISSION_MACRO = ilenums.Metrics.precission_macro.value
METRICS_F1_MACRO = ilenums.Metrics.f1_macro.value
METRICS_RECALL_MACRO = ilenums.Metrics.recall_macro.value
CONFUSION_MATRIX = ilenums.Metrics.confusion_matrix.value
MEAN_TRAIN_TIME = ilenums.Metrics.mean_train_time.value
MEAN_TRAIN_TIME_STD = ilenums.Metrics.mean_train_time_std.value
MEAN_TEST_TIME = ilenums.Metrics.mean_test_time.value
MEAN_TEST_TIME_STD = ilenums.Metrics.mean_test_time_std.value
METRICS_MAE = ilenums.Metrics.mae.value
METRICS_MAE_STD = ilenums.Metrics.mae_std.value
METRICS_MAE_X = ilenums.Metrics.mae_x.value
METRICS_MAE_X_STD = ilenums.Metrics.mae_x_std.value
METRICS_MAE_Y = ilenums.Metrics.mae_y.value
METRICS_MAE_Y_STD = ilenums.Metrics.mae_y_std.value
METRICS_MPE = ilenums.Metrics.mpe.value
METRICS_MPE_STD = ilenums.Metrics.mpe_std.value
METRICS_MSE = ilenums.Metrics.mse.value
METRICS_RMSE = ilenums.Metrics.rmse.value
METRICS_MAX_EPOCHS = ilenums.Metrics.max_epochs.value
METRICS_PATIENCE = ilenums.Metrics.patience.value
METRICS_NUM_TESTS = ilenums.Metrics.num_tests.value
METRICS_ELAPSED_TIME = ilenums.Metrics.elapsed_time.value
METRICS_MEAN_ELAPSED_TIME = ilenums.Metrics.elapsed_time.value
METRICS_OUTPUT_ERRORS = ilenums.Metrics.output_errors.value
PARAMETERS_PARAMS = ilenums.Parameters.parameters.value
PARAMETERS_BEST_PARAMS = ilenums.Parameters.best_parameters.value
PARAMETERS_GRID_PARAMS = ilenums.Parameters.grid_parameters.value
PARAMETERS_N_PARAMS = ilenums.Parameters.n_parameters.value
PARAMETERS_N_NEIGHBORS = ilenums.Parameters.n_neighbors.value
PREP_SCALER = ilenums.Preprocessing.scaler.value
NETWORKS_GNN = ilenums.Networks.gnn.value
MODEL = "model"


class GNNRegressionTrainer:
    """
    Gestiona l'optimització, entrenament i predicció dels models GNN
    per la tasca de regressió.
    """
    def __init__(self):
        self.study_name = None
        self.direction = None
        self.storage = None
        self.load_if_exists = None
        self.n_trials = None
        self.device = torch.device(
            DEVICES_CUDA if torch.cuda.is_available() else DEVICES_CPU
        )

    def _update_nonimprovement_count(
        self, 
        count: int, 
        best_loss: float, 
        current_loss: float
    ) -> tuple[int, float]:
        if current_loss < best_loss:
            best_loss = current_loss
            count = 0
            return count, best_loss
        else:
            count += 1
            return count, best_loss
    
    def _train(
        self, 
        data: torch_geometric.data.Data, 
        model: torch.nn.Module
    ) -> torch.Tensor:
        model.train()
        model.optimizer.zero_grad()
        
        if hasattr(data, 'train_mask'):
            outputs = model(data)
            mask = data.train_mask
            loss = model.criterion(outputs[mask], data.y[mask])
        else:
            outputs = model(data['train'])
            loss = model.criterion(outputs, data['train'].y)
        
        loss.backward()
        model.optimizer.step()
        
        return loss

    @torch.no_grad()
    def _validate(
        self, 
        data: torch_geometric.data.Data, 
        model: torch.nn.Module
    ) -> float:
        model.eval()
        
        if hasattr(data, 'val_mask'):
            outputs = model(data)
            mask = data.val_mask
            outputs_rescaled = torch.tensor(
                data.y_scaler.inverse_transform(outputs[mask].cpu().numpy())
            )
            targets_rescaled = torch.tensor(
                data.y_scaler.inverse_transform(data.y[mask].cpu().numpy())
            )
        else:
            outputs = model(data['val']).to(self.device)
            outputs_rescaled = torch.tensor(
                data['val'].y_scaler.inverse_transform(outputs.cpu().numpy())
            )
            targets_rescaled = torch.tensor(
                data['val'].y_scaler.inverse_transform(data['val'].y.cpu().numpy())
            )
        
        mae = model.criterion(outputs_rescaled, targets_rescaled)
        
        return mae

    def train_validate(
        self, 
        data: torch_geometric.data.Data, 
        model: torch.nn.Module, 
        max_epochs: int, 
        patience: int, 
        verbose: int,
        show_train_process: bool = False,
        trial = None
    ) -> None:
        metrics = {SUBSETS_TRAIN: {METRICS_LOSS: []},
                   SUBSETS_VAL: {METRICS_LOSS: []}}

        best_validation_loss = 0
        nonimprovement_count = 0

        for epoch in range(1, max_epochs + 1):
            train_loss = self._train(data, model)
            validation_loss  = self._validate(data, model)
            model.scheduler.step(validation_loss)
            metrics[SUBSETS_TRAIN][METRICS_LOSS].append(train_loss.item())
            metrics[SUBSETS_VAL][METRICS_LOSS].append(validation_loss.item())

            # Reportar a Optuna per al pruning
            if trial is not None:
                trial.report(validation_loss.item(), epoch)
                # Comprovar si s'ha de fer pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            if epoch == 1:
                best_validation_loss = validation_loss.item()

            if show_train_process and (epoch == 1 or epoch % 50 == 0):
                print_reg_epoch_summary(epoch, train_loss, validation_loss)

            nonimprovement_count, \
                best_validation_loss = self._update_nonimprovement_count(
                    nonimprovement_count, best_validation_loss, validation_loss.item()
            )

            if nonimprovement_count > patience:
                if verbose > 3:
                    print_early_stopping(epoch)                
        
                if show_train_process: 
                    trainingvizs = ilvizs.TrainingVisualizer()
                    trainingvizs.plot_metrics(metrics[SUBSETS_TRAIN], metrics[SUBSETS_VAL])
        
                return validation_loss.item()

        if show_train_process:
                    trainingvizs = ilvizs.TrainingVisualizer()
                    trainingvizs.plot_metrics(metrics[SUBSETS_TRAIN], metrics[SUBSETS_VAL])

        return validation_loss.item()

    @torch.no_grad()
    def test(
        self, 
        data: torch_geometric.data.Data, 
        model: torch.nn.Module,
        pretrained_model: str
    ) -> dict:
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        model.to(self.device)

        if pretrained_model is not None:
            model.load_state_dict(torch.load(pretrained_model))

        mae_loss = nn.L1Loss()
        model.eval()
        
        if hasattr(data, 'test_mask'):
            outputs = model(data)
            mask = data.test_mask
            outputs_rescaled = torch.tensor(
                data.y_scaler.inverse_transform(outputs[mask].cpu().numpy())
            )
            targets_rescaled = torch.tensor(
                data.y_scaler.inverse_transform(data.y[mask].cpu().numpy())
            )
        else:
            outputs = model(data['test'])
            outputs_rescaled = torch.tensor(
                data['test'].y_scaler.inverse_transform(outputs.cpu().numpy())
            )
            targets_rescaled = torch.tensor(
                data['test'].y_scaler.inverse_transform(data['test'].y.cpu().numpy())
            )

        torch.cuda.synchronize()
        end = time.perf_counter()
        elapsed_time = end - start  
        
        positioning_error = torch.sqrt(
            torch.sum((outputs_rescaled - targets_rescaled)**2, dim=1)
        )
        mean_positioning_error = torch.mean(positioning_error)
        
        outputs_x, outputs_y = outputs_rescaled[:, 0], outputs_rescaled[:, 1]
        targets_x, targets_y = targets_rescaled[:, 0], targets_rescaled[:, 1]
        
        mae = mae_loss(outputs_rescaled, targets_rescaled)
        mae_x = mae_loss(outputs_x, targets_x)
        mae_y = mae_loss(outputs_y, targets_y)
                
        return {
            METRICS_MPE: mean_positioning_error.item(),
            METRICS_MAE: mae.item(),
            METRICS_MAE_X: mae_x.item(),
            METRICS_MAE_Y: mae_y.item(),
            METRICS_ELAPSED_TIME: elapsed_time
        }
    
    def set_gridparams(self, trial, data, model_class):
        if model_class.__name__ in ['GCNRegressor', 'SAGERegressor']:

            if hasattr(data, 'train_mask'):
                data = data
            else:
                data = data['train']
        
            params = {
                'input_dim': data.num_features,
                'output_dim': data.y.shape[1],
            }   
            
            hidden_dims = []
            dropouts = []

            params['n_layers'] = trial.suggest_categorical('n_layers', [2])
            for i in range(params['n_layers']):
                hidden_dims.append(trial.suggest_categorical(f'hidden_dim_layer_{i}', [128, 256]))
            
                if i < params['n_layers'] - 1:
                    dropouts.append(trial.suggest_categorical(f'dropout_layer_{i}', [0.2, 0.4, 0.6]))
            
            params['hidden_dim'] = hidden_dims
            params['dropout'] = dropouts
            params['learning_rate'] = trial.suggest_categorical('learning_rate', [0.005, 0.01])            
            params['weight_decay'] = trial.suggest_categorical('weight_decay', [1e-5, 1e-4])            
            params['optim_factor'] = trial.suggest_categorical('optim_factor', [0.9])
            params['mlp_layers'] = trial.suggest_categorical('mlp_layers', [2, 4])
    
        elif model_class.__name__ in ['GATRegressor']:
            params = {
                'input_dim': data.num_features,
                'output_dim': data.y.shape[1],
            }   
            
            hidden_dims = []
            dropouts = []
            heads = []

            params['n_layers'] = trial.suggest_categorical('n_layers', [2])
            for i in range(params['n_layers']):
                hidden_dims.append(trial.suggest_categorical(f'hidden_dim_layer_{i}', [128, 256]))
                heads.append(trial.suggest_categorical(f'heads_layer_{i}', [1]))
            
                if i < params['n_layers'] - 1:
                    dropouts.append(trial.suggest_categorical(f'dropout_layer_{i}', [0.2, 0.4, 0.6]))
            
            params['hidden_dim'] = hidden_dims
            params['dropout'] = dropouts
            params['heads'] = heads
            params['learning_rate'] = trial.suggest_categorical('learning_rate', [0.005, 0.01])            
            params['weight_decay'] = trial.suggest_categorical('weight_decay', [1e-5, 1e-4])            
            params['optim_factor'] = trial.suggest_categorical('optim_factor', [0.9])
            params['mlp_layers'] = trial.suggest_categorical('mlp_layers', [2, 4])
        
        return params
    
    def objective(self, trial, data, model_class):
        params = self.set_gridparams(trial, data, model_class)
        model = model_class(**params).to(self.device)

        mae = self.train_validate(
            data, model, self.max_epochs, self.patience, verbose=0, trial=trial
        )
        
        return mae

    def run_optuna_study(
            self,
            data: torch_geometric.data.Data, 
            model_class: type, 
            study_name,
            direction,
            storage,    
            load_if_exists,
            n_trials,
            callbacks=None
    ):
        
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            load_if_exists=load_if_exists,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, 
                                               n_warmup_steps=100,
                                               n_min_trials=5),
            sampler=optuna.samplers.TPESampler(seed=SEED)
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            lambda trial: self.objective(trial, data, model_class),
            n_trials=n_trials,
            n_jobs=1,
            callbacks=callbacks)
    

def summarize_predictions(predictions, graph_params, model_params, 
                          task=TASKS_REG, save_path=None):
    
    if task == TASKS_REG:
        metrics = ['mpe', 'mae', 'mae_x', 'mae_y', 'elapsed_time']

    if len(predictions) == 0:
        raise ValueError("Predictions list is empty.")

    summary_data = {}
    summary_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for m in metrics:
        if m not in predictions[0]:
            raise KeyError(f"The metric '{m}' is not found.")
        values = [p[m] for p in predictions]
        summary_data[f"{m}_mean"] = round(pd.Series(values).mean(), 4)
        summary_data[f"{m}_std"] = round(pd.Series(values).std(), 4)

    summary_df = pd.DataFrame([summary_data])

    summary_data['graph_params'] = str(graph_params)
    summary_data['model_params'] = str(model_params)

    if save_path is not None:
        try:
            with open(save_path, 'x') as f:
                pd.DataFrame([summary_data]).to_csv(f, index=False)
        except FileExistsError:
           pd.DataFrame([ summary_data]).to_csv(save_path, mode='a', header=False, index=False)

    return summary_df

def print_cls_epoch_summary(epoch, train_loss, train_accuracy, 
                            validation_loss, validation_accuracy):
    """
    Imprimeix les mètriques obtingudes per epoch en la tasca de classificació.
    """
    print(f"Epoch {epoch:02d} => "
          f"Train Loss: {train_loss:.4f}, "
          f"Train Acc.: {100*train_accuracy:.2f}% | "
          f"Validation Loss: {validation_loss:.4f}, " 
          f"Validation Acc.: {100*validation_accuracy:.2f}%")

def print_reg_epoch_summary(epoch, train_loss, validation_loss):
    """
    Imprimeix les mètriques obtingudes per epoch en la tasca de regressió.
    """
    print(f"Epoch {epoch:02d} => "
          f"Train Loss: {train_loss:.4f} | "
          f"Validation Loss: {validation_loss:.4f}")

def print_cls_test_summary(i, num_runs, test_accuracy,
                           training_elapsed_time, testing_elapsed_time):
    """
    Imprimeix el resultat d'una prova de test en la tasca de classificació.
    """
    return (f"Test [{i}/{num_runs}] => "
            f"Accuracy: {test_accuracy:.4f} "
            f"[Training time: {training_elapsed_time:.2f}s.] | "
            f"[Testing time: {testing_elapsed_time:.2e}s.]")

def print_reg_test_summary(i, num_runs, output_metrics,
                            training_elapsed_time, testing_elapsed_time):
    """
    Imprimeix el resultat d'una prova de test en la tasca de regressió.
    """
    return (f"Test [{i+1}/{num_runs}] => "
            f"MAE: {output_metrics[METRICS_MAE]:.2f} | " 
            f"MAE (x): {output_metrics[METRICS_MAE_X]:.2f} | " 
            f"MAE (y): {output_metrics[METRICS_MAE_Y]:.2f} | " 
            f"[Training time: {training_elapsed_time:.2f}s.] | "
            f"[Testing time: {testing_elapsed_time:.2e}s.]")

def print_cls_summary(mean_accuracy, std_accuracy, max_accuracy,
                      mean_train_time, std_train_time,
                      mean_test_time, std_test_time):
    """
    Imprimeix el resum de les proves de test realitzades en la tasca de classificació.
    """
    print(f"{SEPARATOR}\nTests summary \n{SEPARATOR}\n"
          f"Mean accuracy: {mean_accuracy:.2f}"
          f"± {std_accuracy:.2f}\n"
          f"Max. accuracy: {max_accuracy:.2f}\n"
          f"Training mean time (s): {mean_train_time:.2f} ± {std_train_time:.2f}\n"
          f"Testing mean time (s): {mean_test_time:.2e} ± {std_test_time:.2e}\n")

def print_reg_summary(avg_mpe, std_mpe, avg_mae, std_mae,
                      avg_mae_x, std_mae_x, avg_mae_y, std_mae_y,
                      mean_train_time, std_train_time,
                      mean_test_time, std_test_time):
    """
    Imprimeix el resum de les proves de test realitzades en la tasca de regressió.
    """
    print(f"{SEPARATOR}\nTests summary \n{SEPARATOR}\n"
          f"MPE: {avg_mpe:.2f} ± {std_mpe:.2f}\n"
          f"MAE: {avg_mae:.2f} ± {std_mae:.2f}\n"
          f"MAE (x): {avg_mae_x:.2f} ± {std_mae_x:.2f}\n"
          f"MAE (y): {avg_mae_y:.2f} ± {std_mae_y:.2f}\n"
          f"Training mean time (s): {mean_train_time:.2f} ± {std_train_time:.2f}\n"
          f"Testing mean time (s): {mean_test_time:.2e} ± {std_test_time:.2e}\n")

def print_early_stopping(epoch):
    """
    Imprimeix missatge informant de l'aturada anticipada durant l'entrenament.
    """
    print(f"Early stopping triggered at epoch {epoch}.")

def save_results_to_csv(results, filename="results.csv"):
    """
    Guarda els resultats de la prova de test en un fitxer CSV.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    df_row = pd.DataFrame([{
        'timestamp': timestamp,
        **{k: v for k, v in results.items() if k != METRICS_OUTPUT_ERRORS}  
    }])
    
    if os.path.exists(filename):
        df_row.to_csv(filename, mode='a', header=False, index=False)
    else:
        df_row.to_csv(filename, mode='w', header=True, index=False)
