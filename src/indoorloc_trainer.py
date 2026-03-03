import os
import time
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric
import optuna

import indoorloc_viz as ilviz
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
    def __init__(self):
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
        model.to(self.device)

        for epoch in range(1, max_epochs + 1):
            train_loss = self._train(data, model)
            validation_loss  = self._validate(data, model)
            model.scheduler.step(validation_loss)

            metrics[SUBSETS_TRAIN][METRICS_LOSS].append(train_loss.item())
            metrics[SUBSETS_VAL][METRICS_LOSS].append(validation_loss.item())

            if trial is not None:
                trial.report(validation_loss.item(), epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            if epoch == 1:
                best_validation_loss = validation_loss.item()

            if show_train_process and (epoch == 1 or epoch % 100 == 0):
                print_reg_epoch_summary(epoch, train_loss, validation_loss)

            nonimprovement_count, \
                best_validation_loss = self._update_nonimprovement_count(
                    nonimprovement_count, best_validation_loss, validation_loss.item()
            )

            if nonimprovement_count > patience:
                if verbose > 3:
                    print_early_stopping(epoch)                
        
                if show_train_process: 
                    trainingviz = ilviz.TrainingVisualizer()
                    trainingviz.plot_metrics(metrics[SUBSETS_TRAIN], metrics[SUBSETS_VAL])
        
                return validation_loss.item()

        if show_train_process:
                    trainingviz = ilviz.TrainingVisualizer()
                    trainingviz.plot_metrics(metrics[SUBSETS_TRAIN], metrics[SUBSETS_VAL])

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
        
        mae = model.criterion(outputs_rescaled, targets_rescaled)
        mae_x = model.criterion(outputs_x, targets_x)
        mae_y = model.criterion(outputs_y, targets_y)
                
        return {
            METRICS_MPE: mean_positioning_error.item(),
            METRICS_MAE: mae.item(),
            METRICS_MAE_X: mae_x.item(),
            METRICS_MAE_Y: mae_y.item(),
            METRICS_ELAPSED_TIME: elapsed_time
        }
    

class GNNClassificationTrainer:
    def __init__(self):
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
    ) -> tuple[torch.Tensor, float]:
        model.train()
        model.optimizer.zero_grad()
        
        if hasattr(data, 'train_mask'):
            outputs = model(data)
            mask = data.train_mask
            loss = model.criterion(outputs[mask], data.y[mask])
            prediction = outputs[mask].argmax(1)
            accuracy = prediction.eq(data.y[mask]).sum().item() / mask.sum().item()
        else:
            outputs = model(data['train'])
            loss = model.criterion(outputs, data['train'].y)
            prediction = outputs.argmax(1)
            accuracy = prediction.eq(data['train'].y).sum().item() / data['train'].y.size(0)
        
        loss.backward()
        model.optimizer.step()
        
        return loss, accuracy

    @torch.no_grad()
    def _validate(
        self, 
        data: torch_geometric.data.Data, 
        model: torch.nn.Module
    ) -> tuple[float, float]:
        model.eval()
        
        if hasattr(data, 'val_mask'):
            outputs = model(data)
            mask = data.val_mask
            loss = model.criterion(outputs[mask], data.y[mask])
            prediction = outputs[mask].argmax(1)
            accuracy = prediction.eq(data.y[mask]).sum().item() / mask.sum().item()
        else:
            outputs = model(data['val'])
            loss = model.criterion(outputs, data['val'].y)
            prediction = outputs.argmax(1)
            accuracy = prediction.eq(data['val'].y).sum().item() / data['val'].y.size(0)
        
        return loss, accuracy

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
        metrics = {SUBSETS_TRAIN: {METRICS_LOSS: [], METRICS_ACCU: []},
                   SUBSETS_VAL: {METRICS_LOSS: [], METRICS_ACCU: []}}
        model.to(self.device)

        best_validation_loss = INF
        nonimprovement_count = 0
        model.to(self.device)

        for epoch in range(1, max_epochs + 1):
            train_loss, train_accuracy = self._train(data, model)
            validation_loss, validation_accuracy = self._validate(data, model)
            model.scheduler.step(validation_loss)

            metrics[SUBSETS_TRAIN][METRICS_LOSS].append(train_loss.item())
            metrics[SUBSETS_TRAIN][METRICS_ACCU].append(train_accuracy)
            metrics[SUBSETS_VAL][METRICS_LOSS].append(validation_loss.item())
            metrics[SUBSETS_VAL][METRICS_ACCU].append(validation_accuracy)

            if trial is not None:
                trial.report(validation_loss.item(), epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
            if epoch == 1:
                best_validation_loss = validation_loss.item()

            if show_train_process and (epoch == 1 or epoch % 50 == 0):
                print_cls_epoch_summary(epoch, train_loss, train_accuracy, 
                                        validation_loss, validation_accuracy)

            nonimprovement_count, \
                best_validation_loss = self._update_nonimprovement_count(
                    nonimprovement_count, best_validation_loss, validation_loss.item()
            )

            if nonimprovement_count > patience:
                if verbose > 3:
                    print_early_stopping(epoch)    

                if show_train_process:
                    trainingviz = ilviz.TrainingVisualizer()
                    trainingviz.plot_metrics(metrics[SUBSETS_TRAIN], metrics[SUBSETS_VAL])

                return validation_loss.item()

        if show_train_process:
                    trainingviz = ilviz.TrainingVisualizer()
                    trainingviz.plot_metrics(metrics[SUBSETS_TRAIN], metrics[SUBSETS_VAL])

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

        model.eval()
        
        if hasattr(data, 'test_mask'):
            outputs = model(data)
            mask = data.test_mask
            predictions = outputs[mask].argmax(dim=1)
            targets = data.y[mask]
            accuracy = predictions.eq(targets).sum().item() / mask.sum().item()
        else:
            outputs = model(data['test'])
            predictions = outputs.argmax(dim=1)
            targets = data['test'].y
            accuracy = predictions.eq(targets).sum().item() / targets.size(0)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        elapsed_time = end - start  

        return {
            'accuracy': accuracy,
            'predictions': predictions.cpu().numpy(),
            'targets': targets.cpu().numpy(),
            METRICS_ELAPSED_TIME: elapsed_time
        }
        

def summarize_predictions(predictions, graph_params, model_params, 
                          task=TASKS_REG, save_path=None):
    
    if task == TASKS_REG:
        metrics = ['mpe', 'mae', 'mae_x', 'mae_y', 'elapsed_time']
    else:
        metrics = ['accuracy', 'elapsed_time']

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


def get_num_features(data, scheme):
    return (data.cls['train'] if scheme == 'inductive' else data.cls).num_features

def get_num_classes(data, scheme):
    return (data.cls['train'] if scheme == 'inductive' else data.cls).num_classes

def print_cls_epoch_summary(epoch, train_loss, train_accuracy, 
                            validation_loss, validation_accuracy):
    print(f"Epoch {epoch:02d} => "
          f"Train Loss: {train_loss:.4f}, "
          f"Train Acc.: {100*train_accuracy:.2f}% | "
          f"Validation Loss: {validation_loss:.4f}, " 
          f"Validation Acc.: {100*validation_accuracy:.2f}%")

def print_reg_epoch_summary(epoch, train_loss, validation_loss):
    print(f"Epoch {epoch:02d} => "
          f"Train Loss: {train_loss:.4f} | "
          f"Validation Loss: {validation_loss:.4f}")

def print_early_stopping(epoch):
    print(f"Early stopping triggered at epoch {epoch}.")

def save_results_to_csv(results, filename="results.csv"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    df_row = pd.DataFrame([{
        'timestamp': timestamp,
        **{k: v for k, v in results.items() if k != METRICS_OUTPUT_ERRORS}  
    }])
    
    if os.path.exists(filename):
        df_row.to_csv(filename, mode='a', header=False, index=False)
    else:
        df_row.to_csv(filename, mode='w', header=True, index=False)
    