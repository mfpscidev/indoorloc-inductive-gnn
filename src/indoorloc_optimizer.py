import torch
import torch_geometric
import optuna

import indoorloc_enums as ilenums
import indoorloc_trainer as iltrainer

# Numerical constants
SEED = 42

# Enums
CUDA = ilenums.Devices.cuda.value
CPU = ilenums.Devices.cpu.value


class GNNRegressionOptimizer:
    def __init__(self):
        self.device = torch.device(
            CUDA if torch.cuda.is_available() else CPU
        )

    def _set_gridparams(self, trial, data, model_class):
        if model_class.__name__ in ['SAGERegressor']:

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
        
        return params
    
    def _objective(self, trial, data, model_class):
        params = self._set_gridparams(trial, data, model_class)
        model = model_class(**params).to(self.device)
        trainer = iltrainer.GNNRegressionTrainer()

        mae = trainer.train_validate(
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
            lambda trial: self._objective(trial, data, model_class),
            n_trials=n_trials,
            n_jobs=1,
            callbacks=callbacks)