import os
import pandas as pd
import numpy as np
import torch
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from joblib import Parallel, delayed
from models.TriCondNet import MetalPINN, SemiconductorPINN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
_RANDSTATE_NJOBS = int(os.environ.get("BO_RANDOMSTATE_NJOBS", "3"))

class metal_pinn_opt:
    def __init__(self,
                 data_dict, 
                 optimal_features_ranking,
                 num_trials=100,
                 keys = ['features', 'formula', 'target', 'source', 'class'],
                 target_name="target",
                 class_name="class"):
        self.data_dict = data_dict 
        self.opt_features = optimal_features_ranking
        self.target_name = target_name
        self.class_name = class_name
        self.num_trials = num_trials

        self.opt_counter = 0 

        self.ALL_VAL_MAE = {}
        self.ALL_VAL_R2 = {}
        
        self.ALL_TRAIN_MAE = {}
        self.ALL_TRAIN_R2 = {}

        def data_merger(data, keys):    
            dataframe = pd.DataFrame()
            for key in keys: 
                df = data[key]
                dataframe = pd.concat([dataframe, df], axis=1)
            return dataframe
        
        dataframe = data_merger(data_dict, keys) 
        self.data = dataframe
    
        dimensions = [
        Integer(16, len(self.opt_features), name='n_feat'),
        Integer(20, 200, name='num_epochs'),
        Integer(2, 4, name="n_layers"),
        Categorical(["brick"], name="architecture_style"),
        Integer(128, 320, name="brick_width"),
        Integer(64, 512, name="funnel_width"),
        Categorical([0.5, 0.6, 0.7, 0.8], name="funnel_rate"),

        Real(1e-4, 3e-3, prior='log-uniform', name='lr'),
        Real(1e-6, 1e-3, prior='log-uniform', name='weight_decay'),
        Categorical([64, 128, 256], name='batch_size'),

        Real(0.1, 0.3, name='dropout_rate'),

        Categorical(['relu', 'elu'], name='act'),]

        self.dimensions = dimensions


    def data_indexer(self, data, index, feature_columns):
        final_data = {"features": data.iloc[index].reset_index(drop=True)[feature_columns],
                    "formula": data.iloc[index].reset_index(drop=True)["formula"],
                    "target": pd.DataFrame({"target": data.iloc[index].reset_index(drop=True)["target"]}),
                    "class": pd.DataFrame({"class": data.iloc[index].reset_index(drop=True)["class"]}),
                    "source": data.iloc[index].reset_index(drop=True)["source"]}
        return final_data

    def _train_one_fold(self, train_idx, val_idx, n_feat, num_epochs, architecture, lr, weight_decay,
                         batch_size, dropout_rate, act):
        Feature_Columns = [col for col in self.data.columns
                           if col not in ["formula", "source", "entry", "target", "class"]]
        Val_Data   = self.data_indexer(self.data, val_idx,   Feature_Columns)
        Train_Data = self.data_indexer(self.data, train_idx, Feature_Columns)

        Metal = MetalPINN(
            target_name=self.target_name,
            optimal_descriptors=self.opt_features,
            n_feat=n_feat,
            architecture=architecture,
            act=act,
            dropout_rate=dropout_rate,
        )
        history = Metal.fit(
            train_df=Train_Data["features"], train_target=Train_Data["target"],
            lr=lr, epochs=num_epochs, delta=0.0,
            batch_size=batch_size, weight_decay=weight_decay,
            xscale="standard", impute_missing=0,
            xscale_before_impute=True, use_scheduler=True,
            verbose=False,
        )
        with torch.no_grad():
            val_predictions = Metal.predict(Val_Data["features"])
        mse_val = mean_squared_error(Val_Data["target"], val_predictions)
        mae_val = mean_absolute_error( Val_Data["target"], val_predictions)
        r2_val = r2_score(Val_Data["target"], val_predictions)
        return {
            "mae_train": history["mae_loss"][-1],
            "r2_train":  history["r2_loss"][-1],
            "mse_val":   mse_val,
            "mae_val":   mae_val,
            "r2_val":    r2_val,
        }
    def objective(self,
                  n_feat,
                  num_epochs,
                  n_layers,
                  architecture_style,
                  brick_width,
                  funnel_width,
                  funnel_rate,
                  lr,
                  weight_decay,
                  batch_size,
                  dropout_rate,
                  act):
        n_feat = int(n_feat)
        num_epochs = int(num_epochs)
        n_layers = int(n_layers)
        architecture_style = str(architecture_style)
        brick_width = int(brick_width)
        funnel_width = int(funnel_width)
        funnel_rate = float(funnel_rate)
        batch_size = int(batch_size)
        self.opt_counter += 1
        if architecture_style == "brick":
            architecture = tuple([brick_width] for _ in range(n_layers))
        else:
            architecture = ([funnel_width],)
            current_layer = funnel_width
            for _ in range(1, n_layers):
                current_layer = max(1, int(current_layer * funnel_rate))
                architecture = architecture + ([current_layer],)
        gkf = GroupKFold(n_splits=5)
        folds = list(gkf.split(self.data, groups=self.data["formula"]))
        results = Parallel(n_jobs=_RANDSTATE_NJOBS, prefer="threads")(
            delayed(self._train_one_fold)(
                train_idx, val_idx, n_feat, num_epochs, architecture, lr, weight_decay,
                batch_size, dropout_rate, act,
            )
            for train_idx, val_idx in folds
        )
        mae_val_losses = [r["mae_val"] for r in results]
        return float(np.mean(mae_val_losses))


    def run_optimization(self,
                         n_calls=None):
        if n_calls is None:
            n_calls = self.num_trials

        print(f"Starting Bayesian Optimization with {n_calls} calls...")
        print(f"Optimizing {len(self.dimensions)} hyperparameters...")

        @use_named_args(self.dimensions)
        def objective_wrapper(**params):
            return self.objective(**params)


        self.search_result = gp_minimize(
            func=objective_wrapper,
            dimensions=self.dimensions,
            acq_func='EI',
            n_calls=n_calls,
            random_state=22,
            verbose=True
        )

        self.report_best_result()
        return self.search_result

    def report_best_result(self):
        print("\n" + "="*60)
        print("BEST HYPERPARAMETER CONFIGURATION")
        print("="*60)
        print(f"Best Average Val Loss: {self.search_result.fun:.6f}")
        print("\nHyperparameters:")
        print(f"  n_feat: {self.search_result.x[0]}")
        print(f"  num_epochs: {self.search_result.x[1]}")
        print(f"  n_layers: {self.search_result.x[2]}")
        print(f"  arch_style: {self.search_result.x[3]}")
        print(f"  brick_width: {self.search_result.x[4]}")
        print(f"  funnel_width: {self.search_result.x[5]}")
        print(f"  funnel_rate: {self.search_result.x[6]}")
        print(f"  lr: {self.search_result.x[7]:.6e}")
        print(f"  weight_decay: {self.search_result.x[8]:.6e}")
        print(f"  batch_size: {self.search_result.x[9]}")
        print(f"  dropout_rate: {self.search_result.x[10]:.3f}")
        print(f"  act: {self.search_result.x[11]}")
        print("="*60)


class semiconductor_pinn_opt:
    def __init__(self,
                 data_dict, 
                 optimal_features_ranking,
                 num_trials=100,
                 keys = ['features', 'formula', 'target', 'source', 'class'],
                 target_name="target"):
        self.data_dict = data_dict 
        self.opt_features = optimal_features_ranking
        self.target_name = target_name
        self.num_trials = num_trials

        self.opt_counter = 0 

        self.ALL_VAL_MAE = {}
        self.ALL_VAL_R2 = {}
        
        self.ALL_TRAIN_MAE = {}
        self.ALL_TRAIN_R2 = {}
        def data_merger(data, keys): 
            dataframe = pd.DataFrame()
            for key in keys: 
                df = data[key]
                dataframe = pd.concat([dataframe, df], axis=1)
            return dataframe
        
        dataframe = data_merger(data_dict, keys) 
        self.data = dataframe
    
        dimensions = [
        Integer(32, len(self.opt_features), name='n_feat'),
        Integer(20, 500, name="num_epochs"),
        Integer(2, 3, name="n_layers"),
        Categorical(["funnel", "brick"], name="architecture_style"),
        Integer(64, 256, name="brick_width"),
        Integer(256, 512, name="funnel_width"),
        Categorical([0.5, 0.6, 0.7, 0.8], name="funnel_rate"),


        Real(1e-4, 1e-2, prior='log-uniform', name='lr'),
        Real(1e-5, 1e-2, prior='log-uniform', name='weight_decay'),
        Categorical([32, 64, 128, 256], name='batch_size'),

        Real(0.2, 0.5, name='dropout_rate'),

        Categorical(['relu', 'elu'], name='act'),]

        self.dimensions = dimensions


    def data_indexer(self, data, index, feature_columns):
        final_data = {"features": data.iloc[index].reset_index(drop=True)[feature_columns],
                    "formula": data.iloc[index].reset_index(drop=True)["formula"],
                    "target": pd.DataFrame({"target": data.iloc[index].reset_index(drop=True)["target"]}),
                    "class": pd.DataFrame({"class": data.iloc[index].reset_index(drop=True)["class"]}),
                    "source": data.iloc[index].reset_index(drop=True)["source"]}
        return final_data

    def _train_one_state(self, state, n_feat, num_epochs, architecture, lr, weight_decay,
                         batch_size, dropout_rate, act):
        Feature_Columns = [col for col in self.data.columns
                           if col not in ["formula", "source", "entry", "target", "class"]]
        Splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=state)
        Train_Idx, Val_Idx = next(Splitter.split(self.data, groups=self.data["formula"]))
        Val_Data   = self.data_indexer(self.data, Val_Idx,   Feature_Columns)
        Train_Data = self.data_indexer(self.data, Train_Idx, Feature_Columns)

        Semiconductor = SemiconductorPINN(
            target_name=self.target_name,
            optimal_descriptors=self.opt_features,
            n_feat=n_feat,
            architecture=architecture,
            act=act,
            dropout_rate=dropout_rate,
        )
        history = Semiconductor.fit(
            train_df=Train_Data["features"], train_target=Train_Data["target"],
            lr=lr, epochs=num_epochs, delta=0.0,
            batch_size=batch_size, weight_decay=weight_decay,
            xscale="standard", impute_missing=0,
            xscale_before_impute=True, use_scheduler=True,
            verbose=False,
        )
        with torch.no_grad():
            val_predictions = Semiconductor.predict(Val_Data["features"])
        mse_val = mean_squared_error(Val_Data["target"], val_predictions)
        mae_val = mean_absolute_error( Val_Data["target"], val_predictions)
        r2_val = r2_score(Val_Data["target"], val_predictions)
        return {
            "mae_train": history["mae_loss"][-1],
            "r2_train":  history["r2_loss"][-1],
            "mse_val":   mse_val,
            "mae_val":   mae_val,
            "r2_val":    r2_val,
        }

    def objective(self,
                  n_feat,
                  num_epochs,
                  n_layers,
                  architecture_style,
                  brick_width,
                  funnel_width,
                  funnel_rate,
                  lr,
                  weight_decay,
                  batch_size,
                  dropout_rate,
                  act):
        n_feat = int(n_feat)
        num_epochs = int(num_epochs)
        n_layers = int(n_layers)
        architecture_style = str(architecture_style)
        brick_width = int(brick_width)
        funnel_width = int(funnel_width)
        funnel_rate = float(funnel_rate)
        batch_size = int(batch_size)

        random_states = [0, 1, 2, 3, 4]
        self.opt_counter += 1

        if architecture_style == "brick":
            architecture = tuple([brick_width] for _ in range(n_layers))
        else:
            architecture = ([funnel_width],)
            current_layer = funnel_width
            for _ in range(1, n_layers):
                current_layer = max(1, int(current_layer * funnel_rate))
                architecture = architecture + ([current_layer],)

        results = Parallel(n_jobs=_RANDSTATE_NJOBS, prefer="threads")(
            delayed(self._train_one_state)(
                state, n_feat, num_epochs, architecture, lr, weight_decay,
                batch_size, dropout_rate, act,
            )
            for state in random_states
        )

        mae_train_losses = [r["mae_train"] for r in results]
        r2_train_losses  = [r["r2_train"]  for r in results]
        mse_val_losses   = [r["mse_val"]   for r in results]
        mae_val_losses   = [r["mae_val"]   for r in results]
        r2_val_losses    = [r["r2_val"]    for r in results]

        self.ALL_TRAIN_MAE[self.opt_counter] = mae_train_losses
        self.ALL_TRAIN_R2[self.opt_counter]  = r2_train_losses
        self.ALL_VAL_MAE[self.opt_counter]   = mae_val_losses
        self.ALL_VAL_R2[self.opt_counter]    = r2_val_losses

        return float(np.mean(mse_val_losses))
    
    
    def run_optimization(self, 
                         n_calls=None):
        if n_calls is None: 
            n_calls = self.num_trials
        
        print(f"Starting Bayesian Optimization with {n_calls} calls...")
        print(f"Optimizing {len(self.dimensions)} hyperparameters...")

        @use_named_args(self.dimensions)
        def objective_wrapper(**params):
            return self.objective(**params)
    

        self.search_result = gp_minimize(
            func=objective_wrapper,
            dimensions=self.dimensions,
            acq_func='EI',
            n_calls=n_calls,
            random_state=22,
            verbose=True
        )
        
        self.report_best_result()
        return self.search_result
    
    def report_best_result(self):
        print("\n" + "="*60)
        print("BEST HYPERPARAMETER CONFIGURATION")
        print("="*60)
        print(f"Best Average Val Loss: {self.search_result.fun:.6f}")
        print("\nHyperparameters:")
        print(f"  n_feat: {self.search_result.x[0]}")
        print(f"  num_epochs: {self.search_result.x[1]}")
        print(f"  n_layers: {self.search_result.x[2]}")
        print(f"  arch_style: {self.search_result.x[3]}")
        print(f"  brick_width: {self.search_result.x[4]}")
        print(f"  funnel_width: {self.search_result.x[5]}")
        print(f"  funnel_rate: {self.search_result.x[6]}")
        print(f"  lr: {self.search_result.x[7]:.6e}")
        print(f"  weight_decay: {self.search_result.x[8]:.6e}")
        print(f"  batch_size: {self.search_result.x[9]}")
        print(f"  dropout_rate: {self.search_result.x[10]:.3f}")
        print(f"  act: {self.search_result.x[11]}")
        print("="*60)
