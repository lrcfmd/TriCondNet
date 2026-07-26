import pandas as pd
import numpy as np 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import GroupShuffleSplit
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
sys.path.insert(0, str(Path.cwd().parent.parent))
sys.path.insert(0, str(Path.cwd().parent.parent.parent))
sys.path.insert(0, str(Path.cwd().parent.parent.parent.parent))

from evaluation.preprocessing import data_concatenater, data_indexer, architecture_generator, FeatureSelector
from models.TriMODNet_Implementation import MODNet_Classifier, MODNet_Regressor

class modnet_class_opt: 
    def __init__(self, 
                 data_dict,
                 ranked_features,
                 num_trials=10,
                 keys = ['features', 'formula', 'class', 'source'],
                 target_name="class"):
        self.target_name = target_name
        self.num_trials = num_trials
        self.opt_counter = 0
        self.ranked_features = ranked_features

        def data_merger(data, keys): 
            dataframe = pd.DataFrame()
            for key in keys: 
                df = data[key]
                dataframe = pd.concat([dataframe, df], axis=1)
            return dataframe
        dataframe = data_merger(data_dict, keys) 
        self.data = dataframe

        dimensions = [
        # Feature selection
        Integer(32, len(self.ranked_features), name='n_feat'),

        Integer(20, 400, name='num_epochs'),

        Categorical([32, 64, 96, 128, 160, 192, 224, 256, 288, 320], name='n_neurons_first'),
        Categorical([1.0, 0.75, 0.5, 0.25], prior=[0.4, 0.2, 0.2, 0.2], name='fraction1'),
        Categorical([1.0, 0.75, 0.5, 0.25], prior=[0.4, 0.2, 0.2, 0.2], name='fraction2'),
        Categorical([1.0, 0.75, 0.5, 0.25], prior=[0.4, 0.2, 0.2, 0.2], name='fraction3'),

        # Training hyperparameters
        Real(1e-4, 1e-2, prior='log-uniform', name='lr'),
        Categorical([32, 64, 128, 256], name='batch_size'),

        # Activation
        Categorical(['relu', 'elu'], name='act'),]
        self.dimensions = dimensions

    def class_data_indexer(self, data, feature_columns, index=None):

        if index is None: 
            final_data = {"features": data.reset_index(drop=True)[feature_columns],
                        "formula": data.reset_index(drop=True)["formula"],
                        "class": pd.DataFrame({"class": data.reset_index(drop=True)["class"]}),
                        "source": data.reset_index(drop=True)["source"]}
        else:
            final_data = {"features": data.iloc[index].reset_index(drop=True)[feature_columns],
                        "formula": data.iloc[index].reset_index(drop=True)["formula"],
                        "class": pd.DataFrame({"class": data.iloc[index].reset_index(drop=True)["class"]}),
                        "source": data.iloc[index].reset_index(drop=True)["source"]}

        return final_data
        

    def objective(self,
                  n_feat,
                  num_epochs,
                  n_neurons_first,
                  fraction1,
                  fraction2,
                  fraction3,
                  lr,
                  batch_size,
                  act):
        n_feat = int(n_feat)
        num_epochs = int(num_epochs)
        n_neurons_first = int(n_neurons_first)
        fraction1, fraction2, fraction3 = float(fraction1), float(fraction2), float(fraction3)
        batch_size = int(batch_size)

        w0 = n_neurons_first
        w1 = max(1, int(w0 * fraction1))
        w2 = max(1, int(w1 * fraction2))
        w3 = max(1, int(w2 * fraction3))
        architecture = ([w0], [w1], [w2], [w3])

        random_states = [0, 1, 2, 3, 4]
        val_losses_mcc = []
        self.opt_counter += 1
        for i in range(len(random_states)):
            DataToSplit = self.data
            Feature_Columns = [col for col in self.data.columns if col not in ["formula", "source", "entry", "class"]]
            Splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=random_states[i])
            Train_Idx, Val_Idx = next(Splitter.split(DataToSplit, groups=DataToSplit["formula"]))
            Val_Data = self.class_data_indexer(DataToSplit, Feature_Columns, Val_Idx)
            Train_Data = self.class_data_indexer(DataToSplit, Feature_Columns, Train_Idx)
            classifier = MODNet_Classifier(train_input=Train_Data, val_input=Val_Data, feature_rank=False, ranked_features=self.ranked_features)
            classifier.classifier(n_feat=n_feat, architecture=architecture, lr=lr, batch_size=batch_size, act=act, epochs=num_epochs, patience=None)
            val_loss_mcc = classifier.return_val_loss_MCC()
            val_losses_mcc.append(val_loss_mcc)

        return -np.mean(val_losses_mcc)

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
        print(f"  n_neurons_first: {self.search_result.x[2]}")
        print(f"  fraction1: {self.search_result.x[3]}")
        print(f"  fraction2: {self.search_result.x[4]}")
        print(f"  fraction3: {self.search_result.x[5]}")
        print(f"  lr: {self.search_result.x[6]:.6e}")
        print(f"  batch_size: {self.search_result.x[7]}")
        print(f"  act: {self.search_result.x[8]}")
        print("="*60)


class modnet_regressor_opt:
    def __init__(self, 
                 data_dict,
                 ranked_features,
                 num_trials=10,
                 keys = ['features', 'formula', 'target', 'source',],
                 target_name="target"):
        self.target_name = target_name
        self.num_trials = num_trials
        self.opt_counter = 0
        self.ranked_features = ranked_features

        def data_merger(data, keys): 
            dataframe = pd.DataFrame()
            for key in keys: 
                df = data[key]
                dataframe = pd.concat([dataframe, df], axis=1)
            return dataframe
        dataframe = data_merger(data_dict, keys) 
        self.data = dataframe

        dimensions = [
        # Feature selection
        Integer(32, len(self.ranked_features), name='n_feat'),

        Integer(20, 400, name='num_epochs'),

        Categorical([32, 64, 96, 128, 160, 192, 224, 256, 288, 320], name='n_neurons_first'),
        Categorical([1.0, 0.75, 0.5, 0.25], prior=[0.4, 0.2, 0.2, 0.2], name='fraction1'),
        Categorical([1.0, 0.75, 0.5, 0.25], prior=[0.4, 0.2, 0.2, 0.2], name='fraction2'),
        Categorical([1.0, 0.75, 0.5, 0.25], prior=[0.4, 0.2, 0.2, 0.2], name='fraction3'),

        # Training hyperparameters
        Real(1e-4, 1e-2, prior='log-uniform', name='lr'),
        Categorical([32, 64, 128, 256], name='batch_size'),

        # Activation
        Categorical(['relu', 'elu'], name='act'),]
        self.dimensions = dimensions

    def regress_data_indexer(self, data, feature_columns, index=None):

        if index is None: 
            final_data = {"features": data.reset_index(drop=True)[feature_columns],
                        "formula": data.reset_index(drop=True)["formula"],
                        self.target_name: pd.DataFrame({self.target_name: data.reset_index(drop=True)[self.target_name]}),
                        "source": data.reset_index(drop=True)["source"]}
        else:
            final_data = {"features": data.iloc[index].reset_index(drop=True)[feature_columns],
                        "formula": data.iloc[index].reset_index(drop=True)["formula"],
                        self.target_name: pd.DataFrame({self.target_name: data.iloc[index].reset_index(drop=True)[self.target_name]}),
                        "source": data.iloc[index].reset_index(drop=True)["source"]}

        return final_data
        

    def objective(self,
                  n_feat,
                  num_epochs,
                  n_neurons_first,
                  fraction1,
                  fraction2,
                  fraction3,
                  lr,
                  batch_size,
                  act):
        n_feat = int(n_feat)
        num_epochs = int(num_epochs)
        n_neurons_first = int(n_neurons_first)
        fraction1, fraction2, fraction3 = float(fraction1), float(fraction2), float(fraction3)
        batch_size = int(batch_size)

        w0 = n_neurons_first
        w1 = max(1, int(w0 * fraction1))
        w2 = max(1, int(w1 * fraction2))
        w3 = max(1, int(w2 * fraction3))
        architecture = ([w0], [w1], [w2], [w3])

        random_states = [0, 1, 2, 3, 4]
        val_losses = []
        self.opt_counter += 1
        for i in range(len(random_states)):
            DataToSplit = self.data
            Feature_Columns = [col for col in self.data.columns if col not in ["formula", "source", "entry", self.target_name]]
            Splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=random_states[i])
            Train_Idx, Val_Idx = next(Splitter.split(DataToSplit, groups=DataToSplit["formula"]))
            Val_Data = self.regress_data_indexer(DataToSplit, Feature_Columns, Val_Idx)
            Train_Data = self.regress_data_indexer(DataToSplit, Feature_Columns, Train_Idx)
            regressor = MODNet_Regressor(train_input=Train_Data, val_input=Val_Data, feature_rank=False, ranked_features=self.ranked_features, target_name=self.target_name)
            regressor.regressor(n_feat=n_feat, architecture=architecture, lr=lr, batch_size=batch_size, act=act, epochs=num_epochs, patience=None)
            val_loss = regressor.return_final_val_loss()
            val_losses.append(val_loss)

        return np.mean(val_losses)

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
        print(f"  n_neurons_first: {self.search_result.x[2]}")
        print(f"  fraction1: {self.search_result.x[3]}")
        print(f"  fraction2: {self.search_result.x[4]}")
        print(f"  fraction3: {self.search_result.x[5]}")
        print(f"  lr: {self.search_result.x[6]:.6e}")
        print(f"  batch_size: {self.search_result.x[7]}")
        print(f"  act: {self.search_result.x[8]}")
        print("="*60)






            

            

            



