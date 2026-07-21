# Import relevant packages
import sys
import copy
import random
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from lightgbm import LGBMClassifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(Path.cwd().parent.parent))
from evaluation.preprocessing import data_concatenater, RFFeatureSelector
from models import TriCondNet

def _set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def architecture_generator(arch_style, brick_width, funnel_width, n_layers, funnel_rate=None): 
    if arch_style=="brick":
        architecture = ()
        for n in range(0, n_layers):
            architecture = architecture + ([brick_width],)
        return architecture
    if arch_style=="funnel":
        architecture = ()
        current_layer = funnel_width
        for n in range(0, n_layers):
            architecture = architecture + ([current_layer],)
            current_layer = max(1, int(current_layer * funnel_rate))
        return architecture
    
class GBM_ClassifierDeployment:
    def __init__(
        self,
        data_path: str,
        cbfv: str,
        random_state: int,
        target_name: str = "target",
        class_name: str = "class",
        validation: bool = True,):

        self.target_name = target_name
        self.validation = validation
        self.class_name = class_name
        self.cbfv = cbfv
        self.random_state = random_state
        self.bool_loaded_HP = False
 
        Data = pd.read_csv(data_path)
        self.input_data = Data.copy()
 
        if "comp" in Data.columns:
            Data.drop("comp", axis=1, inplace=True)
 
        Feature_Columns = [
            col for col in Data.columns
            if col not in ["formula", "source", "entry", "target", "class", "atmosphere"]
        ]
        self.Feature_Columns = Feature_Columns
        self.max_feat = len(Feature_Columns)
        DataToSplit = Data.copy()
        if self.validation: 
            # 95/5 train / validation split, grouped by formula
            Splitter = GroupShuffleSplit(n_splits=1, train_size=0.95, random_state=random_state)
            Train_Idx, Validation_Idx = next(
                Splitter.split(DataToSplit, groups=DataToSplit["formula"])
            )
            Train = DataToSplit.iloc[Train_Idx]
            Val = DataToSplit.iloc[Validation_Idx]
            Train = Train.drop_duplicates(subset="formula").reset_index(drop=True)
            Val = Val.drop_duplicates(subset="formula").reset_index(drop=True)
            self.train = self.data_indexer(data=Train, index=None, feature_columns=Feature_Columns)
            self.val = self.data_indexer(data=Val, index=None, feature_columns=Feature_Columns)
        else:
            DataToSplit = DataToSplit.drop_duplicates(subset="formula").reset_index(drop=True)
            self.train = self.data_indexer(data=DataToSplit, index=None, feature_columns=Feature_Columns)
            
        # placeholders
        self.model = None
        self.ranked_features = None
        self.probs = None
        self.scaler = None
        self.n_feat = None

    def data_indexer(self, data, index=None, feature_columns=None):
        if index is None:
            df = data.reset_index(drop=True)
        else:
            df = data.iloc[index].reset_index(drop=True)
        indexed = {
            "features": df[feature_columns],
            "formula": df["formula"],
            self.target_name: pd.DataFrame({self.target_name: df[self.target_name]})
                              if self.target_name in df.columns else None,
            "class": pd.DataFrame({"class": df["class"]}),
            "source": df["source"] if "source" in df.columns else None,
        }
        if "atmosphere" in df.columns:
            indexed["atmosphere"] = df["atmosphere"]
        return indexed

    def feature_ranking(self, feature_rank, target_to_correlate,
                        feature_ranking_path=None, n_jobs=15):
        if feature_rank:
            merged_features, per_target_lists, cnmi = RFFeatureSelector.feature_selection(
                self.train["features"],
                self.train[target_to_correlate],
                mode="classification",
                target_name="class",
                n_jobs=n_jobs,
                random_state=42,
            )
            self.ranked_features = per_target_lists[target_to_correlate]
            if feature_ranking_path:
                with open(feature_ranking_path, "wb") as f:
                    pkl.dump({"ranked_features": self.ranked_features}, f)
        else:
            with open(feature_ranking_path, "rb") as f:
                self.ranked_features = pkl.load(f)["ranked_features"]
 
    def tune_hyperparameters( 
        self,
        n_feat_options: List[int] = None,
        n_iter: int = 50,
        cv_splits: int = 5,
        scoring: str = "balanced_accuracy",
        n_jobs: int = -1,
        verbose: int = 1,
    ):
        # 5 fold inner CV with GroupKFold (grouped by formula) to tune GBM hyperparameters and n_feat simultaneously.
        if self.ranked_features is None:
            raise ValueError("Run feature_ranking() before tuning.")

        train_df = data_concatenater(self.train)
        if self.validation:
            val_df = data_concatenater(self.val)
            cv_df = pd.concat([train_df, val_df], ignore_index=True)
        else:
            cv_df = train_df
        # generating list for n_feat option 
        pct_options = [0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        if n_feat_options is None: 
            n_feat_options = []
            for pct in pct_options:
                n_feat_options.append(round(pct*self.max_feat))
        best_score = -np.inf
        best_model = None
        best_n_feat = None
        best_params = None
        param_distributions = {
            "n_estimators":       [200, 300, 500, 700, 1000],
            "learning_rate":      [0.01, 0.05, 0.1, 0.2],
            "max_depth":          [-1, 5, 10, 15, 20],
            "num_leaves":         [15, 31, 63, 127, 255],
            "min_child_samples":  [5, 10, 20, 30, 50],
            "subsample":          [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree":   [0.4, 0.6, 0.7, 0.8, 1.0],
            "reg_alpha":          [0.0, 0.01, 0.1, 0.5, 1.0],
            "reg_lambda":         [0.0, 0.01, 0.1, 0.5, 1.0],
            "class_weight":       ["balanced", None],
            "min_split_gain":     [0.0, 0.01, 0.05, 0.1],
        }
        groups = cv_df["formula"]
        y = cv_df["class"].values
        gkf = GroupKFold(n_splits=cv_splits)

        for nf in n_feat_options:
            features_to_use = [f for f in self.ranked_features if f != "temp" and f != "class"][:nf]
            X = cv_df[features_to_use].values

            gbm = LGBMClassifier(random_state=42, n_jobs=n_jobs)

            search = RandomizedSearchCV(
                gbm,
                param_distributions,
                n_iter=n_iter,
                scoring=scoring,
                cv=gkf,          # pass the splitter object, not the generator
                random_state=42,
                n_jobs=n_jobs,
                verbose=verbose,
                refit=True,
            )
            search.fit(X, y, groups=groups)  # pass groups here

            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_model = search.best_estimator_
                best_n_feat = nf
                best_params = search.best_params_

            print(f"  n_feat={nf:>3d}  best_cv_{scoring}={search.best_score_:.4f}")
        self.n_feat = best_n_feat
        self.gbm_params = best_params
        self.model = best_model
        # Refit on full train+val with best settings
        features_to_use = [f for f in self.ranked_features if f != "temp" and f != "class"][:best_n_feat]
        X_full = cv_df[features_to_use].values
        y_full = cv_df["class"].values
        self.model.fit(X_full, y_full)
        self._features_used = features_to_use
        best_params_dict = copy.deepcopy(best_params)
        best_params_dict["n_feat"] = best_n_feat
        with open(f"gbm_class_opt_{self.cbfv}.pkl", "wb") as f:
            pkl.dump(best_params_dict, f)
        print(f"\nBest: n_feat={best_n_feat}, {scoring}={best_score:.4f}")
        print(f"Params: {best_params}")
        self.n_estimators, self.learning_rate, self.max_depth, self.num_leaves, self.min_child_samples, self.subsample, self.colsample_bytree, self.reg_alpha, self.reg_lambda, self.class_weight, self.min_split_gain = best_params["n_estimators"], best_params["learning_rate"], best_params["max_depth"], best_params["num_leaves"], best_params["min_child_samples"], best_params["subsample"], best_params["colsample_bytree"], best_params["reg_alpha"], best_params["reg_lambda"], best_params["class_weight"], best_params["min_split_gain"]
        self.n_feat = best_params_dict["n_feat"]      # best_params comes from search.best_params_ 
        self.bool_loaded_HP = True

    def load_hyperparameters(self,
                             pickle_path_file):
        
        with open(pickle_path_file, "rb") as f:
            params = pkl.load(f)
        self.n_estimators, self.learning_rate, self.max_depth, self.num_leaves, self.min_child_samples, self.subsample, self.colsample_bytree, self.reg_alpha, self.reg_lambda, self.class_weight, self.min_split_gain = params["n_estimators"], params["learning_rate"], params["max_depth"], params["num_leaves"], params["min_child_samples"], params["subsample"], params["colsample_bytree"], params["reg_alpha"], params["reg_lambda"], params["class_weight"], params["min_split_gain"]
        self.n_feat = params["n_feat"]
        self.bool_loaded_HP = True
        
    def model_development(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        min_split_gain: float = 0.0,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        model_name: Optional[str] = None,
    ):
        if self.ranked_features is None:
            raise ValueError("Run feature_ranking() before model_development().")

        if self.bool_loaded_HP==True:
            n_estimators, learning_rate, max_depth, num_leaves, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda, class_weight, min_split_gain = self.n_estimators, self.learning_rate, self.max_depth, self.num_leaves, self.min_child_samples, self.subsample, self.colsample_bytree, self.reg_alpha, self.reg_lambda, self.class_weight, self.min_split_gain

        n_feat = self.n_feat or 64
        features_to_use = [f for f in self.ranked_features if f != "temp" and f != "class"][:n_feat]
        self._features_used = features_to_use
 
        gbm = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate = learning_rate,
            max_depth = max_depth, 
            num_leaves = num_leaves, 
            min_child_samples = min_child_samples, 
            subsample = subsample, 
            colsample_bytree = colsample_bytree,
            reg_alpha = reg_alpha, 
            reg_lambda = reg_lambda,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            random_state=random_state,
            n_jobs=-1,
        )
 
        # Combine train + val for final fit
        train_df = data_concatenater(self.train)
        if self.validation:
            val_df = data_concatenater(self.val)
            fit_df = pd.concat([train_df, val_df], ignore_index=True)
        else: 
            fit_df = train_df.copy()
 
        X = fit_df[features_to_use].values
        y = fit_df["class"].values
        gbm.fit(X, y)
        self.model = gbm

        print(f"LightGBM classifier trained on {X.shape[0]} samples, {X.shape[1]} features")

        if model_name:
            self.save_model(model_name)
 
    # ------------------------------------------------------------------ #
    #  Save / Load                                                        #
    # ------------------------------------------------------------------ #
    def save_model(self, filepath: str):
        payload = {
            "model": self.model,
            "features_used": self._features_used,
            "ranked_features": self.ranked_features,
            "n_feat": self.n_feat,
            "gbm_params": getattr(self, "gbm_params", None),
        }
        with open(filepath, "wb") as f:
            pkl.dump(payload, f)
        print(f"Model saved to {filepath}")
 
    def load_model(self, filepath: str):
        with open(filepath, "rb") as f:
            payload = pkl.load(f)
        self.model = payload["model"]
        self._features_used = payload["features_used"]
        self.ranked_features = payload.get("ranked_features", self.ranked_features)
        self.n_feat = payload.get("n_feat", self.n_feat)
        self.gbm_params = payload.get("gbm_params", None)
        print(f"Model loaded from {filepath}")
    
    @classmethod
    def load_model_only(cls, filepath: str):
        with open(filepath, "rb") as f: 
            payload = pkl.load(f)
        obj = cls.__new__(cls)
        obj.model = payload["model"]
        obj._features_used = payload.get("features_used")
        obj.ranked_features = payload.get("ranked_features")
        obj.n_feat = payload.get("n_feat")
        obj.gbm_params = payload.get("gbm_params", None)
        print(f"Model loaded from {filepath}")
        return obj
    
    def predict_model_only(self, X): 
        subset_features = self._features_used or self.ranked_features[:self.n_feat]
        X = X[subset_features].to_numpy()
        classes = self.model.predict(X)
        probs = self.model.predict_proba(X)
        return classes, probs

    def ensemble_models(self, seeds, root_path):
        # seeds is a list
        for seed in seeds:
            self.model_development(random_state=seed, model_name=root_path/f"Ensemble_GBM_{self.cbfv}_Seed{seed}.pkl")
            
    # ------------------------------------------------------------------ #
    #  Prediction + results                                               #
    # ------------------------------------------------------------------ #
    def predict(self, X):
        classes = self.model.predict(X)
        probs = self.model.predict_proba(X)
        return classes, probs


 
class Metal_Deployment: 
    def __init__(self,
                 data_path,
                 cbfv,
                 target_name,
                 random_state,
                 validation=False,):
        
        self.data_path = data_path
        self.cbfv = cbfv
        self.target_name = target_name
        self.random_state = random_state
        self.validation = validation

        Data = pd.read_csv(data_path)
        if "comp" in Data.columns:
            Data.drop("comp", axis=1, inplace=True)
        Data = Data.reset_index(drop=True)
        Feature_Columns = [col for col in Data.columns if col not in ["formula", "source", "entry", "target", "class", "atmosphere"]] 
        self.Feature_Columns = Feature_Columns

        DataToSplit = Data.copy()
        if self.validation: 
            Splitter = GroupShuffleSplit(n_splits=1, train_size=0.95, random_state=random_state)
            Train_Idx, Validation_Idx = next(
                Splitter.split(DataToSplit, groups=DataToSplit["formula"])
            )
            Train = DataToSplit.iloc[Train_Idx]
            Val = DataToSplit.iloc[Validation_Idx]
            Train = Train.drop_duplicates(subset="formula").reset_index(drop=True)
            Val = Val.drop_duplicates(subset="formula").reset_index(drop=True)
            Train, Val = Train[Train["class"]==1].reset_index(drop=True), Val[Val["class"]==1].reset_index(drop=True)
            self.train = self.data_indexer(data=Train, index=None, feature_columns=Feature_Columns)
            self.val = self.data_indexer(data=Val, index=None, feature_columns=Feature_Columns)
        else:
            DataToSplit = DataToSplit.drop_duplicates(subset="formula")
            DataToSplit = DataToSplit[DataToSplit["class"]==1].reset_index(drop=True)
            self.train = self.data_indexer(data=DataToSplit, index=None, feature_columns=Feature_Columns)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        
    def feature_ranking(self, feature_rank, feature_ranking_path=None, n_jobs=15):
        # doing feature ranking 
        if feature_rank==True:
            merged_features, per_target_lists, cnmi = RFFeatureSelector.feature_selection(self.train["features"], self.train["target"], target_name="target", n_jobs=n_jobs, random_state=42)
            Ranked_Target_List = per_target_lists["target"]
            self.ranked_features = Ranked_Target_List
            ranked_features_dict = {"ranked_features": Ranked_Target_List}
            if feature_ranking_path:
                with open(feature_ranking_path, "wb") as f:
                    pkl.dump(ranked_features_dict, f)
        else: 
            with open(feature_ranking_path, "rb") as f:
                unserialized_data = pkl.load(f)
                self.ranked_features = unserialized_data["ranked_features"]
    
    def data_indexer(self, data, index=None, feature_columns=None):
        subset = data if index is None else data.iloc[index]
        subset = subset.reset_index(drop=True)
        final_data = {
            "features": subset[feature_columns],
            "formula":  subset["formula"],
            self.target_name: pd.DataFrame({self.target_name: subset[self.target_name]}),
            "class":    pd.DataFrame({"class": subset["class"]}),
            "source":   subset["source"],
        }
        if "atmosphere" in subset.columns:
            final_data["atmosphere"] = subset["atmosphere"]
        return final_data

    def load_hp(self,
                hp_path):
        obj = pkl.load(open(hp_path, "rb"))
        params = obj["best_params"]
        cleaned_params = []
        for i in params: 
            if hasattr(i, "tolist"):
                cleaned_params.append(i.tolist())
            else:
                cleaned_params.append(i)
        n_feat, num_epochs, n_layers, architecture_style, brick_width, funnel_width, funnel_rate, lr, weight_decay, batch_size, dropout_rate, act = cleaned_params 
        architecture = architecture_generator(architecture_style, brick_width, funnel_width, n_layers, funnel_rate)

        # calling optimized properties into self
        self.n_feat = n_feat
        self.num_epochs = num_epochs
        self.architecture = architecture
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.act = act

    def model_development(self,
                          model_name,
                          seed,
                          plot=True,
                          patience_num=20,):
        _set_all_seeds(seed)
        regressor = TriCondNet.MetalPINN(
            target_name="target",
            optimal_descriptors=self.ranked_features, 
            n_feat = self.n_feat,
            architecture=self.architecture,
            act = self.act,
            dropout_rate = self.dropout_rate,
        )
        self.model_name = model_name

        if self.validation:
            history = regressor.fit(
                train_df=self.train["features"],
                train_target=self.train[self.target_name],
                val_df=self.val["features"],
                val_target=self.val[self.target_name],
                lr=self.lr,
                batch_size=self.batch_size,
                weight_decay=self.weight_decay,
                epochs=self.num_epochs,
                patience=patience_num,
                xscale="standard",
                verbose=False,
            )
        else:
            history = regressor.fit(
                train_df=self.train["features"],
                train_target=self.train[self.target_name],
                lr=self.lr,
                batch_size=self.batch_size,
                weight_decay=self.weight_decay,
                epochs=self.num_epochs,
                xscale="standard",
                verbose=False,
            )

        regressor.save(model_name)
        self.model = regressor

        # Loss curve
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(history["mae_loss"], label="Train MAE")
            if self.validation:
                plt.plot(history["mae_val_loss"], label="Val MAE")
            plt.xlabel("Epoch")
            plt.ylabel("MAE")
            plt.title("Training Curve")
            plt.legend()
            plt.grid(True)
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(f"{model_name}_trainingcurve.png", dpi=300)
            plt.show()


    def load_model(self, model_path): 
        model = TriCondNet.MetalPINN.load(model_path)
        self.model = model

    def ensemble_models(self, model_name, root_path, seeds): 
        # pass seeds as a list of seeds
        print(f"The number of models being trained is {len(seeds)}")
        for seed in seeds: 
            name = f"{model_name}_seed{seed}.pt"
            self.model_development(
                                model_name = root_path / name,
                                seed=seed,
                                plot=False)
 
class Semiconductor_Deployment: 
    def __init__(self,
                 data_path,
                 cbfv,
                 target_name,
                 random_state,
                 validation=False,):
        
        self.data_path = data_path
        self.cbfv = cbfv
        self.target_name = target_name
        self.random_state = random_state
        self.validation = validation

        Data = pd.read_csv(data_path)
        if "comp" in Data.columns:
            Data.drop("comp", axis=1, inplace=True)
        Data = Data.reset_index(drop=True)
        Feature_Columns = [col for col in Data.columns if col not in ["formula", "source", "entry", "target", "class", "atmosphere"]] 
        self.Feature_Columns = Feature_Columns

        DataToSplit = Data.copy()
        if self.validation: 
            Splitter = GroupShuffleSplit(n_splits=1, train_size=0.95, random_state=random_state)
            Train_Idx, Validation_Idx = next(
                Splitter.split(DataToSplit, groups=DataToSplit["formula"])
            )
            Train = DataToSplit.iloc[Train_Idx]
            Val = DataToSplit.iloc[Validation_Idx]
            Train = Train.drop_duplicates(subset="formula").reset_index(drop=True)
            Val = Val.drop_duplicates(subset="formula").reset_index(drop=True)
            Train, Val = Train[Train["class"]==0].reset_index(drop=True), Val[Val["class"]==0].reset_index(drop=True)
            self.train = self.data_indexer(data=Train, index=None, feature_columns=Feature_Columns)
            self.val = self.data_indexer(data=Val, index=None, feature_columns=Feature_Columns)
        else:
            DataToSplit = DataToSplit.drop_duplicates(subset="formula")
            DataToSplit = DataToSplit[DataToSplit["class"]==0].reset_index(drop=True)
            self.train = self.data_indexer(data=DataToSplit, index=None, feature_columns=Feature_Columns)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        
    def feature_ranking(self, feature_rank, feature_ranking_path=None, n_jobs=15):
        # doing feature ranking 
        if feature_rank==True:
            merged_features, per_target_lists, cnmi = RFFeatureSelector.feature_selection(self.train["features"], self.train["target"], target_name="target", n_jobs=n_jobs, random_state=42)
            Ranked_Target_List = per_target_lists["target"]
            self.ranked_features = Ranked_Target_List
            ranked_features_dict = {"ranked_features": Ranked_Target_List}
            if feature_ranking_path:
                with open(feature_ranking_path, "wb") as f:
                    pkl.dump(ranked_features_dict, f)
        else: 
            with open(feature_ranking_path, "rb") as f:
                unserialized_data = pkl.load(f)
                self.ranked_features = unserialized_data["ranked_features"]
    
    def data_indexer(self, data, index=None, feature_columns=None):
        subset = data if index is None else data.iloc[index]
        subset = subset.reset_index(drop=True)
        final_data = {
            "features": subset[feature_columns],
            "formula":  subset["formula"],
            self.target_name: pd.DataFrame({self.target_name: subset[self.target_name]}),
            "class":    pd.DataFrame({"class": subset["class"]}),
            "source":   subset["source"],
        }
        if "atmosphere" in subset.columns:
            final_data["atmosphere"] = subset["atmosphere"]
        return final_data

    def load_hp(self,
                hp_path):
        obj = pkl.load(open(hp_path, "rb"))
        params = obj["best_params"]
        cleaned_params = []
        for i in params: 
            if hasattr(i, "tolist"):
                cleaned_params.append(i.tolist())
            else:
                cleaned_params.append(i)
        n_feat, num_epochs, n_layers, architecture_style, brick_width, funnel_width, funnel_rate, lr, weight_decay, batch_size, dropout_rate, act = cleaned_params 
        architecture = architecture_generator(architecture_style, brick_width, funnel_width, n_layers, funnel_rate)

        # calling optimized properties into self
        self.n_feat = n_feat
        self.num_epochs = num_epochs
        self.architecture = architecture
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.act = act

    def model_development(self,
                          model_name,
                          seed,
                          plot=True,
                          patience_num=20,):
        _set_all_seeds(seed)
        regressor = TriCondNet.SemiconductorPINN(
            target_name="target",
            optimal_descriptors=self.ranked_features, 
            n_feat = self.n_feat,
            architecture=self.architecture,
            act = self.act,
            dropout_rate = self.dropout_rate,
        )
        self.model_name = model_name

        if self.validation:
            history = regressor.fit(
                train_df=self.train["features"],
                train_target=self.train[self.target_name],
                val_df=self.val["features"],
                val_target=self.val[self.target_name],
                lr=self.lr,
                batch_size=self.batch_size,
                weight_decay=self.weight_decay,
                epochs=self.num_epochs,
                patience=patience_num,
                xscale="standard",
                verbose=False,
            )
        else:
            history = regressor.fit(
                train_df=self.train["features"],
                train_target=self.train[self.target_name],
                lr=self.lr,
                batch_size=self.batch_size,
                weight_decay=self.weight_decay,
                epochs=self.num_epochs,
                xscale="standard",
                verbose=False,
            )

        regressor.save(model_name)
        self.model = regressor

        # Loss curve
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(history["mae_loss"], label="Train MAE")
            if self.validation:
                plt.plot(history["mae_val_loss"], label="Val MAE")
            plt.xlabel("Epoch")
            plt.ylabel("MAE")
            plt.title("Training Curve")
            plt.legend()
            plt.grid(True)
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(f"{model_name}_trainingcurve.png", dpi=300)
            plt.show()


    def load_model(self, model_path): 
        model = TriCondNet.SemiconductorPINN.load(model_path)
        self.model = model

    def ensemble_models(self, model_name, root_path, seeds): 
        # pass seeds as a list of seeds
        print(f"The number of models being trained is {len(seeds)}")
        for seed in seeds: 
            name = f"{model_name}_seed{seed}.pt"
            self.model_development(
                                model_name = root_path / name,
                                seed=seed,
                                plot=False)

class TriCondNet_Ensemble_Wrapper:
    def __init__(self,
                 Classifier_Root_Path,
                 Semiconductor_Root_Path,
                 Metal_Root_Path,
                 seeds_list):
        self.num_seeds = seeds_list
        self.classifier_path = Classifier_Root_Path
        self.metal_path = Metal_Root_Path
        self.semi_path = Semiconductor_Root_Path

    def data_indexer(self, data, index=None, feature_columns=None):
        if index is None:
            final_data = {"features": data.reset_index(drop=True)[feature_columns],
                        "formula": data.reset_index(drop=True)["formula"]}
        else:
            final_data = {"features": data.iloc[index].reset_index(drop=True)[feature_columns],
                        "formula": data.iloc[index].reset_index(drop=True)["formula"],}
        return final_data

    def load_data(self, test_df):
        Feature_Columns = [col for col in test_df.columns if col not in ["formula", "source", "entry", "target"]]
        test_dict = self.data_indexer(test_df, index=None, feature_columns=Feature_Columns)
        self.test = test_dict
        return self.test

    def rf_classifier_sim(self, model, data):
        output = model.predict_model_only(data)
        probs = output[1]
        p = probs[:,1]
        p_metal = pd.Series(p)
        p_clipped = np.clip(p, 1e-12, 1 - 1e-12)
        entropy = -1*(p_clipped*np.log2(p_clipped) + (1-p_clipped)*np.log2(1-p_clipped))
        return probs, entropy, p_metal

    def ensemble(self, classifier_model_name, semiconductor_model_name, metal_model_name):

        test_data = self.test["features"]
        classifier_input = test_data.drop("temp", axis=1).copy()
        n_test = len(test_data)

        p_metal_per_seed = []
        cond_pred_per_seed = []
        
        # per-seed activation energies (Ea) from the semiconductor branch, for reporting
        Ea_matrix = np.full((len(self.num_seeds), n_test), np.nan)

        for seed_idx, seed in enumerate(self.num_seeds):
            classifier_model = GBM_ClassifierDeployment.load_model_only(
                filepath=self.classifier_path / f"{classifier_model_name}_Seed{seed}.pkl"
            )
            class_preds, class_probs = classifier_model.predict_model_only(classifier_input)
            p_metal_per_seed.append(class_probs[:, 1])

            class_preds_series = pd.Series(class_preds)
            Metal_Indices = class_preds_series[class_preds_series == 1].index.tolist()
            Semi_Indices = class_preds_series[class_preds_series == 0].index.tolist()
            Metal_TestData = self.test["features"].iloc[Metal_Indices]
            Semiconductor_TestData = self.test["features"].iloc[Semi_Indices]

            semiconductor_model = TriCondNet.SemiconductorPINN.load(self.semi_path / f"{semiconductor_model_name}_seed{seed}.pt")
            metal_model = TriCondNet.MetalPINN.load(self.metal_path / f"{metal_model_name}_seed{seed}.pt")
            with torch.no_grad():
                Metal_PredictedConductivity = metal_model.predict(Metal_TestData).flatten() if len(Metal_Indices) > 0 else np.array([])
                Semiconductor_PredictedConductivity = semiconductor_model.predict(Semiconductor_TestData).flatten() if len(Semi_Indices) > 0 else np.array([])
                Ea_matrix[seed_idx, Semi_Indices] = np.asarray(semiconductor_model.Test_Predicted_Ea).flatten()

            cond_vec = np.empty(n_test)
            if len(Metal_Indices) > 0:
                cond_vec[Metal_Indices] = Metal_PredictedConductivity
            if len(Semi_Indices) > 0:
                cond_vec[Semi_Indices] = Semiconductor_PredictedConductivity
            cond_pred_per_seed.append(cond_vec)

        stacked_p_metal = np.stack(p_metal_per_seed, axis=0)
        mean_p_metal = np.mean(stacked_p_metal, axis=0)
        p_clipped = np.clip(mean_p_metal, 1e-12, 1 - 1e-12)
        entropy = -1*(p_clipped*np.log2(p_clipped) + (1-p_clipped)*np.log2(1-p_clipped))
        final_class = np.where(mean_p_metal >= 0.5, "Metal", "Semiconductor")

        stacked_predictions = np.stack(cond_pred_per_seed, axis=0)
        mean_conductivity = np.mean(stacked_predictions, axis=0)
        std_conductivity = np.std(stacked_predictions, axis=0)
        predictions_Scm = 10**(mean_conductivity)
        SEM_log = std_conductivity / ((len(self.num_seeds))**(0.5))
        SEM_linear = np.log(10) * predictions_Scm * SEM_log

        # mean activation energy (Ea) and its uncertainty across seeds, semiconductors only
        mean_Ea, std_Ea = np.nanmean(Ea_matrix, axis=0), np.nanstd(Ea_matrix, axis=0)
        mask = (final_class == "Semiconductor")
        SEM_Ea = std_Ea / ((len(self.num_seeds))**(0.5))
        self.Ea, self.Ea_unc, self.Ea_formulae = mean_Ea[mask], SEM_Ea[mask], self.test["formula"][mask].reset_index(drop=True)

        final_df = pd.DataFrame({
            "formula": self.test["formula"].tolist(),
            "class": final_class,
            "class_uncertainty": entropy,
            "p_metal": mean_p_metal,
            "pred_conductivity": predictions_Scm,
            "regressor_uncertainty": SEM_linear,
        })
        return final_df
        

        
