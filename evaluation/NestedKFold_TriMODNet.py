import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, r2_score, classification_report, balanced_accuracy_score, matthews_corrcoef, roc_auc_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from pymatgen.core import Composition
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics.pairwise import rbf_kernel
from CBFV import composition
import matplotlib.pyplot as plt
from itertools import permutations
from sklearn.model_selection import GroupKFold
from scipy.stats import gaussian_kde

from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent.parent))
sys.path.insert(0, str(Path.cwd().parent))
sys.path.insert(0, str(Path.cwd().parent.parent.parent))

from evaluation.preprocessing import FeatureSelector
from models.TriMODNet_Implementation import MODNet_Classifier, MODNet_Regressor
from hyperparameterisation import modnet_BO

class TriMODNet_NestedKFold_Classifier:
    def __init__(self, data_path, cbfv, n_inner, n_outer, num_opt_trials=2, num_epoch=500, patience=50, target_name="class", random_state=0):
        self.cbfv = cbfv 
        self.target_name = target_name 
        self.random_state = random_state
        self.n_inner = n_inner
        self.n_outer = n_outer
        self.num_opt_trials = num_opt_trials    

        self.num_epoch = num_epoch
        self.patience = patience

        data = pd.read_csv(data_path)
        data = data.drop("temp", axis=1)
        data = data.drop_duplicates(subset="formula")
        data = data.reset_index(drop=True)

        self.feature_columns = [c for c in data.columns
                                if c not in ["formula", "source", "entry", "target", "class"]]

        self.X      = data[self.feature_columns]
        self.y      = data[target_name]
        self.source = data["source"]
        self.entry  = data["entry"]
        self.groups = data["formula"]
    
    def run_nested_cv(self, results_dir="nested_cv_results"):

        groups = self.groups
        gkf = GroupKFold(n_splits=self.n_outer)
        all_hyperparameters = {}

        all_test_balanced_accuracy = []
        all_test_mcc = []
        all_test_metal_precision, all_test_metal_recall, all_test_metal_f1 = [], [], []
        all_test_semi_precision,  all_test_semi_recall,  all_test_semi_f1  = [], [], []
        all_y_true = []
        all_y_pred = []
        parity_all_regressor_predictions = []
        parity_all_regressor_true = []

        results_dir = Path(results_dir) / "TriMODNet" / self.cbfv / "classifier"
        results_dir.mkdir(parents=True, exist_ok=True)

        for fold_k, (train_idx, test_idx) in enumerate(gkf.split(self.X, self.y, groups)):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
            group_train, group_test = groups.iloc[train_idx], groups.iloc[test_idx]
            source_outer = self.source.iloc[train_idx]
            entry_outer  = self.entry.iloc[train_idx]

            Splitter = GroupShuffleSplit(n_splits=1, train_size=0.9, random_state=self.random_state)
            train_postval_idx, val_idx = next(Splitter.split(X_train, groups=group_train))
            X_train, X_val = X_train.iloc[train_postval_idx], X_train.iloc[val_idx]
            y_train, y_val = y_train.iloc[train_postval_idx], y_train.iloc[val_idx]
            group_train, group_val = group_train.iloc[train_postval_idx], group_train.iloc[val_idx]
            source_train = source_outer.iloc[train_postval_idx]
            entry_train  = entry_outer.iloc[train_postval_idx]

            merged_features, per_target_lists, cnmi = FeatureSelector.feature_selection(
                X_train, y_train.to_frame(),
                n_jobs=40, random_state=self.random_state,
            )
            Ranked_Target_List = per_target_lists[self.target_name]
            self.ranked_features = Ranked_Target_List

            hyperparameters = self.inner_optimizer(X_train, y_train, group_train, source_train, entry_train)
            all_hyperparameters[fold_k] = hyperparameters

            load_params = hyperparameters["best_params"]
            cleaned_params = []
            for i in load_params: 
                if hasattr(i, "tolist"):
                    cleaned_params.append(i.tolist())
                else:
                    cleaned_params.append(i)
            n_feat, num_epochs, n_neurons_first, fraction1, fraction2, fraction3, lr, batch_size, act = cleaned_params
            num_epochs = int(num_epochs)
            n_neurons_first = int(n_neurons_first)
            fraction1, fraction2, fraction3 = float(fraction1), float(fraction2), float(fraction3)
            w0 = n_neurons_first
            w1 = max(1, int(w0 * fraction1))
            w2 = max(1, int(w1 * fraction2))
            w3 = max(1, int(w2 * fraction3))
            architecture = ([w0], [w1], [w2], [w3])


            def _wrap(X, y, g):
                return {
                    "features": X.reset_index(drop=True),
                    self.target_name: y.reset_index(drop=True),
                    "formula": g.reset_index(drop=True),
                }
            train = _wrap(X_train, y_train, group_train)
            val   = _wrap(X_val,   y_val,   group_val)
            test  = _wrap(X_test,  y_test,  group_test)

            classifier = MODNet_Classifier(train_input=train, val_input=val, feature_rank=False, ranked_features=self.ranked_features, class_name="class")
            classifier.classifier(n_feat=n_feat, architecture=architecture, lr=lr, batch_size=batch_size, act=act, epochs=num_epochs, patience=None)
            model_state = classifier.model

            pred_test = classifier.predict(test=test)
            Y_Class_Target = test[self.target_name]
            Y_Formula_Target = test["formula"]
            Y_ConfMatrix_Target = pd.DataFrame({"formula": Y_Formula_Target, self.target_name: Y_Class_Target})
            Y_Class_Pred = pred_test[self.target_name]
            Y_ConfMatrix_Pred = pd.DataFrame({"formula": Y_Formula_Target, self.target_name: Y_Class_Pred})
            cm = confusion_matrix(Y_ConfMatrix_Target["class"], Y_ConfMatrix_Pred["class"])

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["semiconductor", "metal"])
            disp.plot(cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.show()
            print(classification_report(Y_ConfMatrix_Target["class"], Y_ConfMatrix_Pred["class"], 
                                    target_names=["semiconductor", "metal"]))
            print(f"Balanced Accuracy: {balanced_accuracy_score(Y_ConfMatrix_Target['class'], Y_ConfMatrix_Pred['class']):.4f}")
            print(f"MCC: {matthews_corrcoef(Y_ConfMatrix_Target['class'], Y_ConfMatrix_Pred['class']):.4f}")


            report = classification_report(Y_ConfMatrix_Target["class"], Y_ConfMatrix_Pred["class"],
                                           target_names=["semiconductor", "metal"],
                                           output_dict=True, zero_division=0)

            test_balanced_accuracy = balanced_accuracy_score(Y_ConfMatrix_Target["class"], Y_ConfMatrix_Pred["class"])
            test_mcc               = matthews_corrcoef(Y_ConfMatrix_Target["class"], Y_ConfMatrix_Pred["class"])
            metal_precision = report["metal"]["precision"]
            metal_recall    = report["metal"]["recall"]
            metal_f1        = report["metal"]["f1-score"]
            semi_precision  = report["semiconductor"]["precision"]
            semi_recall     = report["semiconductor"]["recall"]
            semi_f1         = report["semiconductor"]["f1-score"]

            all_test_balanced_accuracy.append(test_balanced_accuracy)
            all_test_mcc.append(test_mcc)
            all_test_metal_precision.append(metal_precision)
            all_test_metal_recall.append(metal_recall)
            all_test_metal_f1.append(metal_f1)
            all_test_semi_precision.append(semi_precision)
            all_test_semi_recall.append(semi_recall)
            all_test_semi_f1.append(semi_f1)
            all_y_true.extend(Y_Class_Target.tolist())
            all_y_pred.extend(Y_Class_Pred.tolist())

            fold_payload = {
                "fold_k": fold_k,
                "best_hp": hyperparameters,
                "metrics": {
                    "balanced_accuracy": test_balanced_accuracy,
                    "mcc": test_mcc,
                    "metal_precision": metal_precision,
                    "metal_recall":    metal_recall,
                    "metal_f1":        metal_f1,
                    "semi_precision":  semi_precision,
                    "semi_recall":     semi_recall,
                    "semi_f1":         semi_f1,
                    "y_true_vals": Y_Class_Target.tolist(),
                    "y_pred_vals": Y_Class_Pred.tolist(),

                },
                "report_full": report,
                "n_test": int(len(y_test)),
                "n_train": int(len(y_train)),
            }
            fold_path = results_dir / f"classifier_fold_{fold_k}.pkl"
            with open(fold_path, "wb") as f:
                pkl.dump(fold_payload, f)
            print(f"[fold {fold_k}] saved -> {fold_path}")

        def msd(arr):
            return f"{np.mean(arr):.4f} +/- {np.std(arr):.4f}"

        print("\n========= Nested CV summary =========")
        print(f"Balanced Accuracy:  {msd(all_test_balanced_accuracy)}")
        print(f"MCC:                {msd(all_test_mcc)}")
        print(f"Metal Precision:    {msd(all_test_metal_precision)}")
        print(f"Metal Recall:       {msd(all_test_metal_recall)}")
        print(f"Metal F1:           {msd(all_test_metal_f1)}")
        print(f"Semi  Precision:    {msd(all_test_semi_precision)}")
        print(f"Semi  Recall:       {msd(all_test_semi_recall)}")
        print(f"Semi  F1:           {msd(all_test_semi_f1)}")

        self.fold_results = {
            "balanced_accuracy": all_test_balanced_accuracy,
            "mcc":               all_test_mcc,
            "metal_precision":   all_test_metal_precision,
            "metal_recall":      all_test_metal_recall,
            "metal_f1":          all_test_metal_f1,
            "semi_precision":    all_test_semi_precision,
            "semi_recall":       all_test_semi_recall,
            "semi_f1":           all_test_semi_f1,
            "all_y_true":        all_y_true,
            "all_y_pred":        all_y_pred,

        }
        self.all_hyperparameters = all_hyperparameters

        with open(results_dir / "classifier_summary.pkl", "wb") as f:
            pkl.dump({
                "fold_results": self.fold_results,
                "all_hyperparameters": all_hyperparameters,
            }, f)
        print(f"Summary saved -> {results_dir / 'classifier_summary.pkl'}")

        return self.fold_results, all_hyperparameters

        
    def inner_optimizer(self, X, y, group, source, entry):
        data_dict = {
            "features": X.reset_index(drop=True),
            self.target_name: pd.DataFrame({self.target_name: y.reset_index(drop=True)}),
            "formula": pd.DataFrame({"formula": group.reset_index(drop=True)}),
            "source": pd.DataFrame({"source": source.reset_index(drop=True)}),
        }
        self.data_dict = data_dict
        classifier_bo = modnet_BO.modnet_class_opt(
            data_dict=data_dict, ranked_features=self.ranked_features, num_trials=self.num_opt_trials,
        )
        classifier_bo.run_optimization()
        return {"best_params": list(classifier_bo.search_result.x)}


class TriMODNet_NestedKFold_SemiRegressor:
    def __init__(self, data_path, cbfv, n_inner, n_outer, num_opt_trials=2, num_epoch=500, patience=50, target_name="target", random_state=0):
        self.cbfv = cbfv 
        self.target_name = target_name 
        self.random_state = random_state
        self.n_inner = n_inner
        self.n_outer = n_outer
        self.num_opt_trials = num_opt_trials    

        self.num_epoch = num_epoch
        self.patience = patience

        data = pd.read_csv(data_path)
        data = data[data["class"]==0]
        data = data.reset_index(drop=True)

        self.feature_columns = [c for c in data.columns
                                if c not in ["formula", "source", "entry", "target", "class"]]

        self.X      = data[self.feature_columns]
        self.y      = data[target_name]
        self.source = data["source"]
        self.entry  = data["entry"]
        self.groups = data["formula"]

    def run_nested_cv(self, results_dir="nested_cv_results"):

        groups = self.groups
        gkf = GroupKFold(n_splits=self.n_outer)
        all_hyperparameters = {}

        all_train_mae, all_train_r2 = [], []
        all_test_mae,  all_test_r2  = [], []

        results_dir = Path(results_dir) / "TriMODNet" / self.cbfv / "semi_regressor"
        results_dir.mkdir(parents=True, exist_ok=True)

        for fold_k, (train_idx, test_idx) in enumerate(gkf.split(self.X, self.y, groups)):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
            group_train, group_test = groups.iloc[train_idx], groups.iloc[test_idx]
            source_train = self.source.iloc[train_idx]
            entry_train  = self.entry.iloc[train_idx]

            merged_features, per_target_lists, cnmi = FeatureSelector.feature_selection(
                X_train, y_train.to_frame(),
                n_jobs=40, random_state=self.random_state,
            )
            Ranked_Target_List = per_target_lists[self.target_name]
            self.ranked_features = Ranked_Target_List

            hyperparameters = self.inner_optimizer(X_train, y_train, group_train, source_train, entry_train)
            all_hyperparameters[fold_k] = hyperparameters

            load_params = hyperparameters["best_params"]
            cleaned_params = []
            for i in load_params: 
                if hasattr(i, "tolist"):
                    cleaned_params.append(i.tolist())
                else:
                    cleaned_params.append(i)
            n_feat, num_epochs, n_neurons_first, fraction1, fraction2, fraction3, lr, batch_size, act = cleaned_params
            num_epochs = int(num_epochs)
            n_neurons_first = int(n_neurons_first)
            fraction1, fraction2, fraction3 = float(fraction1), float(fraction2), float(fraction3)
            w0 = n_neurons_first
            w1 = max(1, int(w0 * fraction1))
            w2 = max(1, int(w1 * fraction2))
            w3 = max(1, int(w2 * fraction3))
            architecture = ([w0], [w1], [w2], [w3])


            def _wrap(X, y, g):
                return {
                    "features": X.reset_index(drop=True),
                    self.target_name: y.reset_index(drop=True),
                    "formula": g.reset_index(drop=True),
                }
            train = _wrap(X_train, y_train, group_train)
            test  = _wrap(X_test,  y_test,  group_test)

            regressor = MODNet_Regressor(train_input=train, feature_rank=False, ranked_features=self.ranked_features, target_name="target")
            regressor.regressor(n_feat=n_feat, architecture=architecture, lr=lr, batch_size=batch_size, act=act, epochs=num_epochs, patience=None)
            model_state = regressor.model

            arr_pred_train = regressor.predict(train)
            arr_pred_test  = regressor.predict(test)
            train_mae = mean_absolute_error(train[self.target_name].values, arr_pred_train)
            train_r2  = r2_score(train[self.target_name].values, arr_pred_train)
            test_mae  = mean_absolute_error(test[self.target_name].values,  arr_pred_test)
            test_r2   = r2_score(test[self.target_name].values,  arr_pred_test)
            print("This is the training metrics:", train_mae, train_r2)
            print("This is the testing metrics:", test_mae, test_r2)


            all_train_mae.append(train_mae)
            all_train_r2.append(train_r2)
            all_test_mae.append(test_mae)
            all_test_r2.append(test_r2)

            fold_payload = {
                "fold_k": fold_k,
                "best_hp": hyperparameters,
                "metrics": {
                    "train_mae": train_mae,
                    "train_r2": train_r2,
                    "test_mae":     test_mae,
                    "test_r2":         test_r2,
                },
                "n_test": int(len(y_test)),
                "n_train": int(len(y_train)),
            }
            fold_path = results_dir / f"SemiRegressor_fold_{fold_k}.pkl"
            with open(fold_path, "wb") as f:
                pkl.dump(fold_payload, f)
            print(f"[fold {fold_k}] saved -> {fold_path}")

        def msd(arr):
            return f"{np.mean(arr):.4f} +/- {np.std(arr):.4f}"

        print("\n========= Nested CV summary =========")
        print(f"Train MAE:  {msd(all_train_mae)}")
        print(f"Train R2:                {msd(all_train_r2)}")
        print(f"Test MAE:    {msd(all_test_mae)}")
        print(f"Test R2:       {msd(all_test_r2)}")

        self.fold_results = {
            "train_mae": all_train_mae,
            "train_r2":               all_train_r2,
            "test_mae":   all_test_mae,
            "test_r2":      all_test_r2,
        }
        self.all_hyperparameters = all_hyperparameters

        with open(results_dir / "semiconductor_summary.pkl", "wb") as f:
            pkl.dump({
                "fold_results": self.fold_results,
                "all_hyperparameters": all_hyperparameters,
            }, f)
        print(f"Summary saved -> {results_dir / 'semiconductor_summary.pkl'}")

        return self.fold_results, all_hyperparameters

        
    def inner_optimizer(self, X, y, group, source, entry):
        data_dict = {
            "features": X.reset_index(drop=True),
            self.target_name: pd.DataFrame({self.target_name: y.reset_index(drop=True)}),
            "formula": pd.DataFrame({"formula": group.reset_index(drop=True)}),
            "source": pd.DataFrame({"source": source.reset_index(drop=True)}),
        }
        self.data_dict = data_dict
        regressor_bo = modnet_BO.modnet_regressor_opt(
            data_dict=data_dict,
            ranked_features=self.ranked_features,
            num_trials=self.num_opt_trials,
            target_name=self.target_name,
            keys=['features', 'formula', self.target_name, 'source'],
        )
        regressor_bo.run_optimization()
        return {"best_params": list(regressor_bo.search_result.x)}



class TriMODNet_NestedKFold_MetalRegressor:
    def __init__(self, data_path, cbfv, n_inner, n_outer, num_opt_trials=2, num_epoch=500, patience=50, target_name="target", random_state=0):
        self.cbfv = cbfv 
        self.target_name = target_name 
        self.random_state = random_state
        self.n_inner = n_inner
        self.n_outer = n_outer
        self.num_opt_trials = num_opt_trials    

        self.num_epoch = num_epoch
        self.patience = patience

        data = pd.read_csv(data_path)
        data = data[data["class"]==1]
        data = data.reset_index(drop=True)

        self.feature_columns = [c for c in data.columns
                                if c not in ["formula", "source", "entry", "target", "class"]]

        self.X      = data[self.feature_columns]
        self.y      = data[target_name]
        self.source = data["source"]
        self.entry  = data["entry"]
        self.groups = data["formula"]

    def run_nested_cv(self, results_dir="nested_cv_results"):

        groups = self.groups
        gkf = GroupKFold(n_splits=self.n_outer)
        all_hyperparameters = {}

        all_train_mae, all_train_r2 = [], []
        all_test_mae,  all_test_r2  = [], []

        results_dir = Path(results_dir) / "TriMODNet" / self.cbfv / "metal_regressor"
        results_dir.mkdir(parents=True, exist_ok=True)

        for fold_k, (train_idx, test_idx) in enumerate(gkf.split(self.X, self.y, groups)):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
            group_train, group_test = groups.iloc[train_idx], groups.iloc[test_idx]
            source_outer = self.source.iloc[train_idx]
            entry_outer  = self.entry.iloc[train_idx]

            Splitter = GroupShuffleSplit(n_splits=1, train_size=0.9, random_state=self.random_state)
            train_postval_idx, val_idx = next(Splitter.split(X_train, groups=group_train))
            X_train, X_val = X_train.iloc[train_postval_idx], X_train.iloc[val_idx]
            y_train, y_val = y_train.iloc[train_postval_idx], y_train.iloc[val_idx]
            group_train, group_val = group_train.iloc[train_postval_idx], group_train.iloc[val_idx]
            source_train = source_outer.iloc[train_postval_idx]
            entry_train  = entry_outer.iloc[train_postval_idx]

            merged_features, per_target_lists, cnmi = FeatureSelector.feature_selection(
                X_train, y_train.to_frame(),
                n_jobs=40, random_state=self.random_state,
            )
            Ranked_Target_List = per_target_lists[self.target_name]
            self.ranked_features = Ranked_Target_List

            hyperparameters = self.inner_optimizer(X_train, y_train, group_train, source_train, entry_train)
            all_hyperparameters[fold_k] = hyperparameters

            load_params = hyperparameters["best_params"]
            cleaned_params = []
            for i in load_params: 
                if hasattr(i, "tolist"):
                    cleaned_params.append(i.tolist())
                else:
                    cleaned_params.append(i)
            n_feat, num_epochs, n_neurons_first, fraction1, fraction2, fraction3, lr, batch_size, act = cleaned_params
            num_epochs = int(num_epochs)
            n_neurons_first = int(n_neurons_first)
            fraction1, fraction2, fraction3 = float(fraction1), float(fraction2), float(fraction3)
            w0 = n_neurons_first
            w1 = max(1, int(w0 * fraction1))
            w2 = max(1, int(w1 * fraction2))
            w3 = max(1, int(w2 * fraction3))
            architecture = ([w0], [w1], [w2], [w3])


            def _wrap(X, y, g):
                return {
                    "features": X.reset_index(drop=True),
                    self.target_name: y.reset_index(drop=True),
                    "formula": g.reset_index(drop=True),
                }
            train = _wrap(X_train, y_train, group_train)
            val   = _wrap(X_val,   y_val,   group_val)
            test  = _wrap(X_test,  y_test,  group_test)

            regressor = MODNet_Regressor(train_input=train, val_input=val, feature_rank=False, ranked_features=self.ranked_features, target_name="target")
            regressor.regressor(n_feat=n_feat, architecture=architecture, lr=lr, batch_size=batch_size, act=act, epochs=num_epochs, patience=None)
            model_state = regressor.model

            arr_pred_train = regressor.predict(train)
            arr_pred_test  = regressor.predict(test)
            train_mae = mean_absolute_error(train[self.target_name].values, arr_pred_train)
            train_r2  = r2_score(train[self.target_name].values, arr_pred_train)
            test_mae  = mean_absolute_error(test[self.target_name].values,  arr_pred_test)
            test_r2   = r2_score(test[self.target_name].values,  arr_pred_test)
            print("This is the training metrics:", train_mae, train_r2)
            print("This is the testing metrics:", test_mae, test_r2)


            all_train_mae.append(train_mae)
            all_train_r2.append(train_r2)
            all_test_mae.append(test_mae)
            all_test_r2.append(test_r2)

            fold_payload = {
                "fold_k": fold_k,
                "best_hp": hyperparameters,
                "metrics": {
                    "train_mae": train_mae,
                    "train_r2": train_r2,
                    "test_mae":     test_mae,
                    "test_r2":         test_r2,
                },
                "n_test": int(len(y_test)),
                "n_train": int(len(y_train)),
            }
            fold_path = results_dir / f"MetalRegressor_fold_{fold_k}.pkl"
            with open(fold_path, "wb") as f:
                pkl.dump(fold_payload, f)
            print(f"[fold {fold_k}] saved -> {fold_path}")

        def msd(arr):
            return f"{np.mean(arr):.4f} +/- {np.std(arr):.4f}"

        print("\n========= Nested CV summary =========")
        print(f"Train MAE:  {msd(all_train_mae)}")
        print(f"Train R2:                {msd(all_train_r2)}")
        print(f"Test MAE:    {msd(all_test_mae)}")
        print(f"Test R2:       {msd(all_test_r2)}")

        self.fold_results = {
            "train_mae": all_train_mae,
            "train_r2":               all_train_r2,
            "test_mae":   all_test_mae,
            "test_r2":      all_test_r2,
        }
        self.all_hyperparameters = all_hyperparameters

        with open(results_dir / "metal_summary.pkl", "wb") as f:
            pkl.dump({
                "fold_results": self.fold_results,
                "all_hyperparameters": all_hyperparameters,
            }, f)
        print(f"Summary saved -> {results_dir / 'metal_summary.pkl'}")

        return self.fold_results, all_hyperparameters

        
    def inner_optimizer(self, X, y, group, source, entry):
        data_dict = {
            "features": X.reset_index(drop=True),
            self.target_name: pd.DataFrame({self.target_name: y.reset_index(drop=True)}),
            "formula": pd.DataFrame({"formula": group.reset_index(drop=True)}),
            "source": pd.DataFrame({"source": source.reset_index(drop=True)}),
        }
        self.data_dict = data_dict
        regressor_bo = modnet_BO.modnet_regressor_opt(
            data_dict=data_dict,
            ranked_features=self.ranked_features,
            num_trials=self.num_opt_trials,
            target_name=self.target_name,
            keys=['features', 'formula', self.target_name, 'source'],
        )
        regressor_bo.run_optimization()
        return {"best_params": list(regressor_bo.search_result.x)}


class TriMODNet_NestedKFold_Pipeline:

    def __init__(self, data_path, cbfv, n_inner, n_outer, num_opt_trials=2, target_name="target", random_state=0, filetype="csv"):
        self.cbfv = cbfv
        self.target_name = target_name
        self.random_state = random_state
        self.n_inner = n_inner
        self.n_outer = n_outer
        self.num_opt_trials = num_opt_trials

        if filetype=="csv":
            data = pd.read_csv(data_path)
        elif filetype=="excel":
            data = pd.read_excel(data_path)
        data = data.reset_index(drop=True)

        self.feature_columns = [c for c in data.columns
                                if c not in ["formula", "source", "entry", "target", "class"]]
        self.X           = data[self.feature_columns]
        self.X_formula   = data["formula"]
        self.y_class     = data["class"]
        self.y_log10cond = data["target"]
        self.source      = data["source"]
        self.entry       = data["entry"]
        self.groups      = data["formula"]

    def run_nested_cv(self, results_dir="nested_cv_results_PIPELINE", retrain=True):
        groups = self.groups
        gkf = GroupKFold(n_splits=self.n_outer, shuffle=True, random_state=self.random_state)
        all_hyperparameters = {}

        all_test_balanced_accuracy = []
        all_test_mcc = []
        all_test_metal_precision, all_test_metal_recall, all_test_metal_f1 = [], [], []
        all_test_semi_precision,  all_test_semi_recall,  all_test_semi_f1  = [], [], []
        all_metal_train_mae, all_metal_test_mae = [], []
        all_metal_train_r2,  all_metal_test_r2  = [], []
        all_semi_train_mae,  all_semi_test_mae  = [], []
        all_semi_train_r2,   all_semi_test_r2   = [], []
        all_pipeline_train_mae, all_pipeline_test_mae = [], []
        all_pipeline_train_r2,  all_pipeline_test_r2  = [], []
        all_pipeline_decomposition = []
        all_y_true, all_y_pred, all_test_formulae = [], [], []

        parity_all_regressor_predictions = []
        parity_all_regressor_true = []

        # pooled (out-of-fold) prediction accumulators for oracle-routed heads
        all_metal_train_predictions, all_metal_test_predictions = [], []
        all_metal_train_groundtruth, all_metal_test_groundtruth = [], []
        all_semi_train_predictions,  all_semi_test_predictions  = [], []
        all_semi_train_groundtruth,  all_semi_test_groundtruth  = [], []

        results_dir = Path(results_dir) / "TriMODNet"
        results_dir.mkdir(parents=True, exist_ok=True)

        def _slice(df, idx):
            return df.iloc[idx].reset_index(drop=True)

        def _wrap(X, y, g, key):
            return {
                "features": X.reset_index(drop=True),
                key:        y.reset_index(drop=True),
                "formula":  g.reset_index(drop=True),
            }

        def _unpack_modnet_hp(load_params):
            cleaned = []
            for i in load_params:
                cleaned.append(i.tolist() if hasattr(i, "tolist") else i)
            n_feat, num_epochs, n_neurons_first, fraction1, fraction2, fraction3, lr, batch_size, act = cleaned
            n_feat = min(int(n_feat), len(self.ranked_features))
            num_epochs = int(num_epochs)
            n_neurons_first = int(n_neurons_first)
            f1, f2, f3 = float(fraction1), float(fraction2), float(fraction3)
            w0 = n_neurons_first
            w1 = max(1, int(w0 * f1))
            w2 = max(1, int(w1 * f2))
            w3 = max(1, int(w2 * f3))
            architecture = ([w0], [w1], [w2], [w3])
            return n_feat,  num_epochs, architecture, lr, batch_size, act

        for fold_k, (train_idx, test_idx) in enumerate(gkf.split(self.X, self.y_class, groups)):

            X_Train, X_Test = _slice(self.X, train_idx), _slice(self.X, test_idx)
            Y_Train_cl, Y_Test_cl = _slice(self.y_class, train_idx), _slice(self.y_class, test_idx)
            X_Train_Formula, X_Test_Formula = _slice(self.X_formula, train_idx), _slice(self.X_formula, test_idx)
            Y_Train_Log10Cond, Y_Test_Log10Cond = _slice(self.y_log10cond, train_idx), _slice(self.y_log10cond, test_idx)
            Group_Train, Group_Test = _slice(self.groups, train_idx), _slice(self.groups, test_idx)
            Source_Train, Source_Test = _slice(self.source, train_idx), _slice(self.source, test_idx)
            Entry_Train, Entry_Test = _slice(self.entry, train_idx), _slice(self.entry, test_idx)

            X_Class_Train_full_notemp = X_Train.drop("temp", axis=1)
            X_Class_Test_full_notemp  = X_Test.drop("temp", axis=1)
            train_unique_pos = X_Train_Formula.drop_duplicates().index
            test_unique_pos  = X_Test_Formula.drop_duplicates().index
            X_Class_Train = X_Class_Train_full_notemp.loc[train_unique_pos].reset_index(drop=True)
            X_Class_Test  = X_Class_Test_full_notemp.loc[test_unique_pos].reset_index(drop=True)
            Y_Class_Train = Y_Train_cl.loc[train_unique_pos].reset_index(drop=True)
            Y_Class_Test  = Y_Test_cl.loc[test_unique_pos].reset_index(drop=True)
            Group_Class_Train = X_Train_Formula.loc[train_unique_pos].reset_index(drop=True)
            Group_Class_Test  = X_Test_Formula.loc[test_unique_pos].reset_index(drop=True)
            Source_Class_Train = Source_Train.loc[train_unique_pos].reset_index(drop=True)
            Entry_Class_Train  = Entry_Train.loc[train_unique_pos].reset_index(drop=True)

            metal_pkl = results_dir / f"MetalRegressor_fold_{fold_k}.pkl"
            metal_cached = metal_pkl.exists()

            Metal_Mask_Train = (Y_Train_cl == 1).values
            Metal_Mask_Test  = (Y_Test_cl  == 1).values

            X_Train_Metal = X_Train.loc[Metal_Mask_Train].reset_index(drop=True)
            X_Test_Metal  = X_Test.loc[Metal_Mask_Test].reset_index(drop=True)
            Y_Train_Metal_Log10Cond = Y_Train_Log10Cond.loc[Metal_Mask_Train].reset_index(drop=True)
            Y_Test_Metal_Log10Cond  = Y_Test_Log10Cond.loc[Metal_Mask_Test].reset_index(drop=True)
            Group_Train_Metal = Group_Train.loc[Metal_Mask_Train].reset_index(drop=True)
            Group_Test_Metal  = Group_Test.loc[Metal_Mask_Test].reset_index(drop=True)
            Source_Train_Metal = Source_Train.loc[Metal_Mask_Train].reset_index(drop=True)
            Entry_Train_Metal  = Entry_Train.loc[Metal_Mask_Train].reset_index(drop=True)


            merged_features, per_target_lists, cnmi = FeatureSelector.feature_selection(
                X_Train_Metal, Y_Train_Metal_Log10Cond.to_frame(),
                n_jobs=20, random_state=self.random_state,
            )
            Ranked_Target_List = per_target_lists[self.target_name]
            self.ranked_features = Ranked_Target_List

            if metal_cached:
                print(f"[Metal fold {fold_k}] cached HP, refitting (retrain=True)")
                with open(metal_pkl, "rb") as f:
                    cached = pkl.load(f)
                hyperparameters = cached["best_hp"]
            else:
                hyperparameters = self.metal_inner_optimizer(X_Train_Metal, Y_Train_Metal_Log10Cond, Group_Train_Metal, Source_Train_Metal, Entry_Train_Metal)
            all_hyperparameters[fold_k] = {"metal": hyperparameters}

            n_feat, num_epochs, architecture, lr, batch_size, act = _unpack_modnet_hp(hyperparameters["best_params"])

            train_metal = _wrap(X_Train_Metal, Y_Train_Metal_Log10Cond, Group_Train_Metal, self.target_name)
            test_metal  = _wrap(X_Test_Metal,  Y_Test_Metal_Log10Cond,  Group_Test_Metal,  self.target_name)

            metal_regressor = MODNet_Regressor(
                train_input=train_metal,
                feature_rank=False, ranked_features=self.ranked_features,
                target_name=self.target_name,
            )
            metal_regressor.regressor(
                n_feat=n_feat, architecture=architecture, lr=lr,
                batch_size=batch_size, act=act,
                epochs=num_epochs, patience=None,
            )

            arr_pred_train = metal_regressor.predict(train_metal)
            arr_pred_test  = metal_regressor.predict(test_metal)

            train_mae = mean_absolute_error(Y_Train_Metal_Log10Cond.values, arr_pred_train)
            train_r2  = r2_score(Y_Train_Metal_Log10Cond.values, arr_pred_train)
            test_mae  = mean_absolute_error(Y_Test_Metal_Log10Cond.values,  arr_pred_test)
            test_r2   = r2_score(Y_Test_Metal_Log10Cond.values,  arr_pred_test)
            print(f"[Metal fold {fold_k}] train MAE={train_mae:.4f} R2={train_r2:.4f}")
            print(f"[Metal fold {fold_k}] test  MAE={test_mae:.4f} R2={test_r2:.4f}")

            all_metal_train_mae.append(train_mae)
            all_metal_train_r2.append(train_r2)
            all_metal_test_mae.append(test_mae)
            all_metal_test_r2.append(test_r2)

            # pooled outer fold
            all_metal_train_predictions.extend(np.asarray(arr_pred_train).ravel())
            all_metal_test_predictions.extend(np.asarray(arr_pred_test).ravel())
            all_metal_train_groundtruth.extend(Y_Train_Metal_Log10Cond.values)
            all_metal_test_groundtruth.extend(Y_Test_Metal_Log10Cond.values)


            metal_fold_payload = {
                "fold_k": fold_k,
                "best_hp": hyperparameters,
                "metrics": {"train_mae": train_mae, "train_r2": train_r2, "test_mae": test_mae, "test_r2": test_r2},
                "n_test":  int(len(Y_Test_Metal_Log10Cond)),
                "n_train": int(len(Y_Train_Metal_Log10Cond)),
            }
            with open(metal_pkl, "wb") as f:
                pkl.dump(metal_fold_payload, f)
            print(f"[Metal fold {fold_k}] saved -> {metal_pkl}")

            semi_pkl = results_dir / f"SemiRegressor_fold_{fold_k}.pkl"
            semi_cached = semi_pkl.exists()

            Semi_Mask_Train = (Y_Train_cl == 0).values
            Semi_Mask_Test  = (Y_Test_cl  == 0).values

            X_Train_Semi = X_Train.loc[Semi_Mask_Train].reset_index(drop=True)
            X_Test_Semi  = X_Test.loc[Semi_Mask_Test].reset_index(drop=True)
            Y_Train_Semi_Log10Cond = Y_Train_Log10Cond.loc[Semi_Mask_Train].reset_index(drop=True)
            Y_Test_Semi_Log10Cond  = Y_Test_Log10Cond.loc[Semi_Mask_Test].reset_index(drop=True)
            Group_Train_Semi = Group_Train.loc[Semi_Mask_Train].reset_index(drop=True)
            Group_Test_Semi  = Group_Test.loc[Semi_Mask_Test].reset_index(drop=True)
            Source_Train_Semi = Source_Train.loc[Semi_Mask_Train].reset_index(drop=True)
            Entry_Train_Semi  = Entry_Train.loc[Semi_Mask_Train].reset_index(drop=True)


            merged_features, per_target_lists, cnmi = FeatureSelector.feature_selection(
                X_Train_Semi, Y_Train_Semi_Log10Cond.to_frame(),
                n_jobs=10, random_state=self.random_state,
            )
            Ranked_Target_List = per_target_lists[self.target_name]
            self.ranked_features = Ranked_Target_List

            if semi_cached:
                print(f"[Semi fold {fold_k}] cached HP, refitting (retrain=True)")
                with open(semi_pkl, "rb") as f:
                    cached = pkl.load(f)
                hyperparameters = cached["best_hp"]
            else:
                hyperparameters = self.semi_inner_optimizer(X_Train_Semi, Y_Train_Semi_Log10Cond, Group_Train_Semi, Source_Train_Semi, Entry_Train_Semi)
            all_hyperparameters[fold_k]["semi"] = hyperparameters

            n_feat, num_epochs, architecture, lr, batch_size, act = _unpack_modnet_hp(hyperparameters["best_params"])

            train_semi = _wrap(X_Train_Semi, Y_Train_Semi_Log10Cond, Group_Train_Semi, self.target_name)
            test_semi  = _wrap(X_Test_Semi,  Y_Test_Semi_Log10Cond,  Group_Test_Semi,  self.target_name)

            semi_regressor = MODNet_Regressor(
                train_input=train_semi, 
                feature_rank=False, ranked_features=self.ranked_features,
                target_name=self.target_name,
            )
            semi_regressor.regressor(
                n_feat=n_feat, architecture=architecture, lr=lr,
                batch_size=batch_size, act=act,
                epochs=num_epochs, patience=None,
            )
            arr_pred_train = semi_regressor.predict(train_semi)
            arr_pred_test  = semi_regressor.predict(test_semi)

            train_mae = mean_absolute_error(Y_Train_Semi_Log10Cond.values, arr_pred_train)
            train_r2  = r2_score(Y_Train_Semi_Log10Cond.values, arr_pred_train)
            test_mae  = mean_absolute_error(Y_Test_Semi_Log10Cond.values,  arr_pred_test)
            test_r2   = r2_score(Y_Test_Semi_Log10Cond.values,  arr_pred_test)
            print(f"[Semi fold {fold_k}] train MAE={train_mae:.4f} R2={train_r2:.4f}")
            print(f"[Semi fold {fold_k}] test  MAE={test_mae:.4f} R2={test_r2:.4f}")

            all_semi_train_mae.append(train_mae)
            all_semi_train_r2.append(train_r2)
            all_semi_test_mae.append(test_mae)
            all_semi_test_r2.append(test_r2)

            # pooled outer fold
            all_semi_train_predictions.extend(np.asarray(arr_pred_train).ravel())
            all_semi_test_predictions.extend(np.asarray(arr_pred_test).ravel())
            all_semi_train_groundtruth.extend(Y_Train_Semi_Log10Cond.values)
            all_semi_test_groundtruth.extend(Y_Test_Semi_Log10Cond.values)

            semi_fold_payload = {
                "fold_k": fold_k,
                "best_hp": hyperparameters,
                "metrics": {"train_mae": train_mae, "train_r2": train_r2, "test_mae": test_mae, "test_r2": test_r2},
                "n_test":  int(len(Y_Test_Semi_Log10Cond)),
                "n_train": int(len(Y_Train_Semi_Log10Cond)),
            }
            with open(semi_pkl, "wb") as f:
                pkl.dump(semi_fold_payload, f)
            print(f"[Semi fold {fold_k}] saved -> {semi_pkl}")

            clf_pkl = results_dir / f"classifier_fold_{fold_k}.pkl"
            clf_cached = clf_pkl.exists()

            All_Test_Class_Formulae = Group_Test.copy()

            merged_features, per_target_lists, cnmi = FeatureSelector.feature_selection(
                X_Class_Train, Y_Class_Train.to_frame(name="class"),
                n_jobs=10, random_state=self.random_state,
            )
            Ranked_Target_List = per_target_lists["class"]
            self.ranked_features = Ranked_Target_List

            if clf_cached:
                print(f"[Classifier fold {fold_k}] cached HP, refitting (retrain=True)")
                with open(clf_pkl, "rb") as f:
                    cached = pkl.load(f)
                hyperparameters = cached["best_hp"]
            else:
                hyperparameters = self.class_inner_optimizer(X_Class_Train, Y_Class_Train, Group_Class_Train, Source_Class_Train, Entry_Class_Train)
            all_hyperparameters[fold_k]["classifier"] = hyperparameters

            n_feat, num_epochs, architecture, lr, batch_size, act = _unpack_modnet_hp(hyperparameters["best_params"])

            train_class = _wrap(X_Class_Train, Y_Class_Train, Group_Class_Train, "class")
            test_class  = _wrap(X_Class_Test,  Y_Class_Test,  Group_Class_Test,  "class")

            classifier = MODNet_Classifier(
                train_input=train_class, 
                feature_rank=False, ranked_features=self.ranked_features,
                class_name="class",
            )
            classifier.classifier(
                n_feat=n_feat, architecture=architecture, lr=lr,
                batch_size=batch_size, act=act,
                epochs=num_epochs, patience=None,
            )

            pred_test_dict = classifier.predict(test=test_class)
            y_test_arr = Y_Class_Test.values
            test_pred  = pd.Series(pred_test_dict["class"]).values

            test_mcc = matthews_corrcoef(y_test_arr, test_pred)
            test_acc = balanced_accuracy_score(y_test_arr, test_pred)

            cm = confusion_matrix(y_test_arr, test_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["semiconductor", "metal"])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix - fold {fold_k}")
            plt.show()

            report = classification_report(y_test_arr, test_pred, target_names=["semiconductor", "metal"], output_dict=True, zero_division=0)
            print(classification_report(y_test_arr, test_pred, target_names=["semiconductor", "metal"], zero_division=0))

            pred_train_dict = classifier.predict(test=train_class)
            TRAIN_CLASS_PREDS_ = pd.Series(pred_train_dict["class"]).values
            TEST_CLASS_PREDS_  = test_pred

            train_form_to_pred = dict(zip(Group_Class_Train.values, TRAIN_CLASS_PREDS_))
            test_form_to_pred  = dict(zip(Group_Class_Test.values,  TEST_CLASS_PREDS_))

            train_full_pred = Group_Train.map(train_form_to_pred).values
            test_full_pred  = Group_Test.map(test_form_to_pred).values

            train_metal_routed = (train_full_pred == 1)
            train_semi_routed  = (train_full_pred == 0)
            test_metal_routed  = (test_full_pred  == 1)
            test_semi_routed   = (test_full_pred  == 0)

            Metal_TrainData = X_Train.loc[train_metal_routed].reset_index(drop=True)
            Semi_TrainData  = X_Train.loc[train_semi_routed].reset_index(drop=True)
            Metal_TestData  = X_Test.loc[test_metal_routed].reset_index(drop=True)
            Semi_TestData   = X_Test.loc[test_semi_routed].reset_index(drop=True)

            Metal_Train_Truth = Y_Train_Log10Cond.loc[train_metal_routed].reset_index(drop=True).values
            Semi_Train_Truth  = Y_Train_Log10Cond.loc[train_semi_routed].reset_index(drop=True).values
            Metal_Test_Truth  = Y_Test_Log10Cond.loc[test_metal_routed].reset_index(drop=True).values
            Semi_Test_Truth   = Y_Test_Log10Cond.loc[test_semi_routed].reset_index(drop=True).values

            Metal_Train_Group = Group_Train.loc[train_metal_routed].reset_index(drop=True)
            Semi_Train_Group  = Group_Train.loc[train_semi_routed].reset_index(drop=True)
            Metal_Test_Group  = Group_Test.loc[test_metal_routed].reset_index(drop=True)
            Semi_Test_Group   = Group_Test.loc[test_semi_routed].reset_index(drop=True)

            def _empty_or_pred(reg, X, y_truth, g):
                if len(X) == 0:
                    return np.array([])
                wrapped = {
                    "features": X,
                    self.target_name: pd.Series(y_truth).reset_index(drop=True),
                    "formula": g,
                }
                return np.asarray(reg.predict(wrapped)).ravel()
            
            Metal_Train_Pred = _empty_or_pred(metal_regressor, Metal_TrainData, Metal_Train_Truth, Metal_Train_Group)
            Semi_Train_Pred  = _empty_or_pred(semi_regressor,  Semi_TrainData,  Semi_Train_Truth,  Semi_Train_Group)
            Metal_Test_Pred  = _empty_or_pred(metal_regressor, Metal_TestData,  Metal_Test_Truth,  Metal_Test_Group)
            Semi_Test_Pred   = _empty_or_pred(semi_regressor,  Semi_TestData,   Semi_Test_Truth,   Semi_Test_Group)

            y_train_true_e2e = np.concatenate([Metal_Train_Truth, Semi_Train_Truth])
            y_train_pred_e2e = np.concatenate([Metal_Train_Pred,  Semi_Train_Pred])
            y_test_true_e2e  = np.concatenate([Metal_Test_Truth,  Semi_Test_Truth])
            y_test_pred_e2e  = np.concatenate([Metal_Test_Pred,   Semi_Test_Pred])

            pipeline_train_mae = mean_absolute_error(y_train_true_e2e, y_train_pred_e2e)
            pipeline_train_r2  = r2_score(y_train_true_e2e, y_train_pred_e2e)
            pipeline_test_mae  = mean_absolute_error(y_test_true_e2e, y_test_pred_e2e)
            pipeline_test_r2   = r2_score(y_test_true_e2e, y_test_pred_e2e)
            print(f"[PIPELINE fold {fold_k}] train MAE={pipeline_train_mae:.4f} R2={pipeline_train_r2:.4f}")
            print(f"[PIPELINE fold {fold_k}] test  MAE={pipeline_test_mae:.4f} R2={pipeline_test_r2:.4f}")

            parity_all_regressor_predictions.extend(y_test_pred_e2e)
            parity_all_regressor_true.extend(y_test_true_e2e)


            test_form_to_true = dict(zip(Group_Class_Test.values, Y_Class_Test.values))
            test_full_true_cls = Group_Test.map(test_form_to_true).values

            buckets = {
                "TM_PM": (test_full_true_cls == 1) & test_metal_routed,
                "TM_PS": (test_full_true_cls == 1) & test_semi_routed,
                "TS_PM": (test_full_true_cls == 0) & test_metal_routed,
                "TS_PS": (test_full_true_cls == 0) & test_semi_routed,
            }
            y_hat_test = np.full(len(Y_Test_Log10Cond), np.nan)
            y_hat_test[test_metal_routed] = Metal_Test_Pred
            y_hat_test[test_semi_routed]  = Semi_Test_Pred
            y_true_test = Y_Test_Log10Cond.values

            bucket_metrics = {}
            for name, m in buckets.items():
                n = int(m.sum())
                bucket_metrics[name] = {
                    "n":   n,
                    "mae": float(mean_absolute_error(y_true_test[m], y_hat_test[m])) if n else None,
                    "r2":  float(r2_score(y_true_test[m], y_hat_test[m])) if n > 1 else None,
                }
            print(f"[PIPELINE fold {fold_k}] decomposition: {bucket_metrics}")

            all_pipeline_train_mae.append(pipeline_train_mae)
            all_pipeline_train_r2.append(pipeline_train_r2)
            all_pipeline_test_mae.append(pipeline_test_mae)
            all_pipeline_test_r2.append(pipeline_test_r2)
            all_pipeline_decomposition.append(bucket_metrics)

            pipeline_fold_payload = {
                "fold_k": fold_k,
                "metrics": {
                    "train_mae": pipeline_train_mae, "train_r2": pipeline_train_r2,
                    "test_mae":  pipeline_test_mae,  "test_r2":  pipeline_test_r2,
                },
                "decomposition": bucket_metrics,
                "n_test":  int(len(Y_Test_Log10Cond)),
                "n_train": int(len(Y_Train_Log10Cond)),
            }
            with open(results_dir / f"Pipeline_fold_{fold_k}_rs{self.random_state}.pkl", "wb") as f:
                pkl.dump(pipeline_fold_payload, f)

            metal_precision = report["metal"]["precision"]
            metal_recall    = report["metal"]["recall"]
            metal_f1        = report["metal"]["f1-score"]
            semi_precision  = report["semiconductor"]["precision"]
            semi_recall     = report["semiconductor"]["recall"]
            semi_f1         = report["semiconductor"]["f1-score"]

            all_test_balanced_accuracy.append(test_acc)
            all_test_mcc.append(test_mcc)
            all_test_metal_precision.append(metal_precision)
            all_test_metal_recall.append(metal_recall)
            all_test_metal_f1.append(metal_f1)
            all_test_semi_precision.append(semi_precision)
            all_test_semi_recall.append(semi_recall)
            all_test_semi_f1.append(semi_f1)
            all_y_true.extend(np.asarray(y_test_arr).tolist())
            all_y_pred.extend(np.asarray(test_pred).tolist())
            all_test_formulae.extend(All_Test_Class_Formulae.tolist())

            clf_fold_payload = {
                "fold_k": fold_k,
                "best_hp": hyperparameters,
                "metrics": {
                    "balanced_accuracy": test_acc, "mcc": test_mcc,
                    "metal_precision": metal_precision, "metal_recall": metal_recall, "metal_f1": metal_f1,
                    "semi_precision":  semi_precision,  "semi_recall":  semi_recall,  "semi_f1":  semi_f1,
                    "y_true_vals": np.asarray(y_test_arr).tolist(),
                    "y_pred_vals": np.asarray(test_pred).tolist(),
                },
                "report_full": report,
                "n_test":  int(len(Y_Class_Test)),
                "n_train": int(len(Y_Class_Train)),
            }
            with open(clf_pkl, "wb") as f:
                pkl.dump(clf_fold_payload, f)
            print(f"[Classifier fold {fold_k}] saved -> {clf_pkl}")

        def msd(arr):
            return f"{np.mean(arr):.4f} +/- {np.std(arr):.4f}"

        print("\n========= Nested CV summary =========")
        print("--- Classifier ---")
        print(f"Balanced Accuracy:  {msd(all_test_balanced_accuracy)}")
        print(f"MCC:                {msd(all_test_mcc)}")
        print(f"Metal Precision:    {msd(all_test_metal_precision)}")
        print(f"Metal Recall:       {msd(all_test_metal_recall)}")
        print(f"Metal F1:           {msd(all_test_metal_f1)}")
        print(f"Semi  Precision:    {msd(all_test_semi_precision)}")
        print(f"Semi  Recall:       {msd(all_test_semi_recall)}")
        print(f"Semi  F1:           {msd(all_test_semi_f1)}")
        print("--- Metal regressor (oracle routing) ---")
        print(f"Train MAE: {msd(all_metal_train_mae)} | Train R2: {msd(all_metal_train_r2)}")
        print(f"Test  MAE: {msd(all_metal_test_mae)}  | Test  R2: {msd(all_metal_test_r2)}")
        print("--- Semi regressor (oracle routing) ---")
        print(f"Train MAE: {msd(all_semi_train_mae)} | Train R2: {msd(all_semi_train_r2)}")
        print(f"Test  MAE: {msd(all_semi_test_mae)}  | Test  R2: {msd(all_semi_test_r2)}")
        print("--- End-to-end pipeline (classifier-routed) ---")
        print(f"Train MAE: {msd(all_pipeline_train_mae)} | Train R2: {msd(all_pipeline_train_r2)}")
        print(f"Test  MAE: {msd(all_pipeline_test_mae)}  | Test  R2: {msd(all_pipeline_test_r2)}")

        # ----- pooled out-of-fold metrics (single number, no per-fold averaging) -----
        print("\n========= Pooled out-of-fold (single R2 over all test predictions) =========")
        print("--- Metal regressor (oracle routing) - pooled ---")
        print(f"Test  MAE: {mean_absolute_error(all_metal_test_groundtruth, all_metal_test_predictions):.4f}"
              f"  | Test  R2: {r2_score(all_metal_test_groundtruth, all_metal_test_predictions):.4f}")
        print("--- Semi regressor (oracle routing) - pooled ---")
        print(f"Test  MAE: {mean_absolute_error(all_semi_test_groundtruth, all_semi_test_predictions):.4f}"
              f"  | Test  R2: {r2_score(all_semi_test_groundtruth, all_semi_test_predictions):.4f}")

        self.fold_results = {
            "balanced_accuracy": all_test_balanced_accuracy,
            "mcc":               all_test_mcc,
            "metal_precision":   all_test_metal_precision,
            "metal_recall":      all_test_metal_recall,
            "metal_f1":          all_test_metal_f1,
            "semi_precision":    all_test_semi_precision,
            "semi_recall":       all_test_semi_recall,
            "semi_f1":           all_test_semi_f1,
            "all_y_true":        all_y_true,
            "all_y_pred":        all_y_pred,
            "all_test_formulae": all_test_formulae,
            "metal_train_mae":   all_metal_train_mae,
            "metal_train_r2":    all_metal_train_r2,
            "metal_test_mae":    all_metal_test_mae,
            "metal_test_r2":     all_metal_test_r2,
            "semi_train_mae":    all_semi_train_mae,
            "semi_train_r2":     all_semi_train_r2,
            "semi_test_mae":     all_semi_test_mae,
            "semi_test_r2":      all_semi_test_r2,
            "pipeline_train_mae": all_pipeline_train_mae,
            "pipeline_train_r2":  all_pipeline_train_r2,
            "pipeline_test_mae":  all_pipeline_test_mae,
            "pipeline_test_r2":   all_pipeline_test_r2,
            "pipeline_decomposition": all_pipeline_decomposition,
        }
        self.all_hyperparameters = all_hyperparameters

        self.parity_true =  parity_all_regressor_true
        self.parity_pred = parity_all_regressor_predictions

        with open(results_dir / "pipeline_summary.pkl", "wb") as f:
            pkl.dump({"fold_results": self.fold_results, "all_hyperparameters": all_hyperparameters}, f)
        print(f"Summary saved -> {results_dir / 'pipeline_summary.pkl'}")


        return self.fold_results, all_hyperparameters

    def plot_parity(self, title="Parity plot", save_path=None, show=True, dpi=600):

        y_true = np.asarray(self.parity_true, dtype=float)
        y_pred = np.asarray(self.parity_pred, dtype=float)

        if y_true.size == 0 or y_pred.size == 0:
            raise ValueError("Parity data is empty. Ensure regressors ran and predictions were collected.")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Parity data shape mismatch: true {y_true.shape} vs pred {y_pred.shape}")

        mae = mean_absolute_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)

        low  = min(np.min(y_true), np.min(y_pred))
        high = max(np.max(y_true), np.max(y_pred))
        pad  = 0.05 * (high - low) if high > low else 0.5
        lims = (low - pad, high + pad)

        rc = {
            "font.family":         "serif",
            "font.serif":          ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset":    "stix",
            "axes.linewidth":      0.8,
            "axes.edgecolor":      "0.25",
            "axes.labelcolor":     "0.15",
            "axes.labelsize":      9.5,
            "axes.labelpad":       3.5,
            "axes.titlesize":      10.5,
            "axes.titlepad":       5.0,
            "axes.titleweight":    "regular",
            "xtick.labelsize":     8,
            "ytick.labelsize":     8,
            "xtick.color":         "0.25",
            "ytick.color":         "0.25",
            "xtick.direction":     "in",
            "ytick.direction":     "in",
            "xtick.top":           True,
            "ytick.right":         True,
            "xtick.major.size":    3.2,
            "ytick.major.size":    3.2,
            "xtick.major.width":   0.8,
            "ytick.major.width":   0.8,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.size":    1.8,
            "ytick.minor.size":    1.8,
            "xtick.minor.width":   0.55,
            "ytick.minor.width":   0.55,
            "savefig.bbox":        "tight",
            "savefig.dpi":         dpi,
            "pdf.fonttype":        42,
            "ps.fonttype":         42,
        }

        color = "#4393c3"

        with plt.rc_context(rc):
            fig = plt.figure(figsize=(3.8, 3.8), dpi=dpi)
            gs = fig.add_gridspec(
                2, 2,
                width_ratios=[5, 1],
                height_ratios=[1, 5],
                hspace=0.04,
                wspace=0.04,
            )
            ax_main  = fig.add_subplot(gs[1, 0])
            ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
            ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

            ax_main.set_axisbelow(True)
            ax_main.grid(True, which="major", linestyle="-", linewidth=0.35, color="0.88")
            ax_main.scatter(
                y_true, y_pred,
                s=6, alpha=0.45,
                color=color,
                edgecolors="none",
                zorder=3,
                rasterized=True,
            )
            ax_main.plot(lims, lims, linestyle="--", color="#b2182b", linewidth=1.0, zorder=4)
            ax_main.set_xlim(lims)
            ax_main.set_ylim(lims)
            ax_main.set_aspect("equal", adjustable="box")
            ax_main.set_xlabel(r"Actual log$_{10}\,\sigma$ (S cm$^{-1}$)")
            ax_main.set_ylabel(r"Pred. log$_{10}\,\sigma$ (S cm$^{-1}$)")

            stats = f"MAE = {mae:.3f}\n$R^{{2}}$ = {r2:.3f}"
            ax_main.text(
                0.04, 0.97, stats,
                transform=ax_main.transAxes,
                va="top", ha="left",
                fontsize=7.2,
                linespacing=1.6,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="0.78",
                    linewidth=0.5,
                    alpha=0.92,
                ),
            )

            grid = np.linspace(lims[0], lims[1], 400)
            kde_x = gaussian_kde(y_true, bw_method="scott")
            dens_x = kde_x(grid)
            ax_top.fill_between(grid, dens_x, alpha=0.35, color=color, linewidth=0)
            ax_top.plot(grid, dens_x, color=color, linewidth=0.8)
            ax_top.set_xlim(lims)
            ax_top.set_ylim(bottom=0)
            ax_top.axis("off")
            if title:
                ax_top.set_title(title, color="0.1", pad=4)

            kde_y = gaussian_kde(y_pred, bw_method="scott")
            dens_y = kde_y(grid)
            ax_right.fill_betweenx(grid, dens_y, alpha=0.35, color=color, linewidth=0)
            ax_right.plot(dens_y, grid, color=color, linewidth=0.8)
            ax_right.set_ylim(lims)
            ax_right.set_xlim(left=0)
            ax_right.axis("off")

            if save_path:
                save_path = Path(save_path)
                fig.savefig(save_path)
                if save_path.suffix.lower() != ".pdf":
                    fig.savefig(save_path.with_suffix(".pdf"))
            if show:
                plt.show()

        return fig, ax_main

    def metal_inner_optimizer(self, X, y, group, source, entry):
        data_dict = {
            "features": X.reset_index(drop=True),
            self.target_name: pd.DataFrame({self.target_name: y.reset_index(drop=True)}),
            "formula": pd.DataFrame({"formula": group.reset_index(drop=True)}),
            "source":  pd.DataFrame({"source":  source.reset_index(drop=True)}),
        }
        self.data_dict = data_dict
        regressor_bo = modnet_BO.modnet_regressor_opt(
            data_dict=data_dict,
            ranked_features=self.ranked_features,
            num_trials=self.num_opt_trials,
            target_name=self.target_name,
            keys=['features', 'formula', self.target_name, 'source'],
        )
        regressor_bo.run_optimization()
        return {"best_params": list(regressor_bo.search_result.x)}

    def semi_inner_optimizer(self, X, y, group, source, entry):
        data_dict = {
            "features": X.reset_index(drop=True),
            self.target_name: pd.DataFrame({self.target_name: y.reset_index(drop=True)}),
            "formula": pd.DataFrame({"formula": group.reset_index(drop=True)}),
            "source":  pd.DataFrame({"source":  source.reset_index(drop=True)}),
        }
        self.data_dict = data_dict
        regressor_bo = modnet_BO.modnet_regressor_opt(
            data_dict=data_dict,
            ranked_features=self.ranked_features,
            num_trials=self.num_opt_trials,
            target_name=self.target_name,
            keys=['features', 'formula', self.target_name, 'source'],
        )
        regressor_bo.run_optimization()
        return {"best_params": list(regressor_bo.search_result.x)}

    def class_inner_optimizer(self, X, y, group, source, entry):
        data_dict = {
            "features": X.reset_index(drop=True),
            "class":    pd.DataFrame({"class": y.reset_index(drop=True)}),
            "formula":  pd.DataFrame({"formula": group.reset_index(drop=True)}),
            "source":   pd.DataFrame({"source":  source.reset_index(drop=True)}),
        }
        self.data_dict = data_dict
        classifier_bo = modnet_BO.modnet_class_opt(
            data_dict=data_dict, ranked_features=self.ranked_features, num_trials=self.num_opt_trials,
        )
        classifier_bo.run_optimization()
        return {"best_params": list(classifier_bo.search_result.x)}



            




