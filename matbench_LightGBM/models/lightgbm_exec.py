import copy
import pickle as pkl
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    f1_score
)
from lightgbm import LGBMClassifier
from matbench.bench import MatbenchBenchmark
from .preprocessing import RFFeatureSelector
from pathlib import Path
from CBFV import composition

def CBFV_transform(cbfv, train_formula_col):
    df_copy = train_formula_col.copy()
    df_copy["target"] = 0
    X, y, formulae, skipped = composition.generate_features(df_copy,
                                                            elem_prop=cbfv,
                                                            drop_duplicates=False,
                                                            extend_features=True,
                                                            sum_feat=True)
    return X

class LightGBM_MatBench:
    def __init__(
        self,
        cbfv: str,
        class_name: str = "class",
        formula_name: str = "formula",
        n_inner: int = 5,
        num_opt_trials: int = 50,
        random_state: int = 18012019,
    ):
        self.class_name = class_name
        self.target_name = class_name
        self.cbfv = cbfv
        self.MATBENCH_SEED = 18012019
        self.random_state = random_state
        self.n_inner = n_inner
        self.num_opt_trials = num_opt_trials
        self.TIA = False
        self.bool_loaded_HP = False


    def run_matbench_benchmark(self, results_dir="matbench_results"):
        all_test_balanced_accuracy = []
        all_test_mcc = []
        all_test_f1  = []
        all_test_metal_precision, all_test_metal_recall, all_test_metal_f1 = [], [], []
        all_test_nonmetal_precision,  all_test_nonmetal_recall,  all_test_nonmetal_f1  = [], [], []
        all_y_true = []
        all_y_pred = []

        all_hyperparameters = {}

        fold_aucs = []

        results_dir = Path.cwd () / results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "models").mkdir(parents=True, exist_ok=True)
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.MATBENCH_SEED)
        mb = MatbenchBenchmark(autoload=False, subset=["matbench_expt_is_metal"])
        task = mb.matbench_expt_is_metal
        task.load()
        fold_k = 0
        for fold in task.folds:
            fold_k += 1
            train_inputs, train_outputs = task.get_train_and_val_data(fold)
            train_formula = pd.DataFrame(train_inputs.values, columns=["formula"])
            X_train = CBFV_transform(self.cbfv, train_formula)
            y_train = pd.Series(train_outputs.values.ravel().astype(int), name=self.class_name)

            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            test_formula = pd.DataFrame(test_inputs.values, columns=["formula"])
            X_test = CBFV_transform(self.cbfv, test_formula)
            y_test = pd.Series(test_outputs.values.ravel().astype(int), name=self.class_name)

            print("checking overlap")
            overlap = set(train_formula["formula"]) & set(test_formula["formula"])
            print(fold_k, len(overlap), "of", len(test_formula))

            per_target_lists = RFFeatureSelector.feature_selection(
                df_feat=X_train,
                df_targets=y_train.to_frame(),
                target_name=self.target_name,
                mode="classification",
                random_state=self.random_state,
            )
            Ranked_Target_List = per_target_lists[self.target_name]
            self.ranked_features = Ranked_Target_List
            X_train = X_train[self.ranked_features]

            hyperparameters = self.inner_optimizer(X_train, y_train)
            all_hyperparameters[fold_k] = hyperparameters
            best_hp = hyperparameters
            n_feat  = hyperparameters["n_feat"]
            features_to_use = [f for f in self.ranked_features][:n_feat]

            clf = LGBMClassifier(
                n_estimators=best_hp["n_estimators"],
                learning_rate=best_hp["learning_rate"],
                num_leaves=best_hp["num_leaves"],
                max_depth=best_hp["max_depth"],
                min_child_samples=best_hp["min_child_samples"],
                feature_fraction=best_hp["feature_fraction"],
                bagging_fraction=best_hp["bagging_fraction"],
                bagging_freq=best_hp["bagging_freq"],
                reg_alpha=best_hp["reg_alpha"],
                reg_lambda=best_hp["reg_lambda"],
                random_state=self.random_state,
                n_jobs=14,
                verbose=-1,
            )
            X_clf_Input = X_train[features_to_use].values
            y_clf_Input = y_train.values
            clf.fit(X_clf_Input, y_clf_Input)

            print(f"LightGBM trained on {X_clf_Input.shape[0]} samples, {X_clf_Input.shape[1]} features")

            test_pred  = clf.predict(X_test[features_to_use].values)
            test_proba = clf.predict_proba(X_test[features_to_use].values)[:, 1]
            test_mcc = matthews_corrcoef(y_test, test_pred)
            test_acc = balanced_accuracy_score(y_test, test_pred)
            test_auc = roc_auc_score(y_test, test_proba)
            test_f1 = f1_score(y_test, test_pred)

            cm = confusion_matrix(y_test, test_pred)

            report = classification_report(y_test, test_pred, target_names=["non-metal", "metal"], output_dict=True)
            print(classification_report(y_test, test_pred, target_names=["non-metal", "metal"]))

            metal_precision = report["metal"]["precision"]
            metal_recall    = report["metal"]["recall"]
            metal_f1        = report["metal"]["f1-score"]
            nonmetal_precision  = report["non-metal"]["precision"]
            nonmetal_recall     = report["non-metal"]["recall"]
            nonmetal_f1         = report["non-metal"]["f1-score"]

            all_test_balanced_accuracy.append(test_acc)
            all_test_mcc.append(test_mcc)
            all_test_metal_precision.append(metal_precision)
            all_test_f1.append(test_f1)
            all_test_metal_recall.append(metal_recall)
            all_test_metal_f1.append(metal_f1)
            all_test_nonmetal_precision.append(nonmetal_precision)
            all_test_nonmetal_recall.append(nonmetal_recall)
            all_test_nonmetal_f1.append(nonmetal_f1)
            fold_aucs.append(test_auc)
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(test_pred.tolist())
            print(f"[fold {fold_k}] ROC-AUC = {test_auc:.4f}")

            fold_payload = {
                "fold_k": fold_k,
                "best_hp": hyperparameters,
                "confusion_matrix": cm.tolist(),
                "metrics": {
                    "balanced_accuracy": test_acc,
                    "mcc": test_mcc,
                    "f1": test_f1,
                    "metal_precision": metal_precision,
                    "metal_recall":    metal_recall,
                    "metal_f1":        metal_f1,
                    "nonmetal_precision":  nonmetal_precision,
                    "nonmetal_recall":     nonmetal_recall,
                    "nonmetal_f1":         nonmetal_f1,
                    "roc_auc":         test_auc,
                    "y_true_vals": y_test.tolist(),
                    "y_pred_vals": test_pred.tolist(),

                },
                "report_full": report,
                "n_test": int(len(y_test)),
                "n_train": int(len(y_train)),
            }
            fold_path = results_dir / f"classifier_fold_{fold_k}.pkl"
            model_path = results_dir / "models" / f"classifier_model_{fold_k}.pkl"
            with open(fold_path, "wb") as f:
                pkl.dump(fold_payload, f)
            with open(model_path, "wb") as f:
                pkl.dump(clf, f)

        def msd(arr):
            return f"{np.mean(arr):.4f} +/- {np.std(arr):.4f}"

        print("\n========= Nested CV summary =========")
        print(f"ROC-AUC:            {msd(fold_aucs)}")
        print(f"Balanced Accuracy:  {msd(all_test_balanced_accuracy)}")
        print(f"MCC:                {msd(all_test_mcc)}")
        print(f"F1:                 {msd(all_test_f1)}")
        print(f"Metal Precision:    {msd(all_test_metal_precision)}")
        print(f"Metal Recall:       {msd(all_test_metal_recall)}")
        print(f"Metal F1:           {msd(all_test_metal_f1)}")
        print(f"Non-metal Precision:{msd(all_test_nonmetal_precision)}")
        print(f"Non-metal Recall:   {msd(all_test_nonmetal_recall)}")
        print(f"Non-metal F1:       {msd(all_test_nonmetal_f1)}")

        self.fold_results = {
            "roc_auc":             fold_aucs,
            "balanced_accuracy":   all_test_balanced_accuracy,
            "mcc":                 all_test_mcc,
            "f1":                  all_test_f1,
            "metal_precision":     all_test_metal_precision,
            "metal_recall":        all_test_metal_recall,
            "metal_f1":            all_test_metal_f1,
            "nonmetal_precision":  all_test_nonmetal_precision,
            "nonmetal_recall":     all_test_nonmetal_recall,
            "nonmetal_f1":         all_test_nonmetal_f1,
            "all_y_true":          all_y_true,
            "all_y_pred":          all_y_pred,
        }
        self.all_hyperparameters = all_hyperparameters

        with open(results_dir / "classifier_summary.pkl", "wb") as f:
            pkl.dump({
                "fold_results": self.fold_results,
                "all_hyperparameters": all_hyperparameters,
            }, f)
        print("Summary saved to classifier_summary.pkl")

        return self.fold_results, all_hyperparameters


    def inner_optimizer(self, X, y, scoring="roc_auc", n_jobs=None, verbose=1):
        n_jobs=14
        pct_options = [0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        max_feat = len(self.ranked_features)
        n_feat_options = [round(pct * max_feat) for pct in pct_options]
        best_n_feat = None
        best_params = None
        best_score = -np.inf
        param_distributions = {
            "n_estimators": [200, 300, 400, 500],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "num_leaves": [15, 31, 63, 127],
            "max_depth": [-1, 6, 10, 20],
            "min_child_samples": [2, 5, 10, 20],
            "feature_fraction": [0.5, 0.7, 0.9, 1.0],
            "bagging_fraction": [0.6, 0.8, 1.0],
            "bagging_freq": [0, 5],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [0.0, 0.1, 1.0],
        }
        cv  = StratifiedKFold(n_splits=self.n_inner, shuffle=True, random_state=self.MATBENCH_SEED)
        for nf in n_feat_options:
            X_copy = X[self.ranked_features].iloc[:, :nf]
            clf = LGBMClassifier(random_state=self.MATBENCH_SEED, n_jobs=1, verbose=-1)
            search = RandomizedSearchCV(
                clf,
                param_distributions,
                n_iter=self.num_opt_trials,
                scoring=scoring,
                cv=cv,
                random_state=self.MATBENCH_SEED,
                n_jobs=n_jobs,
                pre_dispatch="2*n_jobs",
                verbose=verbose,
                refit=True,)
            search.fit(X_copy, y)
            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_n_feat = nf
                best_params = search.best_params_
        print(f"best_cv_{scoring}={search.best_score_:.4f}")
        best_params["n_feat"] = best_n_feat
        return best_params
