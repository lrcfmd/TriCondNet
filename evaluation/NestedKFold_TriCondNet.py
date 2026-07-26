import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "14"
os.environ["MKL_NUM_THREADS"] = "14"

import sys
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, r2_score,
    classification_report, balanced_accuracy_score, matthews_corrcoef,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from lightgbm import LGBMClassifier
import torch
torch.set_num_threads(14)
try:
    torch.set_num_interop_threads(2)
except RuntimeError:
    pass
print(f"[threads] torch={torch.get_num_threads()}  interop={torch.get_num_interop_threads()}  cuda={torch.cuda.is_available()}")
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
sys.path.insert(0, str(Path.cwd()))

from hyperparameterisation import tricondnet_BO
from models import TriCondNet
from evaluation.preprocessing import RFFeatureSelector

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
    
class TriCondNet_NestedKFold_Pipeline:
    def __init__(self, data_path, cbfv, n_inner, n_outer, num_opt_trials, target_name="target", random_state=0, file_type="csv"):
        self.cbfv = cbfv
        self.n_inner = n_inner
        self.n_outer = n_outer
        self.num_opt_trials = num_opt_trials
        self.target_name = target_name
        self.random_state = random_state

        if file_type == "csv":
            data = pd.read_csv(data_path)
        if file_type=="excel":
            data = pd.read_excel(data_path)

        self.feature_columns = [c for c in data.columns
                                if c not in ["formula", "source", "entry", "target", "class"]]

        self.X      = data[self.feature_columns]
        self.X_formula = data["formula"]
        self.y_class     = data["class"]
        self.y_log10cond = data["target"]
        self.source = data["source"]
        self.entry  = data["entry"]
        self.groups = data["formula"]
    
    def run_nested_cv(self, results_dir="nested_cv_results_PIPELINE_num_epochs", retrain=True,
                      retrain_metal=None, retrain_semi=None, retrain_classifier=None,
                      rehyperparam_metal=False, rehyperparam_semi=False, rehyperparam_classifier=False):
        retrain_metal      = retrain if retrain_metal      is None else retrain_metal
        retrain_semi       = retrain if retrain_semi       is None else retrain_semi
        retrain_classifier = retrain if retrain_classifier is None else retrain_classifier
        if rehyperparam_metal:      retrain_metal      = True
        if rehyperparam_semi:       retrain_semi       = True
        if rehyperparam_classifier: retrain_classifier = True
        print(f"[run_nested_cv] metal: retrain={retrain_metal}, rehp={rehyperparam_metal} | "
              f"semi: retrain={retrain_semi}, rehp={rehyperparam_semi} | "
              f"classifier: retrain={retrain_classifier}, rehp={rehyperparam_classifier}")

        groups = self.groups
        gkf = GroupKFold(n_splits=self.n_outer, shuffle=True, random_state=self.random_state)
        all_hyperparameters = {}

        all_test_balanced_accuracy = []
        all_test_mcc = []
        all_test_metal_precision, all_test_metal_recall, all_test_metal_f1 = [], [], []
        all_test_semi_precision,  all_test_semi_recall,  all_test_semi_f1  = [], [], []



        parity_all_regressor_predictions = []
        parity_all_regressor_true = []

        all_metal_train_predictions, all_metal_test_predictions, all_semi_train_predictions, all_semi_test_predictions = [], [], [], []
        all_metal_train_groundtruth, all_metal_test_groundtruth, all_semi_train_groundtruth, all_semi_test_groundtruth = [], [], [], []



        all_metal_train_mae, all_metal_test_mae = [], []
        all_metal_train_r2, all_metal_test_r2  = [], []

        mean_metal_train_mae, mean_metal_test_mae = [], []
        mean_metal_train_r2, mean_metal_test_r2  = [], []

        all_semi_train_mae, all_semi_test_mae = [], []
        all_semi_train_r2, all_semi_test_r2 = [], []

        mean_semi_train_mae, mean_semi_test_mae = [], []
        mean_semi_train_r2, mean_semi_test_r2 = [], []


        all_pipeline_train_mae, all_pipeline_test_mae = [], []
        all_pipeline_train_r2,  all_pipeline_test_r2  = [], []
        all_pipeline_decomposition = []

        all_y_true = []
        all_y_pred = []
        all_test_formulae  = []
        results_dir = Path(results_dir) / "TriCondNet" 
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "models").mkdir(parents=True, exist_ok=True)

        def _slice(df, idx):
            return df.iloc[idx].reset_index(drop=True)

        for fold_k, (train_idx, test_idx) in enumerate(gkf.split(self.X, self.y_class, groups)):


            X_Train, X_Test= _slice(self.X, train_idx), _slice(self.X, test_idx)
            Y_Train_cl, Y_Test_cl = _slice(self.y_class, train_idx), _slice(self.y_class, test_idx)
            X_Train_Formula, X_Test_Formula =  _slice(self.X_formula, train_idx), _slice(self.X_formula, test_idx)
                

            X_Class_Train, X_Class_Test = X_Train.drop("temp", axis=1), X_Test.drop("temp", axis=1)
            X_Train_Formula, X_Test_Formula = X_Train_Formula.drop_duplicates(), X_Test_Formula.drop_duplicates() 
            X_Class_Train, X_Class_Test = X_Class_Train.loc[X_Train_Formula.index.tolist()], X_Class_Test.loc[X_Test_Formula.index.tolist()]
            Y_Class_Train, Y_Class_Test = Y_Train_cl.loc[X_Class_Train.index.tolist()], Y_Test_cl.loc[X_Class_Test.index.tolist()]


            Y_Train_Log10Cond, Y_Test_Log10Cond = _slice(self.y_log10cond, train_idx), _slice(self.y_log10cond, test_idx)
            Group_Train, Group_Test = _slice(self.groups, train_idx), _slice(self.groups, test_idx)
            Source_Train, Source_Test = _slice(self.source, train_idx), _slice(self.source, test_idx)
            Entry_Train, Entry_Test = _slice(self.entry, train_idx), _slice(self.entry, test_idx)

            metal_pkl = results_dir / f"MetalRegressor_fold_{fold_k}.pkl"
            metal_model_pt = results_dir / "models" / f"MetalRegressor_Model_{fold_k}.pt"
            metal_cached = metal_pkl.exists() and metal_model_pt.exists()

            if metal_cached and not retrain_metal:
                print(f"[Metal fold {fold_k}] cached, loading model + metrics (retrain_metal=False)")
                with open(metal_pkl, "rb") as f:
                    metal_payload_cached = pkl.load(f)
                metal_regressor = TriCondNet.MetalPINN.load(str(metal_model_pt))
                train_mae = metal_payload_cached["metrics"]["train_mae"]
                train_r2  = metal_payload_cached["metrics"]["train_r2"]
                test_mae  = metal_payload_cached["metrics"]["test_mae"]
                test_r2   = metal_payload_cached["metrics"]["test_r2"]
                all_metal_train_mae.append(train_mae)
                all_metal_train_r2.append(train_r2)
                all_metal_test_mae.append(test_mae)
                all_metal_test_r2.append(test_r2)
                all_hyperparameters[fold_k] = {"metal": metal_payload_cached["best_hp"]}
                Metal_Mask_Train = (Y_Train_cl == 1).values
                Metal_Mask_Test  = (Y_Test_cl  == 1).values
                X_Train_Metal = X_Train.loc[Metal_Mask_Train].reset_index(drop=True)
                X_Test_Metal  = X_Test.loc[Metal_Mask_Test].reset_index(drop=True)
                Y_Train_Metal_Log10Cond = Y_Train_Log10Cond.loc[Metal_Mask_Train].reset_index(drop=True)
                Y_Test_Metal_Log10Cond  = Y_Test_Log10Cond.loc[Metal_Mask_Test].reset_index(drop=True)

                target_input1, target_input2 = pd.DataFrame({"target": Y_Train_Metal_Log10Cond}),  pd.DataFrame({"target": Y_Test_Metal_Log10Cond})
                
                mean_train_mae, mean_train_r2, mean_test_mae, mean_test_r2 = self.mean_target_model(target_input1, target_input2)
                
                mean_metal_train_mae.append(mean_train_mae)
                mean_metal_train_r2.append(mean_train_r2)
                mean_metal_test_mae.append(mean_test_mae)
                mean_metal_test_r2.append(mean_test_r2)

                # pooled outer fold
                with torch.no_grad(): 
                    train_preds = metal_regressor.predict(X_Train_Metal).ravel()
                    test_preds = metal_regressor.predict(X_Test_Metal).ravel()

                all_metal_train_predictions.extend(train_preds)
                all_metal_test_predictions.extend(test_preds)
                all_metal_train_groundtruth.extend(Y_Train_Metal_Log10Cond.values)
                all_metal_test_groundtruth.extend(Y_Test_Metal_Log10Cond.values)
            else:
                Metal_Mask_Train = (Y_Train_cl == 1).values
                Metal_Mask_Test  = (Y_Test_cl  == 1).values

                X_Train_Metal = X_Train.loc[Metal_Mask_Train].reset_index(drop=True)
                X_Test_Metal  = X_Test.loc[Metal_Mask_Test].reset_index(drop=True)
                Y_Train_Metal_Log10Cond = Y_Train_Log10Cond.loc[Metal_Mask_Train].reset_index(drop=True)
                Y_Test_Metal_Log10Cond  = Y_Test_Log10Cond.loc[Metal_Mask_Test].reset_index(drop=True)
                Group_Train_Metal  = Group_Train.loc[Metal_Mask_Train].reset_index(drop=True)
                Group_Test_Metal   = Group_Test.loc[Metal_Mask_Test].reset_index(drop=True)
                Source_Train_Metal = Source_Train.loc[Metal_Mask_Train].reset_index(drop=True)
                Source_Test_Metal  = Source_Test.loc[Metal_Mask_Test].reset_index(drop=True)
                Entry_Train_Metal  = Entry_Train.loc[Metal_Mask_Train].reset_index(drop=True)
                Entry_Test_Metal   = Entry_Test.loc[Metal_Mask_Test].reset_index(drop=True)
                Class_Train_Metal  = Y_Train_cl.loc[Metal_Mask_Train].reset_index(drop=True)

                target_input1, target_input2 = pd.DataFrame({"target": Y_Train_Metal_Log10Cond}),  pd.DataFrame({"target": Y_Test_Metal_Log10Cond})
                mean_train_mae, mean_train_r2, mean_test_mae, mean_test_r2 = self.mean_target_model(target_input1, target_input2)

                mean_metal_train_mae.append(mean_train_mae)
                mean_metal_train_r2.append(mean_train_r2)
                mean_metal_test_mae.append(mean_test_mae)
                mean_metal_test_r2.append(mean_test_r2)


                merged_features, per_target_lists, cnmi = RFFeatureSelector.feature_selection(
                    X_Train_Metal, Y_Train_Metal_Log10Cond.to_frame(name=self.target_name),
                    n_jobs=10, random_state=self.random_state,
                    target_name=self.target_name,
                )
                Ranked_Target_List = per_target_lists[self.target_name]
                self.ranked_features = Ranked_Target_List

                if metal_cached and retrain_metal and not rehyperparam_metal:
                    print(f"[Metal fold {fold_k}] cached HP, refitting (retrain_metal=True)")
                    with open(metal_pkl, "rb") as f:
                        cached = pkl.load(f)
                    hyperparameters = cached["best_hp"]
                else:
                    hyperparameters = self.metal_inner_optimizer(X_Train_Metal, Y_Train_Metal_Log10Cond, Group_Train_Metal, Source_Train_Metal, Entry_Train_Metal, Class_Train_Metal)
                all_hyperparameters[fold_k] = {"metal": hyperparameters}
                load_params = hyperparameters["best_params"]
                cleaned_params = []
                for i in load_params:
                    if hasattr(i, "tolist"):
                        cleaned_params.append(i.tolist())
                    else:
                        cleaned_params.append(i)
                n_feat, num_epochs, n_layers, architecture_style, brick_width, funnel_width, funnel_rate, lr, weight_decay, batch_size, dropout_rate, act = cleaned_params
                architecture = architecture_generator(architecture_style, brick_width, funnel_width, n_layers, funnel_rate)
                metal_regressor = TriCondNet.MetalPINN(
                    target_name="target",
                    optimal_descriptors=Ranked_Target_List,
                    n_feat=n_feat,
                    architecture=architecture,
                    act=act,
                    dropout_rate=dropout_rate,
                )
                history = metal_regressor.fit(
                    train_df=X_Train_Metal,
                    train_target=Y_Train_Metal_Log10Cond.to_frame(name=self.target_name),
                    val_df=None,
                    val_target=None,
                    lr=lr,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    epochs=num_epochs,
                    xscale="standard",
                    verbose=False,
                )
                with torch.no_grad():
                    arr_pred_train = metal_regressor.predict(X_Train_Metal).ravel()
                    arr_pred_test  = metal_regressor.predict(X_Test_Metal).ravel()

                train_mae = mean_absolute_error(Y_Train_Metal_Log10Cond.values, arr_pred_train)
                train_r2  = r2_score(Y_Train_Metal_Log10Cond.values, arr_pred_train)
                test_mae  = mean_absolute_error(Y_Test_Metal_Log10Cond.values,  arr_pred_test)
                test_r2   = r2_score(Y_Test_Metal_Log10Cond.values,  arr_pred_test)
                print(f"[Metal fold {fold_k}] train MAE={train_mae:.4f} R2={train_r2:.4f}")
                print(f"[Metal fold {fold_k}] test  MAE={test_mae:.4f} R2={test_r2:.4f}")

                all_metal_train_predictions.extend(arr_pred_train)
                all_metal_test_predictions.extend(arr_pred_test)
                all_metal_train_groundtruth.extend(Y_Train_Metal_Log10Cond.values)
                all_metal_test_groundtruth.extend(Y_Test_Metal_Log10Cond.values)

                all_metal_train_mae.append(train_mae)
                all_metal_train_r2.append(train_r2)
                all_metal_test_mae.append(test_mae)
                all_metal_test_r2.append(test_r2)

                metal_fold_payload = {
                    "fold_k": fold_k,
                    "best_hp": hyperparameters,
                    "metrics": {
                        "train_mae": train_mae,
                        "train_r2":  train_r2,
                        "test_mae":  test_mae,
                        "test_r2":   test_r2,
                    },
                    "n_test":  int(len(Y_Test_Metal_Log10Cond)),
                    "n_train": int(len(Y_Train_Metal_Log10Cond)),
                }
                with open(metal_pkl, "wb") as f:
                    pkl.dump(metal_fold_payload, f)
                metal_regressor.save(str(metal_model_pt))
                print(f"[Metal fold {fold_k}] saved -> {metal_pkl}")


            semi_pkl = results_dir / f"SemiRegressor_fold_{fold_k}.pkl"
            semi_model_pt = results_dir / "models" / f"SemiRegressor_Model_{fold_k}.pt"
            semi_cached = semi_pkl.exists() and semi_model_pt.exists()

            if semi_cached and not retrain_semi:
                print(f"[Semi fold {fold_k}] cached, loading model + metrics (retrain_semi=False)")
                with open(semi_pkl, "rb") as f:
                    semi_payload_cached = pkl.load(f)
                semi_regressor = TriCondNet.SemiconductorPINN.load(str(semi_model_pt))

                train_mae = semi_payload_cached["metrics"]["train_mae"]
                train_r2  = semi_payload_cached["metrics"]["train_r2"]
                test_mae  = semi_payload_cached["metrics"]["test_mae"]
                test_r2   = semi_payload_cached["metrics"]["test_r2"]
                all_semi_train_mae.append(train_mae)
                all_semi_train_r2.append(train_r2)
                all_semi_test_mae.append(test_mae)
                all_semi_test_r2.append(test_r2)
                all_hyperparameters[fold_k]["semi"] = semi_payload_cached["best_hp"]
                Semi_Mask_Train = (Y_Train_cl == 0).values
                Semi_Mask_Test  = (Y_Test_cl  == 0).values
                X_Train_Semi = X_Train.loc[Semi_Mask_Train].reset_index(drop=True)
                X_Test_Semi  = X_Test.loc[Semi_Mask_Test].reset_index(drop=True)
                Y_Train_Semi_Log10Cond = Y_Train_Log10Cond.loc[Semi_Mask_Train].reset_index(drop=True)
                Y_Test_Semi_Log10Cond  = Y_Test_Log10Cond.loc[Semi_Mask_Test].reset_index(drop=True)

                target_input1, target_input2 = pd.DataFrame({"target": Y_Train_Semi_Log10Cond}),  pd.DataFrame({"target": Y_Test_Semi_Log10Cond})
                mean_train_mae, mean_train_r2, mean_test_mae, mean_test_r2 = self.mean_target_model(target_input1, target_input2)


                # pooled outer fold
                with torch.no_grad(): 
                    train_preds = semi_regressor.predict(X_Train_Semi).ravel()
                    test_preds = semi_regressor.predict(X_Test_Semi).ravel()

                all_semi_train_predictions.extend(train_preds)
                all_semi_test_predictions.extend(test_preds)
                all_semi_train_groundtruth.extend(Y_Train_Semi_Log10Cond.values)
                all_semi_test_groundtruth.extend(Y_Test_Semi_Log10Cond.values)


                mean_semi_train_mae.append(mean_train_mae)
                mean_semi_train_r2.append(mean_train_r2)
                mean_semi_test_mae.append(mean_test_mae)
                mean_semi_test_r2.append(mean_test_r2)

            else:
                Semi_Mask_Train = (Y_Train_cl == 0).values
                Semi_Mask_Test  = (Y_Test_cl  == 0).values

                X_Train_Semi = X_Train.loc[Semi_Mask_Train].reset_index(drop=True)
                X_Test_Semi  = X_Test.loc[Semi_Mask_Test].reset_index(drop=True)
                Y_Train_Semi_Log10Cond = Y_Train_Log10Cond.loc[Semi_Mask_Train].reset_index(drop=True)
                Y_Test_Semi_Log10Cond  = Y_Test_Log10Cond.loc[Semi_Mask_Test].reset_index(drop=True)
                Group_Train_Semi  = Group_Train.loc[Semi_Mask_Train].reset_index(drop=True)
                Group_Test_Semi   = Group_Test.loc[Semi_Mask_Test].reset_index(drop=True)
                Source_Train_Semi = Source_Train.loc[Semi_Mask_Train].reset_index(drop=True)
                Source_Test_Semi  = Source_Test.loc[Semi_Mask_Test].reset_index(drop=True)
                Entry_Train_Semi  = Entry_Train.loc[Semi_Mask_Train].reset_index(drop=True)
                Entry_Test_Semi   = Entry_Test.loc[Semi_Mask_Test].reset_index(drop=True)
                Class_Train_Semi  = Y_Train_cl.loc[Semi_Mask_Train].reset_index(drop=True)

                target_input1, target_input2 = pd.DataFrame({"target": Y_Train_Semi_Log10Cond}),  pd.DataFrame({"target": Y_Test_Semi_Log10Cond})
                mean_train_mae, mean_train_r2, mean_test_mae, mean_test_r2 = self.mean_target_model(target_input1, target_input2)
                mean_semi_train_mae.append(mean_train_mae)
                mean_semi_train_r2.append(mean_train_r2)
                mean_semi_test_mae.append(mean_test_mae)
                mean_semi_test_r2.append(mean_test_r2)


                merged_features, per_target_lists, cnmi = RFFeatureSelector.feature_selection(
                    X_Train_Semi, Y_Train_Semi_Log10Cond.to_frame(name=self.target_name),
                    n_jobs=10, random_state=self.random_state,
                    target_name=self.target_name,
                )
                Ranked_Target_List = per_target_lists[self.target_name]
                self.ranked_features = Ranked_Target_List

                if semi_cached and retrain_semi and not rehyperparam_semi:
                    print(f"[Semi fold {fold_k}] cached HP, refitting (retrain_semi=True)")
                    with open(semi_pkl, "rb") as f:
                        cached = pkl.load(f)
                    hyperparameters = cached["best_hp"]
                else:
                    hyperparameters = self.semi_inner_optimizer(X_Train_Semi, Y_Train_Semi_Log10Cond, Group_Train_Semi, Source_Train_Semi, Entry_Train_Semi, Class_Train_Semi)
                all_hyperparameters[fold_k]["semi"] = hyperparameters

                load_params = hyperparameters["best_params"]
                cleaned_params = []
                for i in load_params:
                    if hasattr(i, "tolist"):
                        cleaned_params.append(i.tolist())
                    else:
                        cleaned_params.append(i)
                n_feat, num_epochs, n_layers, architecture_style, brick_width, funnel_width, funnel_rate, lr, weight_decay, batch_size, dropout_rate, act = cleaned_params
                architecture = architecture_generator(architecture_style, brick_width, funnel_width, n_layers, funnel_rate)
                semi_regressor = TriCondNet.SemiconductorPINN(
                    target_name="target",
                    optimal_descriptors=Ranked_Target_List,
                    n_feat=n_feat,
                    architecture=architecture,
                    act=act,
                    dropout_rate=dropout_rate,
                )
                history = semi_regressor.fit(
                    train_df=X_Train_Semi,
                    train_target=Y_Train_Semi_Log10Cond.to_frame(name=self.target_name),
                    val_df=None,
                    val_target=None,
                    lr=lr,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    epochs=num_epochs,
                    xscale="standard",
                    verbose=False,
                )
                with torch.no_grad():
                    arr_pred_train = semi_regressor.predict(X_Train_Semi).ravel()
                    arr_pred_test  = semi_regressor.predict(X_Test_Semi).ravel()

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


                all_semi_train_predictions.extend(arr_pred_train)
                all_semi_test_predictions.extend(arr_pred_test)
                all_semi_train_groundtruth.extend(Y_Train_Semi_Log10Cond.values)
                all_semi_test_groundtruth.extend(Y_Test_Semi_Log10Cond.values)


                semi_fold_payload = {
                    "fold_k": fold_k,
                    "best_hp": hyperparameters,
                    "metrics": {
                        "train_mae": train_mae,
                        "train_r2":  train_r2,
                        "test_mae":  test_mae,
                        "test_r2":   test_r2,
                    },
                    "n_test":  int(len(Y_Test_Semi_Log10Cond)),
                    "n_train": int(len(Y_Train_Semi_Log10Cond)),
                }
                with open(semi_pkl, "wb") as f:
                    pkl.dump(semi_fold_payload, f)
                semi_regressor.save(str(semi_model_pt))
                print(f"[Semi fold {fold_k}] saved -> {semi_pkl}")



            clf_pkl       = results_dir / f"classifier_fold_{fold_k}.pkl"
            clf_model_pkl = results_dir / "models" / f"classifier_model_{fold_k}.pkl"
            clf_cached    = clf_pkl.exists() and clf_model_pkl.exists()

            All_Test_Class_Formulae = Group_Test.copy()

            merged_features, per_target_lists, cnmi = RFFeatureSelector.feature_selection(
                df_feat=X_Class_Train,
                df_targets=Y_Class_Train.to_frame(name="class"),
                target_name="class",
                mode="classification",
                random_state=self.random_state,
            )
            Ranked_Target_List = per_target_lists["class"]
            self.ranked_features = Ranked_Target_List

            if clf_cached and not retrain_classifier:
                print(f"[Classifier fold {fold_k}] cached, loading model + metrics (retrain_classifier=False)")
                with open(clf_pkl, "rb") as f:
                    clf_payload_cached = pkl.load(f)
                with open(clf_model_pkl, "rb") as f:
                    clf = pkl.load(f)
                hyperparameters = clf_payload_cached["best_hp"]
                n_feat = hyperparameters["best_n_feat"]
                features_to_use = [f for f in self.ranked_features if f != "temp" and f != "class"][:n_feat]
                y_test_arr = Y_Class_Test.values.ravel()
                test_pred = clf.predict(X_Class_Test[features_to_use].values)
                test_mcc = clf_payload_cached["metrics"]["mcc"]
                test_acc = clf_payload_cached["metrics"]["balanced_accuracy"]
                report   = clf_payload_cached["report_full"]
                all_hyperparameters[fold_k]["classifier"] = hyperparameters
            else:
                if clf_cached and retrain_classifier and not rehyperparam_classifier:
                    print(f"[Classifier fold {fold_k}] cached HP, refitting (retrain_classifier=True)")
                    with open(clf_pkl, "rb") as f:
                        cached = pkl.load(f)
                    hyperparameters = cached["best_hp"]
                else:
                    hyperparameters = self.class_inner_optimizer(X_Class_Train, Y_Class_Train.values.ravel(), X_Train_Formula)
                all_hyperparameters[fold_k]["classifier"] = hyperparameters

                best_hp = hyperparameters["best_params"]
                n_feat  = hyperparameters["best_n_feat"]
                features_to_use = [f for f in self.ranked_features if f != "temp" and f != "class"][:n_feat]

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
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=14,
                    verbose=-1,
                )

                X_clf_Input = X_Class_Train[features_to_use].values
                y_clf_Input = Y_Class_Train.values.ravel()
                clf.fit(X_clf_Input, y_clf_Input)
                print(f"LightGBM trained on {X_clf_Input.shape[0]} samples, {X_clf_Input.shape[1]} features")

                y_test_arr = Y_Class_Test.values.ravel()
                test_pred = clf.predict(X_Class_Test[features_to_use].values)
                test_mcc = matthews_corrcoef(y_test_arr, test_pred)
                test_acc = balanced_accuracy_score(y_test_arr, test_pred)

                cm = confusion_matrix(y_test_arr, test_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["semiconductor", "metal"])
                disp.plot(cmap=plt.cm.Blues)
                plt.title(f"Confusion Matrix - fold {fold_k}")
                plt.show()

                report = classification_report(y_test_arr, test_pred, target_names=["semiconductor", "metal"], output_dict=True)
                print(classification_report(y_test_arr, test_pred, target_names=["semiconductor", "metal"]))

            TRAIN_CLASS_PREDS_ = clf.predict(X_Class_Train[features_to_use].values)
            TEST_CLASS_PREDS_  = clf.predict(X_Class_Test[features_to_use].values)

            train_form_to_pred = dict(zip(X_Train_Formula.values, TRAIN_CLASS_PREDS_))
            test_form_to_pred  = dict(zip(X_Test_Formula.values,  TEST_CLASS_PREDS_))

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

            with torch.no_grad():
                Metal_Train_Pred = metal_regressor.predict(Metal_TrainData).ravel() if len(Metal_TrainData) else np.array([])
                Semi_Train_Pred  = semi_regressor.predict(Semi_TrainData).ravel()   if len(Semi_TrainData)  else np.array([])
                Metal_Test_Pred  = metal_regressor.predict(Metal_TestData).ravel()  if len(Metal_TestData)  else np.array([])
                Semi_Test_Pred   = semi_regressor.predict(Semi_TestData).ravel()    if len(Semi_TestData)   else np.array([])

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


            test_form_to_true = dict(zip(X_Test_Formula.values, Y_Class_Test.values))
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
                "n_test": int(len(Y_Test_Log10Cond)),
                "n_train": int(len(Y_Train_Log10Cond)),
            }
            with open(results_dir / f"Pipeline_fold_{fold_k}.pkl", "wb") as f:
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
            all_y_true.extend(y_test_arr.tolist())
            all_y_pred.extend(test_pred.tolist())
            all_test_formulae.extend(All_Test_Class_Formulae.tolist())


        # Outer fold aggregate results 

            if not (clf_cached and not retrain_classifier):
                fold_payload = {
                    "fold_k": fold_k,
                    "best_hp": hyperparameters,
                    "metrics": {
                        "balanced_accuracy": test_acc,
                        "mcc": test_mcc,
                        "metal_precision": metal_precision,
                        "metal_recall":    metal_recall,
                        "metal_f1":        metal_f1,
                        "semi_precision":  semi_precision,
                        "semi_recall":     semi_recall,
                        "semi_f1":         semi_f1,
                        "y_true_vals": y_test_arr.tolist(),
                        "y_pred_vals": test_pred.tolist(),
                    },
                    "report_full": report,
                    "n_test":  int(len(Y_Class_Test)),
                    "n_train": int(len(Y_Class_Train)),
                }
                with open(clf_pkl, "wb") as f:
                    pkl.dump(fold_payload, f)
                with open(clf_model_pkl, "wb") as f:
                    pkl.dump(clf, f)
                print(f"[Classifier fold {fold_k}] saved -> {clf_pkl}")

        # Semi 
        MAE_Train_OuterFold_Aggregate_Semi = mean_absolute_error(all_semi_train_groundtruth, all_semi_train_predictions)
        MAE_Test_OuterFold_Aggregate_Semi = mean_absolute_error(all_semi_test_groundtruth, all_semi_test_predictions)
        R2_Train_OuterFold_Aggregate_Semi = r2_score(all_semi_train_groundtruth, all_semi_train_predictions)
        R2_Test_OuterFold_Aggregate_Semi = r2_score(all_semi_test_groundtruth, all_semi_test_predictions)

        # Metal 
        MAE_Train_OuterFold_Aggregate_metal = mean_absolute_error(all_metal_train_groundtruth, all_metal_train_predictions)
        MAE_Test_OuterFold_Aggregate_metal = mean_absolute_error(all_metal_test_groundtruth, all_metal_test_predictions)
        R2_Train_OuterFold_Aggregate_metal = r2_score(all_metal_train_groundtruth, all_metal_train_predictions)
        R2_Test_OuterFold_Aggregate_metal = r2_score(all_metal_test_groundtruth, all_metal_test_predictions)

        
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
        print("--- Mean Target Model - Metal ---")
        print(f"Train MAE: {msd(mean_metal_train_mae)} | Train R2: {msd(mean_metal_train_r2)}")
        print(f"Test  MAE: {msd(mean_metal_test_mae)}  | Test  R2: {msd(mean_metal_test_r2)}")
        print("--- Mean Target Model - Semiconductor ---")
        print(f"Train MAE: {msd(mean_semi_train_mae)} | Train R2: {msd(mean_semi_train_r2)}")
        print(f"Test  MAE: {msd(mean_semi_test_mae)}  | Test  R2: {msd(mean_semi_test_r2)}")

        # ----- pooled out-of-fold metrics (single number, no per-fold averaging) -----
        print("\n========= Pooled out-of-fold (single R2 over all test predictions) =========")
        print("--- Metal regressor (oracle routing) - pooled ---")
        print(f"Test  MAE: {MAE_Test_OuterFold_Aggregate_metal:.4f}  | Test  R2: {R2_Test_OuterFold_Aggregate_metal:.4f}")
        print("--- Semi regressor (oracle routing) - pooled ---")
        print(f"Test  MAE: {MAE_Test_OuterFold_Aggregate_Semi:.4f}  | Test  R2: {R2_Test_OuterFold_Aggregate_Semi:.4f}")


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
            "mean_metal_train_mae": mean_metal_train_mae,
            "mean_metal_train_r2":  mean_metal_train_r2,
            "mean_metal_test_mae":  mean_metal_test_mae,
            "mean_metal_test_r2":   mean_metal_test_r2,
            "mean_semi_train_mae":  mean_semi_train_mae,
            "mean_semi_train_r2":   mean_semi_train_r2,
            "mean_semi_test_mae":   mean_semi_test_mae,
            "mean_semi_test_r2":    mean_semi_test_r2,
        }
        self.all_hyperparameters = all_hyperparameters

        self.parity_true =  parity_all_regressor_true
        self.parity_pred = parity_all_regressor_predictions

        with open(results_dir / "classifier_summary.pkl", "wb") as f:
            pkl.dump({
                "fold_results": self.fold_results,
                "all_hyperparameters": all_hyperparameters,
            }, f)
        print(f"Summary saved -> {results_dir / 'classifier_summary.pkl'}")




        

        return self.fold_results, all_hyperparameters


    def plot_parity(self, title="Parity plot", save_path=None, show=True, dpi=600):
        if not hasattr(self, "parity_true") or not hasattr(self, "parity_pred"):
            raise ValueError("Run run_nested_cv() first to populate parity data.")

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

            stats = f"MAE = {mae:.3f}\n$R^{{2}}$ = {r2:.3f}\n$N$ = {y_true.size:,}"
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

    def class_inner_optimizer(self, X_input, y, groups, scoring="matthews_corrcoef", n_jobs=None, verbose=1):
        n_jobs = 14
        pct_options = [0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        max_feat = len(self.ranked_features)
        n_feat_options = [round(pct * max_feat) for pct in pct_options]

        best_score = -np.inf
        best_model = None
        best_n_feat = None
        best_params = None


        param_distributions = {
            "n_estimators": [200, 300, 400, 500],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "num_leaves": [7, 15, 31, 63],
            "max_depth": [4, 6, 8, 12],
            "min_child_samples": [2, 5, 10, 20],
            "feature_fraction": [0.5, 0.7, 0.9, 1.0],
            "bagging_fraction": [0.6, 0.8, 1.0],
            "bagging_freq": [0, 5],
            "reg_alpha":   [0.0, 0.1, 1.0, 10.0],
            "reg_lambda":  [0.0, 0.1, 1.0, 10.0],
        }

        gkf = GroupKFold(n_splits=self.n_inner)
        for nf in n_feat_options:
            features_to_use = [f for f in self.ranked_features if f != "temp" and f != "class"][:nf]
            X = X_input[features_to_use].values

            clf = LGBMClassifier(random_state=42, n_jobs=1, verbose=-1, class_weight="balanced")

            search = RandomizedSearchCV(
                clf,
                param_distributions,
                n_iter=self.num_opt_trials,
                scoring=scoring,
                cv=gkf,
                random_state=42,
                n_jobs=n_jobs,
                pre_dispatch="2*n_jobs",
                verbose=verbose,
                refit=True,
            )
            search.fit(X, y, groups=groups)

            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_model = search.best_estimator_
                best_n_feat = nf
                best_params = search.best_params_

            print(f"  n_feat={nf:>3d}  best_cv_{scoring}={search.best_score_:.4f}")
        
        return {"best_params": best_params, "best_n_feat": best_n_feat}


    def metal_inner_optimizer(self, X, y, group, source, entry, class_train):
        data_dict = {
            "features": X.reset_index(drop=True),
            self.target_name: pd.DataFrame({self.target_name: y.reset_index(drop=True)}),
            "formula": pd.DataFrame({"formula": group.reset_index(drop=True)}),
            "source": pd.DataFrame({"source": source.reset_index(drop=True)}),
            "class": pd.DataFrame({"class": class_train.reset_index(drop=True)}),
        }
        self.data_dict = data_dict
        regressor_bo = tricondnet_BO.metal_pinn_opt(
            data_dict=data_dict,
            optimal_features_ranking=self.ranked_features,
            num_trials=self.num_opt_trials,
            target_name=self.target_name,
            keys=['features', 'formula', self.target_name, 'source', 'class'],
        )
        regressor_bo.run_optimization()
        return {"best_params": list(regressor_bo.search_result.x)}


    def semi_inner_optimizer(self, X, y, group, source, entry, class_train):
        data_dict = {
            "features": X.reset_index(drop=True),
            self.target_name: pd.DataFrame({self.target_name: y.reset_index(drop=True)}),
            "formula": pd.DataFrame({"formula": group.reset_index(drop=True)}),
            "source": pd.DataFrame({"source": source.reset_index(drop=True)}),
            "class": pd.DataFrame({"class": class_train.reset_index(drop=True)}),
        }
        self.data_dict = data_dict
        regressor_bo = tricondnet_BO.semiconductor_pinn_opt(
            data_dict=data_dict,
            optimal_features_ranking=self.ranked_features,
            num_trials=self.num_opt_trials,
            target_name=self.target_name,
            keys=['features', 'formula', self.target_name, 'source', 'class'],
        )
        regressor_bo.run_optimization()
        return {"best_params": list(regressor_bo.search_result.x)}

    def mean_target_model(self, Train_Target, Test_Target): 
        train_target = Train_Target["target"]
        mean_train_target_val = np.mean(train_target)
        test_target = Test_Target["target"]

        array_train = np.full(train_target.shape[0], mean_train_target_val,)
        array_test = np.full(test_target.shape[0], mean_train_target_val,)

        train_mae, train_r2 = mean_absolute_error(train_target, array_train), r2_score(train_target, array_train)
        test_mae, test_r2 = mean_absolute_error(test_target, array_test), r2_score(test_target, array_test)
        return train_mae, train_r2, test_mae, test_r2

