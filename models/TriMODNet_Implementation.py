import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from sklearn.metrics import roc_auc_score
from modnet.models import MODNetModel
from modnet.preprocessing import MODData
from evaluation.preprocessing import FeatureSelector
import tensorflow as tf 
from sklearn.metrics import matthews_corrcoef
"""
In this file, we include a classifier via MODNet and a regressor via MODNet. 
There are a few workflows that can be done using this file. 

1):
Instead, we use MODNet to classify
the materials, with 2 following regressors for metal and semiconductor conductivity modelling. How 
This uses the original implementation of MODNet (Python 3.9), no GPU acceleration (not compatible with Barkla CUDA).
        
2): 
Using MODNet regressor for the entire datab
"""

class MODNet_Classifier:
    def __init__(
            self,
            train_input,
            val_input=None,
            feature_rank=True,
            ranked_features=None,
            class_name="class"):
        
        self.class_name = class_name
        
        modnet_train = MODData(
            df_featurized = train_input["features"],
            targets = train_input[self.class_name],
            target_names = [self.class_name]
        )

        self.train_targets = train_input[self.class_name]

        if feature_rank:
            merged_features, per_target_lists, cnmi = FeatureSelector.feature_selection(train_input["features"], train_input[self.class_name], n_jobs=15, random_state=42)
            ranked_tgt_list = per_target_lists[self.class_name]
            self.ranked_features = ranked_tgt_list

        else: 
            self.ranked_features = ranked_features

        modnet_train.optimal_features = self.ranked_features
        modnet_train.optimal_descriptors = self.ranked_features

        self.train = modnet_train
        self.val = None
        self.test = None

        if val_input: 
            modnet_val = MODData(
                df_featurized = val_input["features"],
                targets = val_input[self.class_name],
                target_names = [self.class_name])
            modnet_val.optimal_features = self.ranked_features
            modnet_val.optimal_descriptors = self.ranked_features
            self.val = modnet_val 
            self.val_targets = val_input[self.class_name]


    def classifier(self,
                   n_feat,
                   architecture,
                   lr,
                   batch_size,
                   act,
                   epochs,
                   patience=None):

        model = MODNetModel(
            targets=[[[self.class_name,]]],
            weights={self.class_name: 1.0},
            num_classes={self.class_name: 2},
            num_neurons=architecture,
            n_feat=n_feat,
            act=act,
        )

        callbacks = []
        if patience is not None and patience > 0:
            self.es = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                min_delta=0.001,
                restore_best_weights=True,
                verbose=1)
            callbacks = [self.es]


        train_classes = self.train_targets.values
        n_semi = (train_classes == 0).sum()
        n_metal = (train_classes == 1).sum()
        metal_weight = float(n_semi) / float(n_metal)
        sample_weight = np.where(train_classes == 1, metal_weight, 1.0)

        fit_kwargs = dict(lr=lr, epochs=epochs, batch_size=batch_size, callbacks=callbacks, loss="categorical_crossentropy", sample_weight=sample_weight)
        if self.val:
            fit_kwargs["val_data"] = self.val
        model.fit(
            self.train,  # MODData object
            **fit_kwargs)
        self.model = model
    
    def load(self, model_path):
        model = MODNetModel.load(model_path)
        self.model = model
    
    def return_val_loss(self): 
        history = self.model.history 
        val_loss = min(history["val_loss"])
        return val_loss
    
    def return_val_loss_MCC(self):
        y_pred = self.model.predict(self.val, remap_out_of_bounds=False)[self.class_name].to_list()
        y_true = self.val_targets
        y_true = y_true[self.class_name].to_list()
        mcc = matthews_corrcoef(y_true, y_pred)
        return mcc

    
    def predict(self, test): 
        """Predict metal/semiconductor class labels for a test set.

        test: dict with a "features" dataframe and a "class" column.
        """
        modnet_test = MODData(
            df_featurized = test["features"],
            targets = test[self.class_name],
            target_names = [self.class_name])
        modnet_test.optimal_features = self.ranked_features
        modnet_test.optimal_descriptors = self.ranked_features
        test_pred_class = self.model.predict(modnet_test, remap_out_of_bounds=False)
        return test_pred_class
    
    

class MODNet_Regressor:
    def __init__(
            self,
            train_input,
            val_input=None,
            test_input=None,
            feature_rank=True,
            ranked_features=None,
            target_name="target"):
        
        self.target_name = target_name
        
        modnet_train = MODData(
            df_featurized = train_input["features"],
            targets = train_input[self.target_name],
            target_names = [self.target_name]
        )

        if feature_rank:
            merged_features, per_target_lists, cnmi = FeatureSelector.feature_selection(train_input["features"], train_input[self.target_name], n_jobs=15, random_state=42)
            ranked_tgt_list = per_target_lists[self.target_name]
            self.ranked_features = ranked_tgt_list

        else: 
            self.ranked_features = ranked_features

        modnet_train.optimal_features = self.ranked_features
        modnet_train.optimal_descriptors = self.ranked_features

        self.train = modnet_train
        self.val = None
        self.test = None

        if val_input: 
            modnet_val = MODData(
                df_featurized = val_input["features"],
                targets = val_input[self.target_name],
                target_names = [self.target_name])
            modnet_val.optimal_features = self.ranked_features
            modnet_val.optimal_descriptors = self.ranked_features
            self.val = modnet_val 

        if test_input:
            modnet_test = MODData(
                df_featurized = test_input["features"],
                targets = test_input[self.target_name],
                target_names = [self.target_name])
            self.test = modnet_test
        print(f"Successfully loaded MODNet data")
    
    def regressor(self,
                   n_feat,
                   architecture,
                   lr,
                   batch_size,
                   act,
                   epochs,
                   patience=None):

        model = MODNetModel(
            targets=[[[self.target_name,]]],
            weights={self.target_name: 1.0},
            num_neurons=architecture,
            n_feat=n_feat,
            act=act,
        )

        callbacks = []
        if patience is not None and patience > 0:
            self.es = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                min_delta=0.001,
                restore_best_weights=True,
                verbose=1)
            callbacks = [self.es]

        fit_kwargs = dict(lr=lr, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        if self.val:
            fit_kwargs["val_data"] = self.val
        model.fit(
            self.train,  # MODData object
            **fit_kwargs,)

        self.model = model

    
    def load(self, model_path):
        model = MODNetModel.load(model_path)
        self.model = model


    def return_val_loss(self):
        history = self.model.history
        val_loss = min(history["val_loss"])
        return val_loss

    def return_final_val_loss(self):
        history = self.model.history
        return history["val_loss"][-1]

    def predict(self, test):
        """Predict target values for a test set.

        test: dict with a "features" dataframe and a target column.
        """

        modnet_test = MODData(
            df_featurized = test["features"],
            targets = test[self.target_name],
            target_names = [self.target_name])
        
        modnet_test.optimal_features = self.ranked_features
        modnet_test.optimal_descriptors = self.ranked_features
        test_pred = self.model.predict(modnet_test)
        return test_pred
        
        
