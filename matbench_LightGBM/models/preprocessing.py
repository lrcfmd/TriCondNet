import pandas as pd
import numpy as np
from multiprocessing import Pool
import tqdm

from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit

def nan_indices_finder(df):
    nan_indices = []
    for col in df.columns:
        col_check = df[col].isna()
        col_check = col_check[col_check==True]
        col_check_indices = col_check.index
        nan_indices.extend(col_check_indices.tolist())
    nan_indices = set(nan_indices)
    nan_indices = list(nan_indices)
    return nan_indices


class RFFeatureSelector:
    @classmethod
    def feature_selection(
        cls,
        df_feat: pd.DataFrame,
        df_targets: pd.DataFrame,
        target_name: str,
        mode = "regression",
        n: int = 5000,
        cross_nmi: pd.DataFrame = None,
        n_samples: int = 5000,
        drop_thr: float = 0.2,
        n_jobs: int = 1,
        random_state: int = None,
        n_estimators: int = 200,
    ):
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        df_feat = df_feat.copy()
        df_feat = df_feat.loc[:, (df_feat != df_feat.iloc[0]).any()]
        df_feat = df_feat.T.drop_duplicates().T

        df_feat = df_feat.fillna(df_feat.median())

        if len(df_feat) > n_samples > 0:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(df_feat), size=n_samples, replace=False)
            X = df_feat.iloc[idx]
            y = df_targets[target_name].iloc[idx]
        else:
            X = df_feat
            y = df_targets[target_name]

        if mode=="regression":
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        elif mode=="classification":
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=random_state,
            )

        rf.fit(X, y)
        per_target_lists = {}
        importances = rf.feature_importances_
        ranked_idx = np.argsort(importances)[::-1]
        per_target_lists[target_name] = df_feat.columns[ranked_idx].tolist()

        return per_target_lists
