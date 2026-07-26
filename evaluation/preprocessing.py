"""Preprocessing utilities: data indexing/concatenation helpers and two feature
selectors - an NMI relevance-redundancy selector (FeatureSelector) and a random-forest
importance selector (RFFeatureSelector)."""

from functools import partial
from multiprocessing import Pool

import pandas as pd
import numpy as np
import tqdm
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

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


def data_indexer(data, index=None, feature_columns=None): 
    """
    Creates a final dataset based on the index and feature columns provided, streamlines and simplifies TTS step.

    index can be "all" or a list of indices
    """
    if index is None: 
        final_data = {"features": data.reset_index(drop=True)[feature_columns],
                    "formula": data.reset_index(drop=True)["formula"],
                    "target": pd.DataFrame({"target": data.reset_index(drop=True)["target"]}),
                    "class": pd.DataFrame({"class": data.reset_index(drop=True)["class"]}),
                    "source": data.reset_index(drop=True)["source"]}
    else:
        final_data = {"features": data.iloc[index].reset_index(drop=True)[feature_columns],
                    "formula": data.iloc[index].reset_index(drop=True)["formula"],
                    "target": pd.DataFrame({"target": data.iloc[index].reset_index(drop=True)["target"]}),
                    "class": pd.DataFrame({"class": data.iloc[index].reset_index(drop=True)["class"]}),
                    "source": data.iloc[index].reset_index(drop=True)["source"]}

    return final_data


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

def data_concatenater(data): 
    """
    data: dict containing keys
    """
    df_data = pd.DataFrame()
    for key in data.keys(): 
        df_data = pd.concat([df_data, data[key]], axis=1)
    return df_data
        

def moddata_formula_split(data, formula, test_size=0.1): 
    # Currently designed for single TTS for validation, will change into KFold in the future
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size)
    train_idx, val_idx = next(gss.split(formula, groups=formula["formula"]))
    train_data, val_data = data.split((train_idx, val_idx))
    # can return indices for group shuffle split validation
    return train_data, val_data

def df_formula_split_with_targets(data, targets, formula, test_size=0.1): 
    """
    This splits a pandas DataFrame and the corresponding targets based on formula to 
    prevent 
    """
    # Currently designed for single TTS for validation, will change into KFold in the future
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size)
    train_idx, val_idx = next(gss.split(data, groups=formula["formula"]))

    train_data, val_data = data.iloc[train_idx], data.iloc[val_idx]
    train_targets, val_targets = targets.iloc[train_idx], targets.iloc[val_idx]
    return train_data, val_data, train_targets, val_targets

class FeatureSelector:
    """
    A class providing methods to perform feature selection based on
    mutual information (MI) and a relevance–redundancy (RR) approach.

    **How it works**:
      1) Computes cross-feature NMI (feature vs feature).
      2) Computes feature–target NMI for each target.
      3) Selects features by maximizing relevance (to the target)
         and minimizing redundancy (to already-chosen features).
      4) Optionally merges multi-target results into a single final ranking.
    """

    EPS = 1e-16

    @staticmethod
    def compute_mi(
        x: np.ndarray,
        y: np.ndarray,
        x_name: str,
        y_name: str,
        random_state=None,
        n_neighbors=3
    ):
        """
        Computes mutual information (MI) between two 1D arrays x and y.
        Returns the MI value plus the feature names x_name, y_name (for reference).
        """
        mi_val = mutual_info_regression(
            x.reshape(-1, 1),
            y,
            random_state=random_state,
            n_neighbors=n_neighbors
        )[0]
        return mi_val, x_name, y_name

    @classmethod
    def map_mi(cls, kwargs):
        """
        Helper for multiprocessing to map arguments to compute_mi.
        """
        return cls.compute_mi(**kwargs)

    @classmethod
    def get_cross_nmi(
        cls,
        df_feat: pd.DataFrame,
        drop_thr: float = 0.2,
        return_entropy: bool = False,
        n_jobs: int = 1,
        random_state=None,
        n_neighbors: int = 3
    ):
        """
        Computes the (normalized) mutual information (NMI) between all pairs
        of features in `df_feat`. This is used to detect redundancy among features.

        Args:
            df_feat: DataFrame of shape (n_samples, n_features).
            drop_thr: Entropy threshold below which features get dropped entirely.
            return_entropy: If True, return a dict of each feature's "self MI" (entropy).
            n_jobs: Number of parallel processes for MI.
            random_state: Seed for reproducible MI calculations.
            n_neighbors: Number of neighbors for MI estimation.

        Returns:
            cross_nmi: A (features x features) DataFrame of normalized MI values.
            (optionally) diag: A dict of feature entropies if `return_entropy=True`.
        """

        # Handle NaNs: scale data to (-0.5, 0.5) and fill with -1
        if df_feat.isna().any().any():
            scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
            x = df_feat.values
            x = scaler.fit_transform(x)
            x = np.nan_to_num(x, nan=-1)
            df_feat = pd.DataFrame(x, index=df_feat.index, columns=df_feat.columns)

        features = df_feat.columns
        cross_nmi = pd.DataFrame(index=features, columns=features, data=0.0)

        # Prepare a multiprocessing pool
        pool = Pool(processes=n_jobs)

        # 1) Compute "self" MI for each feature (its entropy)
        diag = {}
        tasks = []
        for feat in features:
            tasks.append({
                "x": df_feat[feat].values,
                "y": df_feat[feat].values,
                "x_name": feat,
                "y_name": feat,
                "random_state": random_state,
                "n_neighbors": n_neighbors
            })

        to_drop = []
        for res in tqdm.tqdm(pool.imap_unordered(cls.map_mi, tasks), total=len(tasks)):
            feat_name = res[1]
            diag[feat_name] = res[0]
            # Drop feature if self-MI < drop_thr or if feature has near-zero variance
            if diag[feat_name] < drop_thr or (
                abs(df_feat[feat_name].max() - df_feat[feat_name].min()) < cls.EPS
            ):
                to_drop.append(feat_name)
            else:
                cross_nmi.loc[feat_name, feat_name] = 1.0

        cross_nmi.drop(to_drop, axis=0, inplace=True)
        cross_nmi.drop(to_drop, axis=1, inplace=True)

        # 2) Compute cross-feature MI for the surviving features
        keep_feats = cross_nmi.index
        tasks = []
        for i, feat_x in enumerate(keep_feats):
            for feat_y in keep_feats[i+1:]:
                tasks.append({
                    "x": df_feat[feat_x].values,
                    "y": df_feat[feat_y].values,
                    "x_name": feat_x,
                    "y_name": feat_y,
                    "random_state": random_state,
                    "n_neighbors": n_neighbors
                })

        for res in tqdm.tqdm(pool.imap_unordered(cls.map_mi, tasks), total=len(tasks)):
            x_name, y_name = res[1], res[2]
            # Normalize by 0.5 * (entropy(x_name) + entropy(y_name))
            cross_nmi.loc[x_name, y_name] = cross_nmi.loc[y_name, x_name] = (
                res[0] / (0.5 * (diag[x_name] + diag[y_name]))
            )

        pool.close()
        pool.join()

        cross_nmi.fillna(0, inplace=True)

        if return_entropy:
            return cross_nmi, diag
        return cross_nmi

    @staticmethod
    def nmi_target(
        df_feat: pd.DataFrame,
        df_target: pd.DataFrame,
        task_type: str = "regression",
        drop_constant_features: bool = True,
        drop_duplicate_features: bool = True,
        random_state=None,
        n_neighbors: int = 3
    ) -> pd.Series:
        """
        Computes the Normalized Mutual Information (NMI) between each feature
        and a single target (the target DataFrame must have exactly one column).

        Args:
            df_feat: (n_samples, n_features)
            df_target: (n_samples, 1) The single target column as a DataFrame.
            task_type: "regression" or "classification".
            drop_constant_features: Drop features that are constant.
            drop_duplicate_features: Drop features that are duplicates.
            random_state: For reproducible MI.
            n_neighbors: For MI.

        Returns:
            A Series of shape (n_features,) with NMI vs target.
        """
        if df_target.shape[1] != 1:
            raise ValueError("df_target must have exactly one column.")

        # Make a copy
        df_feat = df_feat.copy()

        # Drop duplicates
        if drop_duplicate_features:
            df_feat = df_feat.T.drop_duplicates().T

        # Drop constant features
        if drop_constant_features:
            df_feat = df_feat.loc[:, (df_feat != df_feat.iloc[0]).any()]

        # Scale if NaNs
        if df_feat.isna().any().any():
            scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
            x = df_feat.values
            x = scaler.fit_transform(x)
            x = np.nan_to_num(x, nan=-1)
            df_feat = pd.DataFrame(x, index=df_feat.index, columns=df_feat.columns)

        if task_type == "regression":
            mi_fun = mutual_info_regression
            self_mi_fun = mutual_info_regression
        else:
            mi_fun = mutual_info_classif
            self_mi_fun = partial(mutual_info_classif, discrete_features=True)

        target_name = df_target.columns[0]
        feats = df_feat.columns
        mi_vals = mi_fun(
            df_feat,
            df_target[target_name],
            random_state=random_state,
            n_neighbors=n_neighbors
        )
        mi_series = pd.Series(data=mi_vals, index=feats, name="MI")

        # Compute "self" MI for target
        target_entropy = self_mi_fun(
            df_target[target_name].values.reshape(-1, 1),
            df_target[target_name],
            random_state=random_state,
            n_neighbors=n_neighbors
        )[0]

        # Compute "self" MI for each feature
        diag = {}
        for f in feats:
            diag[f] = mutual_info_regression(
                df_feat[f].values.reshape(-1, 1),
                df_feat[f],
                random_state=random_state,
                n_neighbors=n_neighbors
            )[0]

        # Normalize: NMI = MI / [0.5*(entropy_of_feature + entropy_of_target)]
        for f in feats:
            nmi = mi_series[f] / (0.5 * (target_entropy + diag[f]))
            mi_series[f] = 0 if pd.isna(nmi) else nmi

        return mi_series

    @classmethod
    def get_features_dyn(
        cls,
        n_feat: int,
        cross_nmi: pd.DataFrame,
        target_nmi: pd.Series
    ) -> list:
        """
        Select `n_feat` features by a dynamic relevance–redundancy approach.

        Args:
            n_feat: How many features to select. Use -1 for "select all".
            cross_nmi: DataFrame of shape (features x features) with cross-feature NMI.
            target_nmi: Series of shape (features,) with feature-target NMI.

        Returns:
            A list of selected feature names in order of selection.
        """

        def get_rr_p_parameter_default(nn: int) -> float:
            """Default formula for parameter 'p' in the RR step."""
            return max(0.1, 4.5 - 0.4 * nn**0.4)

        def get_rr_c_parameter_default(nn: int) -> float:
            """Default formula for parameter 'c' in the RR step."""
            return min(1e5, 1e-6 * nn**3)

        # Filter to common features
        common_feats = cross_nmi.index.intersection(target_nmi.index)
        cross_nmi = cross_nmi.loc[common_feats, common_feats]
        target_nmi = target_nmi.loc[common_feats].fillna(0)

        # Start with the feature having largest target NMI
        first_feature = target_nmi.idxmax()
        chosen = [first_feature]

        if n_feat == -1:
            n_feat = len(cross_nmi.index)
        else:
            n_feat = min(len(cross_nmi.index), n_feat)

        # RR-based selection loop
        for step in range(n_feat - 1):
            p = get_rr_p_parameter_default(step)
            c = get_rr_c_parameter_default(step)

            # Evaluate each unchosen feature
            remaining = list(set(cross_nmi.index) - set(chosen))
            scores = {}
            for feat in remaining:
                # Compare cross_nmi with the chosen subset
                cross_vals = cross_nmi.loc[feat, chosen]
                # Original approach: ratio per chosen, then take the min
                # ratio_j = target_nmi[feat] / (cross_val^p + c)
                ratio_vals = [
                    target_nmi[feat] / (val**p + c) for val in cross_vals
                ]
                scores[feat] = min(ratio_vals)

            # Pick the feature with the best (max) min ratio
            next_feature = max(scores, key=scores.get)
            chosen.append(next_feature)

        return chosen

    @staticmethod
    def merge_ranked(ranked_lists: list) -> list:
        """
        Merge multiple lists of ranked features. Each list is traversed
        in parallel, picking each feature the first time it appears.

        Example:
            list1 = [A, B, C, D]
            list2 = [B, A, E, F]
        Merge order = A, B, C, D, E, F

        Returns:
            A single merged list with no duplicates.
        """
        max_len = max(len(lst) for lst in ranked_lists)
        for i, lst in enumerate(ranked_lists):
            if len(lst) < max_len:
                ranked_lists[i] = lst + [None]*(max_len - len(lst))

        merged = []
        seen = set()
        for row in zip(*ranked_lists):
            for elem in row:
                if elem is not None and elem not in seen:
                    merged.append(elem)
                    seen.add(elem)
        return merged

    @classmethod
    def feature_selection(
        cls,
        df_feat: pd.DataFrame,
        df_targets: pd.DataFrame,
        n: int = 5000,
        cross_nmi: pd.DataFrame = None,
        n_samples: int = 5000,
        drop_thr: float = 0.2,
        n_jobs: int = 1,
        random_state: int = None,
        ignore_targets: list = None
    ):
        """
        Selects up to `n` features per target by a relevance–redundancy approach,
        then merges these lists for all targets into one final ranking.

        Args:
            df_feat: (n_samples, n_features)
            df_targets: (n_samples, n_targets)
            n: How many features to select per target
            cross_nmi: (Optional) cross-feature NMI matrix; computed if None
            n_samples: If > 0, uses at most this many rows to compute cross_nmi
            drop_thr: Entropy threshold below which features get dropped
            n_jobs: # of parallel processes
            random_state: For reproducible MI
            ignore_targets: List of targets to skip in feature selection

        Returns:
            - merged_list: single merged ranking (list of feature names)
            - per_target_lists: dict of target -> feature list in order
            - cross_nmi_used: the cross_nmi matrix (either provided or newly computed)
        """

        if ignore_targets is None:
            ignore_targets = []

        # Decide which target columns we actually use
        selected_targets = [t for t in df_targets.columns if t not in ignore_targets]
        if not selected_targets:
            raise ValueError("No valid targets found after ignoring the specified ones.")

        # Possibly compute cross_nmi
        if cross_nmi is None:
            if len(df_feat) > n_samples > 0:
                df_sub = df_feat.sample(n=n_samples, random_state=12)
            else:
                df_sub = df_feat
            cross_nmi_used = cls.get_cross_nmi(
                df_sub,
                drop_thr=drop_thr,
                n_jobs=n_jobs,
                random_state=random_state
            )
        else:
            cross_nmi_used = cross_nmi

        # Build per-target lists
        per_target_lists = {}
        for target_name in selected_targets:
            # 1) Estimate feature-target NMI (assume regression for simplicity)
            if len(df_feat) > n_samples > 0:
                subset = np.random.permutation(len(df_feat))[:n_samples]
                df_f_sub = df_feat.iloc[subset]
                df_t_sub = df_targets[[target_name]].iloc[subset]
            else:
                df_f_sub = df_feat
                df_t_sub = df_targets[[target_name]]

            tar_nmi = cls.nmi_target(
                df_f_sub, df_t_sub,
                task_type="regression",
                random_state=random_state
            )

            # 2) Dynamic selection of top-n features
            top_feats = cls.get_features_dyn(n, cross_nmi_used, tar_nmi)
            per_target_lists[target_name] = top_feats

        # Merge all lists
        merged_list = cls.merge_ranked(list(per_target_lists.values()))

        return merged_list, per_target_lists, cross_nmi_used


class RFFeatureSelector:
    @classmethod
    def feature_selection(
        cls,
        df_feat: pd.DataFrame,
        df_targets: pd.DataFrame,
        target_name: str, 
        mode = "regression",
        n: int = 5000,
        cross_nmi: pd.DataFrame = None,       # unused, kept for interface compatibility
        n_samples: int = 5000,
        drop_thr: float = 0.2,                # unused, kept for interface compatibility
        n_jobs: int = 1,
        random_state: int = None,
        n_estimators: int = 200,
    ):
        # Drop constant and duplicate features up front
        df_feat = df_feat.copy()
        df_feat = df_feat.loc[:, (df_feat != df_feat.iloc[0]).any()]
        df_feat = df_feat.T.drop_duplicates().T

        # Fill NaNs with column median so RF can handle them
        df_feat = df_feat.fillna(df_feat.median())

        # Subsample rows if the dataset is large (speeds things up)
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
        # Sort all features by importance score, highest first
        per_target_lists = {}
        importances = rf.feature_importances_
        ranked_idx = np.argsort(importances)[::-1]
        per_target_lists[target_name] = df_feat.columns[ranked_idx].tolist()

        # Merge per-target rankings into one list (same logic as FeatureSelector)
        merged_list = FeatureSelector.merge_ranked(list(per_target_lists.values()))
        return merged_list, per_target_lists, None
