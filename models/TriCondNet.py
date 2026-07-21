import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import List, Optional, Tuple, Union
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

K_BOLTZ = 8.617333262145e-5
_LN10 = math.log(10.0)


def _leaky_clamp(x, min_val, max_val, leak=0.01):
    clamped = torch.clamp(x, min_val, max_val)
    return clamped + leak * (x - clamped)


def _composition_groups(train_df: pd.DataFrame):
    cols = [c for c in train_df.columns if c != "temp"]
    arr = np.ascontiguousarray(np.asarray(train_df[cols], dtype=float))
    groups = {}
    for pos in range(arr.shape[0]):
        groups.setdefault(arr[pos].tobytes(), []).append(pos)
    return list(groups.values())


def _quality_weights(fit_mae, scale=0.1):
    """Down-weight compositions the closed-form model describes poorly."""
    return 1.0 / (1.0 + np.asarray(fit_mae, dtype=float) / scale)


def semiconductor_warmstart_targets(train_df, train_target, target_name="target",
                                    min_points=2):
    """Per-composition (lnA, Ea) from OLS of log10(sigma) on 1/T."""
    y_all = np.asarray(train_target[target_name], dtype=float)
    T_all = np.asarray(train_df["temp"], dtype=float)

    rows, params, maes = [], [], []
    for grp in _composition_groups(train_df):
        if len(grp) < min_points:
            continue
        T, y = T_all[grp], y_all[grp]
        x = 1.0 / T
        if np.ptp(x) < 1e-12:                     # single temperature: unidentifiable
            continue
        slope, intercept = np.polyfit(x, y, 1)
        rows.append(grp[0])
        params.append((intercept * _LN10, -slope * _LN10 * K_BOLTZ))
        maes.append(float(np.mean(np.abs(y - (intercept + slope * x)))))

    if not rows:
        return None
    maes = np.asarray(maes, dtype=float)
    return {"rows": np.asarray(rows, dtype=int),
            "params": np.asarray(params, dtype=float),
            "weights": np.repeat(_quality_weights(maes)[:, None], 2, axis=1),
            "fit_mae": maes}


def metal_warmstart_targets(train_df, train_target, target_name="target",
                            min_points=3, n_max=4.0, n_steps=201,
                            identifiable_only=True, share_tol=0.05, irls_iters=6):
    """Per-composition (lnA, lnBrel, n) for rho = A*(T/500)^n + B.

    n is found by a grid search (a local optimiser converges to a degenerate
    solution whenever one term dominates rho); A and B follow in closed form at
    each n from an IRLS-weighted linear least squares, weighted so the residual
    is minimised in log space.

    With ``identifiable_only`` the degenerate compositions are simply left out
    of stage 1 - they still take part in the physics loss. Otherwise they are
    kept, with the parameter their data cannot determine replaced by the
    population median and down-weighted.
    """
    y_all = np.asarray(train_target[target_name], dtype=float)
    T_all = np.asarray(train_df["temp"], dtype=float)
    n_grid = np.linspace(0.0, n_max, n_steps)

    recs = []
    for grp in _composition_groups(train_df):
        if len(grp) < min_points:
            continue
        T, y = T_all[grp], y_all[grp]
        if len(np.unique(T)) < 3:                 # need 3 distinct T for 3 parameters
            continue
        rho = np.power(10.0, -y)
        U = np.power(T[None, :] / 500.0, n_grid[:, None])
        W = np.repeat((1.0 / np.clip(rho, 1e-300, None) ** 2)[None, :], n_steps, axis=0)

        A = B = None
        for _ in range(irls_iters):
            Suu = (W * U * U).sum(1); Su = (W * U).sum(1); Sw = W.sum(1)
            Suy = (W * U * rho).sum(1); Sy = (W * rho).sum(1)
            det = Suu * Sw - Su * Su
            safe = np.where(np.abs(det) > 1e-300, det, 1.0)
            A = np.clip((Suy * Sw - Su * Sy) / safe, 1e-30, None)
            B = np.clip((Suu * Sy - Su * Suy) / safe, 1e-30, None)
            W = 1.0 / np.clip(A[:, None] * U + B[:, None], 1e-30, None) ** 2

        pred = A[:, None] * U + B[:, None]
        resid = np.abs(-np.log10(np.clip(pred, 1e-300, None)) - y[None, :]).mean(1)
        k = int(np.argmin(resid))
        Ak, Bk, nk = float(A[k]), float(B[k]), float(n_grid[k])
        share = Ak * U[k] / (Ak * U[k] + Bk)      # fraction of rho carried by the A term
        recs.append({"row": grp[0], "lnA": math.log(Ak), "lnBrel": math.log(Bk / Ak),
                     "n": nk, "mae": float(resid[k]),
                     "smin": float(share.min()), "smax": float(share.max())})

    if not recs:
        return None

    ident = [r for r in recs if r["smax"] >= share_tol and r["smin"] <= 1.0 - share_tol]

    if identifiable_only or not ident:
        keep = ident if ident else recs
        maes = np.array([r["mae"] for r in keep])
        print(f"[warm-start] metal: {len(keep)}/{len(recs)} compositions identifiable")
        return {"rows": np.array([r["row"] for r in keep], dtype=int),
                "params": np.array([[r["lnA"], r["lnBrel"], r["n"]] for r in keep]),
                "weights": np.repeat(_quality_weights(maes)[:, None], 3, axis=1),
                "fit_mae": maes}

    med = {k: float(np.median([r[k] for r in ident])) for k in ("lnA", "lnBrel", "n")}
    rows, params, weights, maes = [], [], [], []
    for r in recs:
        lnA, lnBrel, n = r["lnA"], r["lnBrel"], r["n"]
        w3 = np.ones(3)
        if r["smax"] < share_tol:                 # B dominates: only lnB = lnA + lnBrel is fixed
            lnB = lnA + lnBrel
            lnA, lnBrel, n = med["lnA"], lnB - med["lnA"], med["n"]
            w3 = np.array([0.3, 1.0, 0.3])
        elif r["smin"] > 1.0 - share_tol:         # A dominates: lnBrel is unconstrained
            lnBrel = med["lnBrel"]
            w3 = np.array([1.0, 0.3, 1.0])
        rows.append(r["row"]); params.append([lnA, lnBrel, n]); maes.append(r["mae"])
        weights.append(w3 * float(_quality_weights([r["mae"]])[0]))
    print(f"[warm-start] metal: {len(ident)}/{len(recs)} identifiable, "
          f"{len(recs) - len(ident)} kept with a median prior")
    return {"rows": np.array(rows, dtype=int), "params": np.array(params),
            "weights": np.array(weights), "fit_mae": np.array(maes)}


def _run_pretrain(obj, X_scaled, targets, weights, epochs, lr, weight_decay,
                  batch_size, verbose=False):
    """Stage 1: regress the physical parameters directly, one row per composition.

    Targets are standardised by their own spread so the heads train at
    comparable rates. Uses obj._extract_params, so the clamps and transforms are
    exactly the ones the physics loss will see in stage 2.
    """
    Xt = torch.FloatTensor(np.asarray(X_scaled, dtype=float)).to(device)
    Y = torch.FloatTensor(np.asarray(targets, dtype=float)).to(device)
    Wt = torch.FloatTensor(np.asarray(weights, dtype=float)).to(device)
    sd = Y.std(dim=0, keepdim=True).clamp_min(1e-6)

    optimizer = optim.Adam(obj.model.parameters(), lr=lr, weight_decay=weight_decay)
    n = Xt.shape[0]
    bs = max(8, min(int(batch_size), n))
    obj.model.train()
    obj.pretrain_history = []

    for ep in range(int(epochs)):
        perm = torch.randperm(n, device=device)
        total, nb = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            optimizer.zero_grad()
            pred = torch.cat(obj._extract_params(obj.model(Xt[idx])), dim=1)
            loss = (Wt[idx] * ((pred - Y[idx]) / sd) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(obj.model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item(); nb += 1
        obj.pretrain_history.append(total / max(nb, 1))
        if verbose and (ep + 1) % 50 == 0:
            print(f"  [pretrain] epoch {ep + 1}/{epochs}  loss={obj.pretrain_history[-1]:.5f}")

    print(f"[pretrain] {epochs} epochs on {n} compositions; "
          f"loss {obj.pretrain_history[0]:.4f} -> {obj.pretrain_history[-1]:.4f}")
    return obj.pretrain_history


class MetalPINN:
    def __init__(
            self,
            target_name: str, 
            optimal_descriptors: List[str], 
            n_feat: Optional[int] = 64, 
            architecture=([64, 32],),
            act: str = "relu",
            out_act: str = "linear",
            dropout_rate = 0.2,):
        self.target_name = target_name
        self.OriginalOptimalDescriptors = optimal_descriptors
        self.n_feat = n_feat 
        self.num_neurons = architecture
        self.act_str = act
        self.out_act_str = out_act
        self.dropout_rate = dropout_rate
        CBFV_Features = [feature for feature in self.OriginalOptimalDescriptors if feature!="temp" and feature!="class"]
        # Subsetting the num feature set 
        if n_feat is None: 
            n_feat = len(CBFV_Features)
        if self.n_feat: # Ensures correct n_feat fed in
            CBFV_Top_Features = CBFV_Features[0:self.n_feat]
        self.features = CBFV_Top_Features
        print(f"Initializing MetalPINN for target {target_name}")
        # Standardizaing/Imputing Attributes 
        self.xscale = None
        self._scaler = None
        self._imputer = None
        self.scale_impute = None
        self.temp_min = None
        self.temp_max = None
        self.temp_range = None
        self.model = self._build_network().to(device)
    def activation_from_string(self, act) -> nn.Module: 
        act = act.lower()
        if act=="relu":
            return nn.ReLU()
        elif act=="sigmoid":
            return nn.Sigmoid()
        elif act=="tanh":
            return nn.Tanh()
        elif act=="elu":
            return nn.ELU()
        elif act=="linear":
            return nn.Identity()
        else: 
            raise ValueError("Invalid Activation Function Used")
    def _build_network(self) -> nn.Module:
        class MetalArchitecture(nn.Module):
            def __init__(self, n_input_features, num_neurons_config, act, out_act, dropout_rate):
                super().__init__()
                def activation_from_string(act) -> nn.Module: 
                    act = act.lower()
                    if act=="relu":
                        return nn.ReLU()
                    elif act=="sigmoid":
                        return nn.Sigmoid()
                    elif act=="tanh":
                        return nn.Tanh()
                    elif act=="elu":
                        return nn.ELU()
                    elif act=="linear":
                        return nn.Identity()
                    else: 
                        raise ValueError("Invalid Activation Function Used")
                def make_block(in_dim: int, hidden_layers_dims: List[int]) -> Tuple[nn.Module, int]:
                    if not hidden_layers_dims:
                        return nn.Identity(), in_dim
                        
                    layers = []
                    current_dim = in_dim
                    for out_dim_neurons in hidden_layers_dims:
                        layers.append(nn.Linear(current_dim, out_dim_neurons))
                        layers.append(nn.LayerNorm(out_dim_neurons))
                        layers.append(activation_from_string(act))
                        layers.append(nn.Dropout(p=dropout_rate))
                        current_dim = out_dim_neurons
                    return nn.Sequential(*layers), current_dim
                
                # flexible dynamic blocks 
                self.blocks = nn.ModuleList()
                input_features = n_input_features
                for block_layer in num_neurons_config: 
                    block_i, shared_dim_i = make_block(input_features, block_layer)
                    input_features = shared_dim_i
                    self.blocks.append(block_i)

                final_process_layers = []
                final_linear = nn.Linear(input_features, 3)
                nn.init.normal_(final_linear.weight, mean=0.0, std=0.01)
                with torch.no_grad():
                    final_linear.bias.copy_(torch.tensor([-6.46, 0.18, -0.71]))  # medians of the per-composition fit
                final_process_layers.append(final_linear)
                final_process_layers.append(activation_from_string(out_act))
                self.final_process = nn.Sequential(*final_process_layers)
            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                metal_params = self.final_process(x)
                return metal_params
            
        model = MetalArchitecture(len(self.features), self.num_neurons, self.act_str, self.out_act_str, self.dropout_rate)
        print(f"MetalPINN built with {sum(p.numel() for p in model.parameters())} trainable parameters.")
        return model

    def _extract_params(self, metal_params):
        """Physical parameters (lnA, lnBrel, n) from the raw network head.

        Split out of _calculate_conductivity so the warm-start phase supervises
        exactly the same clamped quantities the physics loss uses.
        """
        lnA_raw, lnBrel_raw, n_raw = metal_params.chunk(3, dim=1)
        # Clamp ranges from the observed parameter distribution (p1/p99, widened ~15%); soft leaky bounds
        lnA = _leaky_clamp(lnA_raw, min_val=-12.8, max_val=2.2)
        lnBrel = _leaky_clamp(lnBrel_raw, min_val=-3.7, max_val=5.0)
        n_metal = 0.1 + 4.0 * torch.sigmoid(n_raw)   # floor lowered to 0 (data has n down to ~0); ceiling raised from 3.31, which was binding
        return lnA, lnBrel, n_metal

    def _calculate_conductivity(self, metal_params, temps):
        lnA, lnBrel, n_metal = self._extract_params(metal_params)
        t_scaled = temps / 500
        A = torch.exp(lnA)
        B = A * torch.exp(lnBrel)
        log_resistivity = torch.log10(A * (t_scaled ** n_metal) + B + 1e-10)
        log_conductivity = -log_resistivity
        return log_conductivity

    def _set_scale_impute(self, impute_missing, xscale_before_impute, scaler=None, imputer=None):
        ### Directly taken/inspired from MODNet original source code, Citing De Breucks work
        """Sets the inner scaling and imputer mechanism."""
        if scaler is not None: self._scaler = scaler
        elif self.xscale == "minmax": self._scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        elif self.xscale == "standard": self._scaler = StandardScaler()
        else: self._scaler = None # No scaling

        if imputer is not None: self._imputer = imputer
        elif isinstance(impute_missing, str): self._imputer = SimpleImputer(missing_values=np.nan, strategy=impute_missing)
        elif impute_missing is not None : self._imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=impute_missing)
        else: self._imputer = None # No imputation
        steps = []
        if xscale_before_impute:
            if self._scaler: steps.append(("scaler", self._scaler))
            if self._imputer: steps.append(("imputer", self._imputer))
        else:
            if self._imputer: steps.append(("imputer", self._imputer))
            if self._scaler: steps.append(("scaler", self._scaler))
        
        if not steps: self.scale_impute = None # No pipeline if no steps
        else: self.scale_impute = Pipeline(steps)
    def fit(self, 
            train_df: pd.DataFrame, 
            train_target: pd.DataFrame, 
            val_df: Optional[pd.DataFrame] = None, 
            val_target: Optional[pd.DataFrame]=None,
            lr: float = 0.01, # learning rate
            epochs: int = 500, 
            patience: int = 50,
            delta: float = 0.0,
            batch_size: int = 128,
            weight_decay: float = 1e-5,
            xscale: Optional[str] = "standard", # e.g., "minmax", "standard", or None
            impute_missing: Optional[Union[float, str]] = 0, # e.g., "mean", "median", 0, or None
            xscale_before_impute: bool = True,
            loss_function: str = "mse",
            use_scheduler: bool = True,
            pretrain_epochs: int = 0,               # 0 disables the warm start
            pretrain_lr: Optional[float] = None,    # defaults to lr
            pretrain_identifiable_only: bool = True,
            verbose = False,):

        self.verbose = verbose

        self.xscale = xscale
        self._set_scale_impute(impute_missing, xscale_before_impute)
        self.weight_decay = weight_decay

        train_temp= train_df["temp"]
        self.temp_min = float(train_temp.min())
        self.temp_max = float(train_temp.max())
        self.temp_range = max(self.temp_max - self.temp_min, 1e-3)
        train_temp_tensor = torch.FloatTensor(train_temp.values).unsqueeze(1).to(device)

        X_train = train_df[self.features]
        X_train_np = np.asarray(X_train, dtype=float)
        if self.scale_impute is not None:
            X_train_final = self.scale_impute.fit_transform(X_train_np)
        else: 
            X_train_final = X_train_np
        
        y_train = train_target[self.target_name]
        y_train_np = np.asarray(y_train)

        # Stage 1: supervise (lnA, lnBrel, n) directly before the physics loss.
        if pretrain_epochs and pretrain_epochs > 0:
            ws = metal_warmstart_targets(
                train_df, train_target, self.target_name,
                n_max=4.0, identifiable_only=pretrain_identifiable_only,
            )
            if ws is not None and len(ws["rows"]) >= 5:
                _run_pretrain(
                    self, X_train_final[ws["rows"]], ws["params"], ws["weights"],
                    epochs=pretrain_epochs,
                    lr=(lr if pretrain_lr is None else pretrain_lr),
                    weight_decay=self.weight_decay, batch_size=batch_size,
                    verbose=verbose,
                )
            else:
                print("[pretrain] skipped: too few compositions with usable parameters")

        X_train_tensor = torch.FloatTensor(X_train_final).to(device)
        y_train_tensor = torch.FloatTensor(y_train_np).unsqueeze(1).to(device)

        train_dataset = TensorDataset(X_train_tensor, train_temp_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        # In the case we have validation set
        val_loader = None
        if val_df is not None and val_target is not None:
            val_temp = val_df["temp"]
            val_temp_tensor = torch.FloatTensor(val_temp.values).unsqueeze(1).to(device)
            X_val = val_df[self.features]
            X_val_np = np.asarray(X_val)
            if self.scale_impute is not None:
                X_val_final = self.scale_impute.transform(X_val_np)
            else:
                X_val_final = X_val_np
            y_val = val_target[self.target_name]
            y_val_np = np.asarray(y_val)

            # tensor  shaping
            X_val_tensor = torch.FloatTensor(X_val_final).to(device)
            y_val_tensor = torch.FloatTensor(y_val_np).unsqueeze(1).to(device)

            val_dataset = TensorDataset(X_val_tensor, val_temp_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False, shuffle=False)

        # Defining optimizer requirements
        criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        def r2_criterion(y_pred, y_true):
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
            r2 = 1 - ss_res / ss_tot
            return r2

        early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=self.verbose)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        if use_scheduler:
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-7)
        else:
            scheduler = None

        def physics_loss(metal_output, targets, temps):
            log_cond_metal  = self._calculate_conductivity(metal_output, temps)
            regression_loss = criterion(log_cond_metal, targets)
            mae_report_loss = mae_criterion(log_cond_metal, targets)
            r2_report_loss = r2_criterion(log_cond_metal, targets)
            return regression_loss, mae_report_loss, r2_report_loss

        self.history = {'loss': [], 'val_loss': [], "mae_loss": [], "mae_val_loss": [], "r2_loss": [], "r2_val_loss": []}
        loss_name = "MSELoss"
        print(f"Starting training with epochs {epochs} and for loss function {loss_name}")
        for epoch in range(epochs): # epochs is whole int, so range
            self.model.train()
            epoch_train_loss = 0.0 # initiliazing training loss, so this can be summated later
            epoch_train_mae_score = 0.0
            epoch_train_r2_score = 0.0
            for inputs, temps, targets in train_loader: 
                # ensuring they are on the devicde if pin_memory=True and CUDA available 
                inputs, temps, targets = inputs.to(device), temps.to(device), targets.to(device)
                optimizer.zero_grad() # clears the grasidnets fromt he previous generation
                metal_output = self.model(inputs)
                loss, mae_score_loss, r2_score_loss = physics_loss(metal_output, targets, temps) # this is a tensor that can be used to backpropagate the network
                epoch_train_loss += loss.item()
                epoch_train_mae_score += mae_score_loss.item()
                epoch_train_r2_score += r2_score_loss.item() # because it is not in torch
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
            current_train_loss = epoch_train_loss / len(train_loader)
            current_mae_train_score = epoch_train_mae_score / len(train_loader)
            current_r2_train_score = epoch_train_r2_score / len(train_loader)
            self.history['loss'].append(current_train_loss)
            self.history["mae_loss"].append(current_mae_train_score)
            self.history["r2_loss"].append(current_r2_train_score)

            if val_loader:
                self.model.eval() # set the model in evaluation mode
                epoch_val_loss = 0.0
                epoch_val_mae_score = 0.0
                epoch_val_r2_score = 0.0
                with torch.no_grad(): # this sets it in a mode where we dont alter the gradients/weights of the model. we only update it during training 
                    for inputs, temps, targets in val_loader: 
                        inputs, temps, targets = inputs.to(device), temps.to(device), targets.to(device)
                        metal_output = self.model(inputs)
                        val_loss, val_mae_score, val_r2_score = physics_loss(metal_output, targets, temps) # this is a tensor that can be used to backpropagate the network 
                        epoch_val_loss += val_loss.item()
                        epoch_val_mae_score += val_mae_score.item()
                        epoch_val_r2_score += val_r2_score.item()
                        
                    current_val_loss = epoch_val_loss / len(val_loader)
                    current_val_mae_score = epoch_val_mae_score / len(val_loader)
                    current_val_r2_score = epoch_val_r2_score / len(val_loader)
                    
                    self.history['val_loss'].append(current_val_loss)
                    self.history["mae_val_loss"].append(current_val_mae_score)
                    self.history["r2_val_loss"].append(current_val_r2_score)
                    
                if scheduler: 
                    scheduler.step(current_val_loss)                
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - Train {loss_name}: {current_train_loss:.6f} - Val {loss_name}: {current_val_loss:.6f}")
                
                early_stopping(current_val_loss, self.model)
                if early_stopping.early_stop:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break
            else: # No validation loader
                self.history['val_loss'].append(None)
                if verbose and (epoch + 1) % 10 == 0 : # Print every 10 epochs if no validation
                     print(f"Epoch {epoch + 1}/{epochs} - Train {loss_name}: {current_train_loss:.6f}")

        if early_stopping.best_model_state_dict:
            self.model.load_state_dict(early_stopping.best_model_state_dict)
            print(f"Loaded best model with validation loss: {-early_stopping.best_score:.6f}")
        elif val_loader is None and epochs > 0:
            print("Training finished. No validation set, using model from last epoch.")
        
        return self.history

    def predict(self, 
                test_df: pd.DataFrame,
                keep_training=False,
                ):
        
        # Preparing test_df through a series of tensors
        test_temp = test_df["temp"]
        test_temp_tensor = torch.FloatTensor(test_temp.values).unsqueeze(1).to(device)
        X_test = test_df[self.features] # dotn need the y_test tensor as we are not calculating a loss yet
        X_test_np = np.asarray(X_test)
        if self.scale_impute is not None: 
            X_test_final = self.scale_impute.transform(X_test_np)
        else:
            X_test_final = X_test_np
        X_test_tensor = torch.FloatTensor(X_test_final).to(device)
        # Physical calculations
        if keep_training==False: 
            self.model.eval()
        with torch.no_grad():
            metal_params_test = self.model(X_test_tensor)
            log_cond_m = self._calculate_conductivity(metal_params_test, test_temp_tensor)
        
        predictions_np = log_cond_m.detach().cpu().numpy()
        return predictions_np

    def save(self, filename: str):
        """Save the trained model to disk."""
        state = {
            'target_name': self.target_name,
            'optimal_descriptors': self.OriginalOptimalDescriptors,
            'n_feat': self.n_feat,
            'architecture': self.num_neurons,
            'act': self.act_str,
            'out_act': self.out_act_str,
            'model_state_dict': self.model.state_dict(),
            'xscale': self.xscale,
            'scale_impute_pipeline': self.scale_impute,
            'dropout_rate': self.dropout_rate,
        }
        torch.save(state, filename)
        print(f"Model saved to {filename}")

    @staticmethod
    def load(filename: str) -> "MetalPINN":
        """Load a saved model checkpoint created by `save()`."""
        print(f"Loading model from {filename}")

        try:
            state = torch.load(filename, map_location=device, weights_only=False)
        except Exception as e:
            print(f"Primary load failed ({e}). Retrying without map_location …")
            state = torch.load(filename, weights_only=False)

        # Re-instantiate the network skeleton using saved meta-data
        model_instance = MetalPINN(
            target_name=state['target_name'],
            optimal_descriptors=state['optimal_descriptors'],
            n_feat=state['n_feat'],
            architecture=state['architecture'],
            act=state['act'],
            out_act=state['out_act'],
            dropout_rate=state.get('dropout_rate', 0.2),
        )

        # Restore weights
        model_instance.model.load_state_dict(state['model_state_dict'])

        # Restore preprocessing pipeline
        model_instance.xscale = state.get('xscale')
        model_instance.scale_impute = state.get('scale_impute_pipeline')

        # Switch to eval mode
        model_instance.model.eval()

        print("Model loaded successfully.")
        return model_instance

class SemiconductorPINN:
    def __init__(
            self,
            target_name: str, 
            optimal_descriptors: List[str], 
            n_feat: Optional[int] = 64, 
            architecture=([1024], [512]),
            act: str = "relu",
            out_act: str = "linear",
            dropout_rate = 0.2,
            ):
        self.target_name = target_name
        self.OriginalOptimalDescriptors = optimal_descriptors
        self.n_feat = n_feat
        self.num_neurons = architecture
        self.act_str = act
        self.out_act_str = out_act
        self.dropout_rate = dropout_rate
        CBFV_Features = [feature for feature in self.OriginalOptimalDescriptors if feature!="temp" and feature!="class"]
        # Subsetting the num feature set 
        if n_feat is None: 
            n_feat = len(CBFV_Features)
        if self.n_feat: # Ensures correct n_feat fed in
            CBFV_Top_Features = CBFV_Features[0:self.n_feat]
        self.features = CBFV_Top_Features
        print(f"Initializing SemiconductorPINN for target {target_name}")
        # Standardizaing/Imputing Attributes 
        self.xscale = None
        self._scaler = None
        self._imputer = None
        self.scale_impute = None
        self.model = self._build_network().to(device)
    def activation_from_string(self, act) -> nn.Module: 
        act = act.lower()
        if act=="relu":
            return nn.ReLU()
        elif act=="sigmoid":
            return nn.Sigmoid()
        elif act=="tanh":
            return nn.Tanh()
        elif act=="elu":
            return nn.ELU()
        elif act=="linear":
            return nn.Identity()
        else: 
            raise ValueError("Invalid Activation Function Used")
    def _build_network(self) -> nn.Module:
        class SemiconductorArchitecture(nn.Module):
            def __init__(self, n_input_features, num_neurons_config, act, out_act, dropout_rate):
                super().__init__()
                def activation_from_string(act) -> nn.Module: 
                    act = act.lower()
                    if act=="relu":
                        return nn.ReLU()
                    elif act=="sigmoid":
                        return nn.Sigmoid()
                    elif act=="tanh":
                        return nn.Tanh()
                    elif act=="elu":
                        return nn.ELU()
                    elif act=="linear":
                        return nn.Identity()
                    else: 
                        raise ValueError("Invalid Activation Function Used")
                def make_block(in_dim: int, hidden_layers_dims: List[int]) -> Tuple[nn.Module, int]:
                    if not hidden_layers_dims:
                        return nn.Identity(), in_dim
                        
                    layers = []
                    current_dim = in_dim
                    for out_dim_neurons in hidden_layers_dims:
                        layers.append(nn.Linear(current_dim, out_dim_neurons))
                        layers.append(nn.LayerNorm(out_dim_neurons))
                        layers.append(activation_from_string(act))
                        layers.append(nn.Dropout(p=dropout_rate))
                        current_dim = out_dim_neurons
                    return nn.Sequential(*layers), current_dim
                
                self.block_list = nn.ModuleList()
                input_features = n_input_features
                for block_layer in num_neurons_config: 
                    block_i, shared_dim_i = make_block(input_features, block_layer)
                    input_features = shared_dim_i
                    self.block_list.append(block_i)

                final_process_layers = []
                final_linear = nn.Linear(input_features, 2)
                nn.init.normal_(final_linear.weight, mean=0.0, std=0.01)
                with torch.no_grad():
                    final_linear.bias.copy_(torch.tensor([4.67, -2.96]))  # medians of the per-composition Arrhenius fit: lnA_semi 4.67, Ea 0.050 eV
                final_process_layers.append(final_linear)
                final_process_layers.append(activation_from_string(out_act))
                self.final_process = nn.Sequential(*final_process_layers)
            def forward(self, x):
                for block in self.block_list:
                    x = block(x)
                semi_params = self.final_process(x)
                return semi_params
        model = SemiconductorArchitecture(len(self.features), self.num_neurons, self.act_str, self.out_act_str, self.dropout_rate)
        print(f"SemiconductorPINN built with {sum(p.numel() for p in model.parameters())} trainable parameters.")
        return model

    def _extract_params(self, semi_params):
        """Physical parameters (lnA, Ea) from the raw network head.

        Split out of _calculate_conductivity so the warm-start phase supervises
        exactly the same clamped quantities the physics loss uses.
        """
        lnA_semi, Ea_semi_raw = semi_params.chunk(2, dim=1)
        # Clamp ranges from the observed parameter distribution (p1/p99, widened ~15%); soft leaky bounds
        lnA_semi = _leaky_clamp(lnA_semi, min_val=-4.3, max_val=11.8)
        Ea_semi = torch.nn.functional.softplus(Ea_semi_raw) + 1e-6
        Ea_semi = _leaky_clamp(Ea_semi, min_val=0.0, max_val=0.45)   # soft cap to the dataset Ea range (Ea had no upper bound before)
        return lnA_semi, Ea_semi

    def _calculate_conductivity(self, semi_params, temps):
        lnA_semi, Ea_semi = self._extract_params(semi_params)
        ln_cond_semi = lnA_semi - (Ea_semi / (K_BOLTZ * temps))
        log_cond_semi = ln_cond_semi / _LN10
        log_cond_semi = _leaky_clamp(log_cond_semi, min_val=-4.3, max_val=4.4)
        return log_cond_semi, Ea_semi # for reporting

    def _set_scale_impute(self, impute_missing, xscale_before_impute, scaler=None, imputer=None):
        ### Directly taken/inspired from MODNet original source code, Citing De Breucks work
        """Sets the inner scaling and imputer mechanism."""
        if scaler is not None: self._scaler = scaler
        elif self.xscale == "minmax": self._scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        elif self.xscale == "standard": self._scaler = StandardScaler()
        else: self._scaler = None # No scaling

        if imputer is not None: self._imputer = imputer
        elif isinstance(impute_missing, str): self._imputer = SimpleImputer(missing_values=np.nan, strategy=impute_missing)
        elif impute_missing is not None : self._imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=impute_missing)
        else: self._imputer = None # No imputation
        steps = []
        if xscale_before_impute:
            if self._scaler: steps.append(("scaler", self._scaler))
            if self._imputer: steps.append(("imputer", self._imputer))
        else:
            if self._imputer: steps.append(("imputer", self._imputer))
            if self._scaler: steps.append(("scaler", self._scaler))
        
        if not steps: self.scale_impute = None # No pipeline if no steps
        else: self.scale_impute = Pipeline(steps)
    def fit(self, 
            train_df: pd.DataFrame, 
            train_target: pd.DataFrame, 
            val_df: Optional[pd.DataFrame] = None, 
            val_target: Optional[pd.DataFrame]=None,
            lr: float = 0.01, # learning rate
            epochs: int = 500, 
            patience: int = 50,
            delta: float = 0.0,
            batch_size: int = 128,
            weight_decay: float = 1e-5,
            xscale: Optional[str] = "standard", # e.g., "minmax", "standard", or None
            impute_missing: Optional[Union[float, str]] = 0, # e.g., "mean", "median", 0, or None
            xscale_before_impute: bool = True,
            loss_function: str = "mse",
            use_scheduler: bool = False,
            pretrain_epochs: int = 0,               # 0 disables the warm start
            pretrain_lr: Optional[float] = None,    # defaults to lr
            verbose = False,):

        self.verbose = verbose

        self.xscale = xscale
        self._set_scale_impute(impute_missing, xscale_before_impute)
        self.weight_decay = weight_decay

        train_temp= train_df["temp"]
        train_temp_tensor = torch.FloatTensor(train_temp.values).unsqueeze(1).to(device)

        X_train = train_df[self.features]
        X_train_np = np.asarray(X_train, dtype=float)
        if self.scale_impute is not None:
            X_train_final = self.scale_impute.fit_transform(X_train_np)
        else: 
            X_train_final = X_train_np
        
        y_train = train_target[self.target_name]
        y_train_np = np.asarray(y_train)

        # Stage 1: supervise (lnA, Ea) directly before the physics loss.
        if pretrain_epochs and pretrain_epochs > 0:
            ws = semiconductor_warmstart_targets(train_df, train_target, self.target_name)
            if ws is not None and len(ws["rows"]) >= 5:
                _run_pretrain(
                    self, X_train_final[ws["rows"]], ws["params"], ws["weights"],
                    epochs=pretrain_epochs,
                    lr=(lr if pretrain_lr is None else pretrain_lr),
                    weight_decay=self.weight_decay, batch_size=batch_size,
                    verbose=verbose,
                )
            else:
                print("[pretrain] skipped: too few compositions with usable parameters")

        X_train_tensor = torch.FloatTensor(X_train_final).to(device)
        y_train_tensor = torch.FloatTensor(y_train_np).unsqueeze(1).to(device)

        train_dataset = TensorDataset(X_train_tensor, train_temp_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        # In the case we have validation set
        val_loader = None
        if val_df is not None and val_target is not None:
            val_temp = val_df["temp"]
            val_temp_tensor = torch.FloatTensor(val_temp.values).unsqueeze(1).to(device)
            X_val = val_df[self.features]
            X_val_np = np.asarray(X_val)
            if self.scale_impute is not None:
                X_val_final = self.scale_impute.transform(X_val_np)
            else:
                X_val_final = X_val_np
            y_val = val_target[self.target_name]
            y_val_np = np.asarray(y_val)
            
            # tensor  shaping
            X_val_tensor = torch.FloatTensor(X_val_final).to(device)
            y_val_tensor = torch.FloatTensor(y_val_np).unsqueeze(1).to(device)

            val_dataset = TensorDataset(X_val_tensor, val_temp_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False, shuffle=False)    

        # Defining optimizer requirements
        criterion = nn.SmoothL1Loss(beta=1.0)
        mae_criterion = nn.L1Loss()
        def r2_criterion(y_pred, y_true):
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
            r2 = 1 - ss_res / ss_tot
            return r2

        early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=self.verbose)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        if use_scheduler:
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-7)
        else:
            scheduler = None

        # defining loss calculation
        def physics_loss(semi_output, targets, temps):
            log_cond_semi, Ea_semi = self._calculate_conductivity(semi_output, temps)
            regression_loss = criterion(log_cond_semi, targets)
            mae_loss = mae_criterion(log_cond_semi, targets)
            r2_loss = r2_criterion(log_cond_semi, targets)
            return regression_loss, mae_loss, r2_loss
        
        self.history = {'loss': [], 'val_loss': [], "mae_loss": [], "mae_val_loss": [], "r2_loss": [], "r2_val_loss": []}
        loss_name = "Huber"
        print(f"Starting training with epochs {epochs} and for loss function {loss_name}")
        for epoch in range(epochs): # epochs is whole int, so range
            self.model.train()
            epoch_train_loss = 0.0 # initiliazing training loss, so this can be summated later
            epoch_train_mae_score = 0.0
            epoch_train_r2_score = 0.0
            for inputs, temps, targets in train_loader: 
                # ensuring they are on the devicde if pin_memory=True and CUDA available 
                inputs, temps, targets = inputs.to(device), temps.to(device), targets.to(device)
                optimizer.zero_grad() # clears the grasidnets from th previous generation
                semiconductor_output = self.model(inputs)
                loss, mae_score_loss, r2_score_loss = physics_loss(semiconductor_output, targets, temps) # this is a tensor that can be used to backpropagate the network
                epoch_train_loss += loss.item()
                epoch_train_mae_score += mae_score_loss.item()
                epoch_train_r2_score += r2_score_loss.item() # because it is not in torch
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

            current_train_loss = epoch_train_loss / len(train_loader)
            current_mae_train_score = epoch_train_mae_score / len(train_loader)
            current_mae_r2_score = epoch_train_r2_score / len(train_loader)
            self.history['loss'].append(current_train_loss)
            self.history["mae_loss"].append(current_mae_train_score)
            self.history["r2_loss"].append(current_mae_r2_score)

            if val_loader:
                self.model.eval() # set the model in evaluation mode
                epoch_val_loss = 0.0
                epoch_val_mae_score = 0.0
                epoch_val_r2_score = 0.0
                with torch.no_grad(): # this sets it in a mode where we dont alter the gradients/weights of the model. we only update it during training
                    for inputs, temps, targets in val_loader:
                        inputs, temps, targets = inputs.to(device), temps.to(device), targets.to(device)
                        semiconductor_output = self.model(inputs)
                        val_loss, val_mae_score, val_r2_score = physics_loss(semiconductor_output, targets, temps)
                        epoch_val_loss += val_loss.item()
                        epoch_val_mae_score += val_mae_score.item()
                        epoch_val_r2_score += val_r2_score.item()

                    current_val_loss = epoch_val_loss / len(val_loader)
                    current_val_mae_score = epoch_val_mae_score / len(val_loader)
                    current_val_r2_score = epoch_val_r2_score / len(val_loader)

                    self.history['val_loss'].append(current_val_loss)
                    self.history["mae_val_loss"].append(current_val_mae_score)
                    self.history["r2_val_loss"].append(current_val_r2_score)

                if scheduler:
                    scheduler.step(current_val_loss)
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - Train {loss_name}: {current_train_loss:.6f} - Val {loss_name}: {current_val_loss:.6f}")

                early_stopping(current_val_loss, self.model)
                if early_stopping.early_stop:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break
            else: # No validation loader
                self.history['val_loss'].append(None)
                if verbose and (epoch + 1) % 10 == 0 : # Print every 10 epochs if no validation
                     print(f"Epoch {epoch + 1}/{epochs} - Train {loss_name}: {current_train_loss:.6f}")

        if early_stopping.best_model_state_dict:
            self.model.load_state_dict(early_stopping.best_model_state_dict)
            print(f"Loaded best model with validation loss: {-early_stopping.best_score:.6f}")
        elif val_loader is None and epochs > 0:
            print("Training finished. No validation set, using model from last epoch.")
        
        return self.history

    def predict(self, 
                test_df: pd.DataFrame,
                keep_training=False,
                ):
        
        # Preparing test_df through a series of tensors
        test_temp = test_df["temp"]
        test_temp_tensor = torch.FloatTensor(test_temp.values).unsqueeze(1).to(device)
        X_test = test_df[self.features] # dotn need the y_test tensor as we are not calculating a loss yet
        X_test_np = np.asarray(X_test)
        if self.scale_impute is not None: 
            X_test_final = self.scale_impute.transform(X_test_np)
        else:
            X_test_final = X_test_np
        X_test_tensor = torch.FloatTensor(X_test_final).to(device)
        # Physical calculations
        if keep_training==False:
            self.model.eval()
        with torch.no_grad():
            semi_params_test = self.model(X_test_tensor)
            log_cond_s, Ea_semi = self._calculate_conductivity(semi_params_test, test_temp_tensor)
        
        predictions_np = log_cond_s.detach().cpu().numpy()
        Ea_semi = Ea_semi.detach().cpu().numpy()
        self.Test_Predicted_Ea = Ea_semi
        return predictions_np
    
    def save(self, filename: str):
        """Save the trained model to disk."""
        state = {
            'target_name': self.target_name,
            'optimal_descriptors': self.OriginalOptimalDescriptors,
            'n_feat': self.n_feat,
            'architecture': self.num_neurons,
            'act': self.act_str,
            'out_act': self.out_act_str,
            'model_state_dict': self.model.state_dict(),
            'xscale': self.xscale,
            'scale_impute_pipeline': self.scale_impute,
            'dropout_rate': self.dropout_rate,
        }
        torch.save(state, filename)
        print(f"Model saved to {filename}")

    @staticmethod
    def load(filename: str) -> "SemiconductorPINN":
        print(f"Loading model from {filename}")

        try:
            state = torch.load(filename, map_location=device, weights_only=False)
        except Exception as e:
            print(f"Primary load failed ({e}). Retrying without map_location …")
            state = torch.load(filename, weights_only=False)

        # Re-instantiate the network skeleton using saved meta-data
        model_instance = SemiconductorPINN(
            target_name=state['target_name'],
            optimal_descriptors=state['optimal_descriptors'],
            n_feat=state['n_feat'],
            architecture=state['architecture'],
            act=state['act'],
            out_act=state['out_act'],
            dropout_rate=state.get('dropout_rate', 0.2),
        )

        # Restore weights
        model_instance.model.load_state_dict(state['model_state_dict'])

        # Restore preprocessing pipeline
        model_instance.xscale = state.get('xscale')
        model_instance.scale_impute = state.get('scale_impute_pipeline')

        # Switch to eval mode
        model_instance.model.eval()

        print("Model loaded successfully.")
        return model_instance

# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=7, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state_dict = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state_dict = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f'Validation loss decreased to {val_loss:.6f}. saving the model!')
        elif score < self.best_score + self.delta: 
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('Early stopping triggered')
        else:
            self.best_score = score
            self.best_model_state_dict = copy.deepcopy(model.state_dict()) 
            if self.verbose:
                print(f'Validation loss decreased to {val_loss:.6f}. saving the model!')
            self.counter = 0


