
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset


from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset



class ScalerLoader:
    """
    Encapsulates Min–Max scaling + train/val/test splitting +
    DataLoader creation for PyTorch, fully robust to arbitrary
    (f_train, f_val, f_test) combos, including degenerate
    cases like (0,0,1), (1,0,0), etc., and supports inference-only.
    """

    def __init__(
        self,
        feature_cols: List[str],
        target_cols: Optional[List[str]] = None,
        scale_range: Tuple[float, float] = (0.0, 1.0),
    ):
        """
        Args:
            feature_cols: names of input columns (X)
            target_cols:  names of output columns (y); if None or empty,
                          you may only call get_inference_loader()
            scale_range:  (min, max) for Min–Max scaling
        """
        self.feature_cols = feature_cols
        self.target_cols = target_cols or []
        self.scale_range = scale_range

        self.input_scaler  = MinMaxScaler(feature_range=scale_range)
        self.output_scaler = MinMaxScaler(feature_range=scale_range)
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "ScalerLoader":
        """
        Fit scaler(s) on the DataFrame.  Always fits input_scaler;
        fits output_scaler only if target_cols is non-empty.
        """
        X = df[self.feature_cols].to_numpy()
        self.input_scaler.fit(X)

        if self.target_cols:
            y = df[self.target_cols].to_numpy()
            self.output_scaler.fit(y)

        self._fitted = True
        return self

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Scale X (and optionally y).  If X has zero rows, returns
        raw arrays without calling the scaler to avoid errors.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit(...) before transform(...)")

        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        if X_arr.shape[0] == 0:
            if y is None or not self.target_cols:
                return X_arr
            y_arr = y.to_numpy() if isinstance(y, pd.DataFrame) else y
            return X_arr, y_arr

        X_s = self.input_scaler.transform(X_arr)

        if y is None or not self.target_cols:
            return X_s

        y_arr = y.to_numpy() if isinstance(y, pd.DataFrame) else y
        y_s   = self.output_scaler.transform(
            y_arr.reshape(-1, len(self.target_cols))
        )
        return X_s, y_s

    def inverse_transform(
        self,
        X_s: np.ndarray,
        y_s: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Inverse‐scale.  Zero‐length inputs are passed through unchanged.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit(...) before inverse_transform(...)")

        if X_s.shape[0] == 0:
            if y_s is None or not self.target_cols:
                return X_s
            return X_s, y_s

        X = self.input_scaler.inverse_transform(X_s)
        if y_s is None or not self.target_cols:
            return X
        y = self.output_scaler.inverse_transform(y_s)
        return X, y

    def get_loaders(
        self,
        df: pd.DataFrame,
        f_train: float,
        f_val: float,
        f_test: float,
        batch_sizes: Tuple[int, int, int],
        shuffle_train: bool = True,
        random_state: int = 42,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """
        Split df into train/val/test (time‐ordered for test), scale, and
        return four DataLoaders: train, val, test, combined.
        Fully supports degenerate splits:
          - any of f_train/f_val/f_test == 1.0 (others 0.0)
          - empty splits become zero‐length loaders (shuffle=False)
        """
        if not self.target_cols:
            raise ValueError(
                "No target_cols defined; use get_inference_loader() instead."
            )
        if not np.isclose(f_train + f_val + f_test, 1.0):
            raise ValueError("f_train + f_val + f_test must sum to 1.0")

        n = len(df)
        # 1) fit on full df so scalers see global min/max
        self.fit(df)

        # 2) time‐ordered test split
        if f_test == 1.0:
            df_te, df_tv = df, df.iloc[:0]
        elif f_test == 0.0:
            df_te, df_tv = df.iloc[:0], df
        else:
            n_test = int(n * f_test)
            df_te  = df.iloc[-n_test:]
            df_tv  = df.iloc[:-n_test]

        # 3) train/val split on df_tv
        if f_val == 0.0:
            df_va, df_tr = df_tv.iloc[:0], df_tv
        elif f_train == 0.0:
            df_va, df_tr = df_tv, df_tv.iloc[:0]
        else:
            rel_val = f_val / (f_train + f_val)
            df_tr, df_va = train_test_split(
                df_tv, test_size=rel_val,
                shuffle=False, random_state=random_state
            )

        # 4) extract & scale (empty splits bypass scaler)
        X_tr_s, y_tr_s = self.transform(
            df_tr[self.feature_cols], df_tr[self.target_cols]
        )
        X_va_s, y_va_s = self.transform(
            df_va[self.feature_cols], df_va[self.target_cols]
        )
        X_te_s, y_te_s = self.transform(
            df_te[self.feature_cols], df_te[self.target_cols]
        )

        # 5) to torch.Tensor
        tX_tr, ty_tr = torch.from_numpy(X_tr_s).float(), torch.from_numpy(y_tr_s).float()
        tX_va, ty_va = torch.from_numpy(X_va_s).float(), torch.from_numpy(y_va_s).float()
        tX_te, ty_te = torch.from_numpy(X_te_s).float(), torch.from_numpy(y_te_s).float()

        # 6) build TensorDatasets
        ds_tr = TensorDataset(tX_tr, ty_tr)
        ds_va = TensorDataset(tX_va, ty_va)
        ds_te = TensorDataset(tX_te, ty_te)

        bs_tr, bs_va, bs_te = batch_sizes

        # 7) DataLoaders (disable shuffle if empty)
        loader_tr = DataLoader(
            ds_tr,
            batch_size=bs_tr,
            shuffle=shuffle_train and len(ds_tr) > 0
        )
        loader_va = DataLoader(ds_va, batch_size=bs_va, shuffle=False)
        loader_te = DataLoader(ds_te, batch_size=bs_te, shuffle=False)

        # 8) combined dataset & loader
        combined   = ConcatDataset([ds_tr, ds_va, ds_te])
        loader_all = DataLoader(combined, batch_size=bs_te, shuffle=False)

        return loader_tr, loader_va, loader_te, loader_all

    def get_inference_loader(
        self,
        df: pd.DataFrame,
        batch_size: int,
        shuffle: bool = False,
    ) -> DataLoader:
        """
        Build a feature‐only DataLoader for inference.  Must have called
        fit(...) on training data first.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit(...) before get_inference_loader()")

        X_s = self.transform(df[self.feature_cols])  # returns X_s only
        tX  = torch.from_numpy(X_s).float()
        ds  = TensorDataset(tX)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)





class ScalerLoaderV01:
    """
    Encapsulates feature/target Min–Max scaling (and inverse‐scaling) +
    train/val/test splitting + DataLoader creation for PyTorch.

    Now you only give `scale_range`, and every column (both X and y)
    is scaled from its own min/max into that range.
    """

    def __init__(
        self,
        feature_cols: List[str],
        target_cols: List[str],
        scale_range: Tuple[float, float] = (0.0, 1.0),
    ):
        """
        Args:
            feature_cols: list of column names to use as inputs (X).
            target_cols:  list of column names to use as outputs (y).
            scale_range:  (min, max) into which *every* column is scaled.
        """
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.scale_range = scale_range

        # Both scalers use the same feature_range:
        self.input_scaler  = MinMaxScaler(feature_range=scale_range)
        self.output_scaler = MinMaxScaler(feature_range=scale_range)

        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "ScalerLoader":
        """
        Fit both scalers on the entire DataFrame (each column to its own min/max).
        """
        X = df[self.feature_cols].to_numpy()
        y = df[self.target_cols].to_numpy()
        self.input_scaler .fit(X)
        self.output_scaler.fit(y)
        self._fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Scale X (and optionally y) into `scale_range`.  Must call `fit` first.
        """
        if not self._fitted:
            raise RuntimeError("Call fit(...) before transform(...)")

        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        X_s   = self.input_scaler.transform(X_arr)

        if y is None:
            return X_s

        y_arr = y.to_numpy() if isinstance(y, pd.DataFrame) else y
        y_s   = self.output_scaler.transform(y_arr.reshape(-1, len(self.target_cols)))
        return X_s, y_s

    def inverse_transform(
        self,
        X_s: np.ndarray,
        y_s: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Undo the scaling on X_s (and optionally y_s).
        """
        if not self._fitted:
            raise RuntimeError("Call fit(...) before inverse_transform(...)")

        X = self.input_scaler.inverse_transform(X_s)
        if y_s is None:
            return X

        y = self.output_scaler.inverse_transform(y_s)
        return X, y

    def get_loaders(
        self,
        df: pd.DataFrame,
        f_train: float,
        f_val: float,
        f_test: float,
        batch_sizes: Tuple[int, int, int],
        shuffle_train: bool = True,
        random_state: int = 42,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """
        Split df into train/val/test (time‐ordered test), scale into `scale_range`,
        and wrap into four DataLoaders: train, val, test, combined.
        """
        if not np.isclose(f_train + f_val + f_test, 1.0):
            raise ValueError("f_train + f_val + f_test must sum to 1.0")

        n = len(df)
        n_test      = int(n * f_test)
        n_train_val = n - n_test

        # Fit on *all* data so scaler knows global mins/maxs
        self.fit(df)

        df_tv = df.iloc[:n_train_val]
        df_te = df.iloc[n_train_val:]

        rel_val = f_val / (f_train + f_val)
        df_tr, df_va = train_test_split(
            df_tv,
            test_size=rel_val,
            shuffle=False,
            random_state=random_state
        )

        # arrays
        X_tr = df_tr[self.feature_cols].to_numpy()
        y_tr = df_tr[self.target_cols].to_numpy()
        X_va = df_va[self.feature_cols].to_numpy()
        y_va = df_va[self.target_cols].to_numpy()
        X_te = df_te[self.feature_cols].to_numpy()
        y_te = df_te[self.target_cols].to_numpy()

        # scale
        X_tr_s, y_tr_s = self.transform(X_tr, y_tr)
        X_va_s, y_va_s = self.transform(X_va, y_va)
        X_te_s, y_te_s = self.transform(X_te, y_te)

        # to PyTorch tensors
        tX_tr = torch.from_numpy(X_tr_s).float()
        ty_tr = torch.from_numpy(y_tr_s).float()
        tX_va = torch.from_numpy(X_va_s).float()
        ty_va = torch.from_numpy(y_va_s).float()
        tX_te = torch.from_numpy(X_te_s).float()
        ty_te = torch.from_numpy(y_te_s).float()

        # datasets & loaders
        ds_tr = TensorDataset(tX_tr, ty_tr)
        ds_va = TensorDataset(tX_va, ty_va)
        ds_te = TensorDataset(tX_te, ty_te)

        bs_tr, bs_va, bs_te = batch_sizes
        loader_tr = DataLoader(ds_tr, batch_size=bs_tr, shuffle=shuffle_train)
        loader_va = DataLoader(ds_va, batch_size=bs_va, shuffle=False)
        loader_te = DataLoader(ds_te, batch_size=bs_te, shuffle=False)

        combined = ConcatDataset([ds_tr, ds_va, ds_te])
        loader_all = DataLoader(combined, batch_size=bs_te, shuffle=False)

        return loader_tr, loader_va, loader_te, loader_all



# === Example usage ===
if __name__ == "__main__":
    import pandas as pd

    # suppose df has columns "t","j","U"
    df = pd.DataFrame({
        "t": np.linspace(0,10,100),
        "j": np.sin(np.linspace(0,10,100)),
        "T": np.cos(np.linspace(0, 10, 100)),
        "U": np.cos(np.linspace(0,10,100))
    })

    scaler_loader = ScalerLoader(
        feature_cols=["t", "j", "T"],
        target_cols=["U"],
        scale_range=(0, 1),
    )

    train_loader, val_loader, test_loader, all_loader = scaler_loader.get_loaders(
        df,
        f_train=0.6,
        f_val=0.2,
        f_test=0.2,
        batch_sizes=(32, 64, 128),
    )

