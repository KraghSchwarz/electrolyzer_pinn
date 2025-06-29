import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import DataLoader
from tqdm import trange
import abc
from typing import Dict, Tuple, Optional, Any, Union
from elec_pinn.data.loader import ScalerLoader
import json
import time
from datetime import datetime

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from elec_pinn.models.MLP import MLP_Softplus

# =======================================================
# Base PINN Abstract Class
# =======================================================
class BasePINN(nn.Module, abc.ABC):
    """
    Abstract base class for our Physics Informed Neural Networks (PINNs) implementations.

    This class provides the basic structure and common functionality for different PINN implementations.
    Subclasses must implement the forward method and can customize other methods as needed.
    """

    def __init__(self,
                 input_dim: int = 2,
                 f_hidden_dim: int = 32,
                 g_hidden_dim: int = 32,
                 f_layers: int = 4,
                 g_layers: int = 4,
                 lr: float = 1e-4,
                 pde_weight: float = 1.0):
        """
        Initialize the Base PINN.

        Args:
            f_hidden_dim: Number of nodes in hidden layers of the F network
            g_hidden_dim: Number of nodes in hidden layers of the G network
            f_layers: Number of layers in the F network
            g_layers: Number of layers in the G network
            lr: Learning rate for optimizers
            pde_weight: Weight for the PDE loss term
        """

        super().__init__()

        # Common parameters
        self.input_dim = input_dim
        self.f_hidden_dim = f_hidden_dim
        self.g_hidden_dim = g_hidden_dim
        self.f_layers = f_layers
        self.g_layers = g_layers
        self.lr = lr

        # Loss weights
        self.alpha = pde_weight

        # Loss function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Model tracking
        self.best_F_state = None
        self.best_G_state = None
        self.best_val_loss = float('inf')
        self.best_training_loss = float('inf')
        self.best_model_index = None

        # Storage for visualization
        self.epoch_solutions = []
        self.epoch_x_values = None
        self.epoch_losses = {"total": [], "data": [], "pde": []}
        self.val_losses = {"total": [], "data": [], "pde": []}

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize networks and optimizers (will be implemented in subclasses)
        self._initialize_networks()
        self._initialize_optimizers()

        # Move to device
        self.to(self.device)

    @abc.abstractmethod
    def _initialize_networks(self):
        """Initialize neural networks specific to each PINN implementation."""
        pass

    @abc.abstractmethod
    def _initialize_optimizers(self):
        """Initialize optimizers for the neural networks."""
        pass

    @abc.abstractmethod
    def forward(self, X: torch.Tensor, return_aux: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the PINN.

        Args:
            X: Input tensor of shape (N, 2) with columns [t, j]
            return_aux: Whether to return auxiliary outputs

        Returns:
            Dictionary containing model outputs
        """
        pass


    def train_one_epoch(self, data_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Train the model for one epoch.

        Args:
            data_loader: DataLoader containing training data

        Returns:
            Tuple of (total_loss, data_loss, pde_loss) for the epoch
        """
        self.train()
        epoch_total_loss = 0.0
        epoch_data_loss = 0.0
        epoch_pde_loss = 0.0
        num_batches = len(data_loader)

        for X_data, Y_data in data_loader:
            X_data = X_data.to(self.device)
            Y_data = Y_data.to(self.device)

            # Forward pass
            out = self.forward(X_data)
            u_pred = out["u_deg_norm"]
            f_pde = out.get("f_pde", torch.zeros_like(u_pred))  # Some implementations might not have PDE terms

            # Data loss
            data_loss = self.loss_fn(u_pred, Y_data)

            # PDE loss
            zeros_target = torch.zeros_like(f_pde, device=self.device)
            pde_loss = self.loss_fn(f_pde, zeros_target)

            # Total loss
            total_loss = data_loss + self.alpha * pde_loss

            # Backpropagation and optimization
            self._zero_grad_optimizers()
            total_loss.backward()
            self._step_optimizers()

            # Accumulate losses
            epoch_total_loss += total_loss.item()
            epoch_data_loss += data_loss.item()
            epoch_pde_loss += pde_loss.item()

        # Calculate average losses
        avg_total_loss = epoch_total_loss / num_batches
        avg_data_loss = epoch_data_loss / num_batches
        avg_pde_loss = epoch_pde_loss / num_batches

        return avg_total_loss, avg_data_loss, avg_pde_loss

    def _zero_grad_optimizers(self):
        """Zero gradients for all optimizers."""
        # Implementation will be in subclasses
        pass

    def _step_optimizers(self):
        """Step all optimizers."""
        # Implementation will be in subclasses
        pass

    def validate(self, val_loader: DataLoader) -> Union[
        Tuple[float, float, float], Dict[str, torch.Tensor]]:
        """
        Validate the model on a validation dataset.

        Args:
            val_loader: DataLoader containing validation data

        Returns:
            If final_validation is True: All model outputs
            Otherwise: Tuple of (val_loss, val_data_loss, val_pde_loss)
        """
        self.eval()
        val_total_loss = 0.0
        val_data_loss = 0.0
        val_pde_loss = 0.0

        for X_val, Y_val in val_loader:
            X_val = X_val.to(self.device)
            Y_val = Y_val.to(self.device)
            X_val.requires_grad_(True)

            # Forward pass
            out = self.forward(X_val)
            u_pred = out["u_deg_norm"]
            f_pde = out.get("f_pde", torch.zeros_like(u_pred))

            # Calculate losses
            data_loss = self.loss_fn(u_pred, Y_val)
            zeros_target = torch.zeros_like(f_pde, device=self.device)
            pde_loss = self.loss_fn(f_pde, zeros_target)

            total_loss = data_loss + self.alpha * pde_loss

            val_total_loss += total_loss.item()
            val_data_loss += data_loss.item()
            val_pde_loss += pde_loss.item()

        num_batches = len(val_loader)
        avg_val_loss = val_total_loss / num_batches
        avg_val_data_loss = val_data_loss / num_batches
        avg_val_pde_loss = val_pde_loss / num_batches

        self.val_losses["total"].append(avg_val_loss)
        self.val_losses["data"].append(avg_val_data_loss)
        self.val_losses["pde"].append(avg_val_pde_loss)

        return avg_val_loss, avg_val_data_loss, avg_val_pde_loss

    def _store_solution_snapshot(self, data_loader: DataLoader):
        """
        Store current solution predictions for visualization.

        Args:
            data_loader: DataLoader containing data to visualize
        """
        X_list, Y_list = [], []
        for X_data, Y_data in data_loader:
            X_list.append(X_data.to(self.device))
            Y_list.append(Y_data.to(self.device))
        X_all = torch.cat(X_list, dim=0)
        Y_all = torch.cat(Y_list, dim=0)

        if not hasattr(self, 'epoch_data'):
            self.epoch_data = []
            self.target_data = {
                'inputs': X_all.detach().cpu().numpy(),
                'targets': Y_all.detach().cpu().numpy()
            }

        #with torch.no_grad():
        out = self.forward(X_all)
        u_pred = out["u_deg_norm"]
        u_values = u_pred.detach().cpu().numpy()

        self.epoch_data.append({
            'inputs': X_all.detach().cpu().numpy(),
            'predictions': u_values
        })

    def train_model(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                    num_epochs: int = 100, save_freq: int = 10, patience: int = 20) -> Dict[str, Any]:
        """
        Main training loop with early stopping and solution snapshotting.

        Args:
            train_loader: DataLoader containing training data
            val_loader: Optional DataLoader containing validation data
            num_epochs: Maximum number of training epochs
            save_freq: Frequency to save model snapshots
            patience: Number of epochs to wait for improvement before early stopping

        Returns:
            Dictionary containing training statistics
        """
        best_loss = float('inf')
        patience_counter = 0
        t_bar = trange(1, num_epochs + 1, desc="Training")

        for epoch in t_bar:
            train_loss, data_loss, pde_loss = self.train_one_epoch(train_loader)
            self.epoch_losses["total"].append(train_loss)
            self.epoch_losses["data"].append(data_loss)
            self.epoch_losses["pde"].append(pde_loss)

            if val_loader is not None:
                val_loss, val_data_loss, val_pde_loss = self.validate(val_loader)
                eval_loss = val_loss
                loss_desc = f"Train: {train_loss:.6f}, Val: {val_loss:.6f}"
            else:
                eval_loss = train_loss
                loss_desc = f"Train: {train_loss:.6f}"

            t_bar.set_description(f"Epoch {epoch} | {loss_desc}")

            if epoch % save_freq == 0 or epoch == num_epochs:
                pass
                # self._store_solution_snapshot(train_loader)

            if eval_loss < best_loss:
                best_loss = eval_loss
                self.best_training_loss = train_loss
                self.best_model_index = epoch - 1
                self._save_best_model_state()
                patience_counter = 0
                tqdm.write(f"Epoch {epoch}: New best model saved! Loss: {best_loss:.10f}")
            else:
                patience_counter += 1

            if 0 < patience <= patience_counter:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Training complete! Best loss: {best_loss:.8f} at epoch {self.best_model_index + 1}")
        return {
            'total_loss': self.epoch_losses["total"],
            'data_loss': self.epoch_losses["data"],
            'pde_loss': self.epoch_losses["pde"],
            'best_model_index': self.best_model_index
        }

    @abc.abstractmethod
    def _save_best_model_state(self):
        """Save the current model state as the best model."""
        pass

    @abc.abstractmethod
    def load_best_model(self) -> bool:
        """
        Load the best model weights.

        Returns:
            Boolean indicating if the best model was successfully loaded
        """
        pass

    def plot_losses(self, save_path):
        """Plot training and validation losses over epochs."""
        import matplotlib.pyplot as plt
        epochs = range(1, len(self.epoch_losses["total"]) + 1)
        plt.figure(figsize=(8, 3.6))
        plt.subplot(1, 2, 1)
        plt.semilogy(epochs, self.epoch_losses["total"], 'b-', label='Total Loss')
        plt.semilogy(epochs, self.epoch_losses["data"], 'r-', label='Data Loss')
        plt.semilogy(epochs, self.epoch_losses["pde"], 'g-', label='Physics Loss')
        if self.best_model_index is not None:
            plt.axvline(x=self.best_model_index + 1, color='k', linestyle='--',
                        label=f'Best Model (Epoch {self.best_model_index + 1})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Losses')
        plt.legend(frameon = False)
        #plt.grid(True, which="both", ls="--", alpha=0.5)

        if len(self.val_losses["total"]) > 0:
            plt.subplot(1, 2, 2)
            plt.semilogy(epochs[:len(self.val_losses["total"])], self.val_losses["total"], 'b-', label='Total Loss')
            plt.semilogy(epochs[:len(self.val_losses["data"])], self.val_losses["data"], 'r-', label='Data Loss')
            plt.semilogy(epochs[:len(self.val_losses["pde"])], self.val_losses["pde"], 'g-', label='Physics Loss')
            if self.best_model_index is not None:
                plt.axvline(x=self.best_model_index + 1, color='k', linestyle='--',
                            label=f'Best Model (Epoch {self.best_model_index + 1})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (log scale)')
            plt.title('Validation Losses')
            plt.legend(frameon = False)
            #plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()

        plt.savefig( os.path.join(save_path, "losses.png"), transparent = True, dpi = 300)
        plt.show()


    def plot_solution(self, target=None):
        """
        Plot the evolution of the solution over training epochs using Plotly.

        Args:
            target: Optional target solution to include in the plot

        Returns:
            Plotly figure object
        """
        import plotly.graph_objects as go
        try:
            from plotly_resampler import FigureResampler
        except ImportError:
            FigureResampler = None

        fig = go.Figure()
        num_epochs = len(self.epoch_data)
        step = max(1, num_epochs // 20)

        # Plot network predictions per epoch
        for i in range(0, num_epochs, step):
            epoch_num = i + 1
            t_viz = self.epoch_data[i]['inputs'][:, 0]
            solution = self.epoch_data[i]['predictions'].flatten()
            sort_idx = np.argsort(t_viz)
            t_viz_sorted = t_viz[sort_idx]
            solution_sorted = solution[sort_idx]
            fig.add_trace(go.Scattergl(
                x=t_viz_sorted,
                y=solution_sorted,
                mode='lines',
                name=f'Epoch {epoch_num}',
                line=dict(width=2),
                visible=True
            ))

        # Plot target solution if provided
        if target is not None:
            t_sorted = np.sort(self.epoch_data[-1]['inputs'][:, 0])
            fig.add_trace(go.Scattergl(
                x=t_sorted,
                y=target,
                mode='lines',
                name='Target',
                line=dict(color='black', width=3, dash='dot')
            ))

        # Plot training data
        if hasattr(self, 'target_data'):
            t_train = self.target_data['inputs'][:, 0]
            targets = self.target_data['targets'].flatten()
            sort_idx = np.argsort(t_train)
            t_train_sorted = t_train[sort_idx]
            targets_sorted = targets[sort_idx]
            fig.add_trace(go.Scattergl(
                x=t_train_sorted,
                y=targets_sorted,
                mode='markers',
                name='Training Data',
                marker=dict(size=6, symbol='circle')
            ))

        # Update layout
        fig.update_layout(
            title='PINN Solution Evolution',
            xaxis_title='Time (t)',
            yaxis_title='Cell Potential (u)',
            legend=dict(yanchor="top", y=1, xanchor="left", x=1.05, orientation="v", title="Epochs"),
            hovermode='closest',
            template="plotly_white"
        )

        # Add slider if many epochs are available
        if num_epochs > 10:
            steps = []
            for i in range(0, num_epochs, step):
                step_data = {
                    'method': 'update',
                    'args': [{'visible': [False] * len(fig.data)}, {'title': f'PINN Solution - Epoch {i + 1}'}],
                    'label': f'{i + 1}'
                }
                step_visibility = [False] * len(fig.data)
                step_visibility[i // step] = True
                if target is not None:
                    step_visibility[-2] = True
                step_visibility[-1] = True
                step_data['args'][0]['visible'] = step_visibility
                steps.append(step_data)
            slider = [dict(
                active=len(steps) - 1,
                currentvalue={"prefix": "Epoch: "},
                pad={"t": 50},
                steps=steps
            )]
            fig.update_layout(sliders=slider)

        # Use FigureResampler if available
        if FigureResampler is not None:
            fig = FigureResampler(fig, default_n_shown_samples=2000)

        return fig

    def evaluate(
            self,
            scaler: ScalerLoader,
            df: pd.DataFrame,
            test_loader: DataLoader,
            feature_names: list,
            target_names: list,
            save_folder: str = "plots") -> pd.DataFrame:

        """
        Evaluate the PINN on test data—now works with DataLoaders that yield either
        (X_batch, Y_batch) or just X_batch. When Y_batch is missing, true-target
        columns will be NaN but predictions, PDE residuals, physics-layer outputs,
        and performance metrics will still be saved.
        """

        os.makedirs(save_folder, exist_ok=True)
        self.load_best_model()
        self.eval()

        # 1) prepare storage for every batch in normalized (0–1) space
        norm = {}
        for f in feature_names:
            norm[f] = []
        for tgt in target_names:
            norm[f"{tgt}_targ"] = []
            norm[f"{tgt}_pred"] = []
        for extra in ["f_pde", "eta_kinetic_deg", "R_ohmic_deg", "G_pred", "u_deg_t"]:
            norm[extra] = []

        # 2) loop batches, unpacking Xb ± Yb
        for batch in tqdm(test_loader, desc="Evaluating PINN"):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                Xb, Yb = batch
                has_labels = True
                Yb = Yb.to(self.device)
            else:
                Xb = batch if not isinstance(batch, (list, tuple)) else batch[0]
                Yb = None
                has_labels = False

            Xb = Xb.to(self.device)
            Xb.requires_grad_(True)

            out = self.forward(Xb, return_aux=True)

            # — record features (still 0–1)
            for i, feat in enumerate(feature_names):
                try:
                    norm[feat].append(Xb[:, i:i + 1].cpu().detach().numpy().flatten())
                except IndexError:
                    pass

            # — record targets & preds
            pred_arr = out["u_deg_norm"].cpu().detach().numpy().flatten()
            for i, tgt in enumerate(target_names):
                norm[f"{tgt}_pred"].append(pred_arr)
                if has_labels:
                    true_arr = Yb[:, 0:1].cpu().detach().numpy().flatten()
                    norm[f"{tgt}_targ"].append(true_arr)
                else:
                    norm[f"{tgt}_targ"].append(np.full_like(pred_arr, np.nan, dtype=float))

            # — record PDE & physics outputs (or zero‐fills)
            try:
                norm["f_pde"].append(out["f_pde"].cpu().detach().numpy().flatten())
            except KeyError:
                norm["f_pde"].append(np.zeros_like(pred_arr))

            try:
                norm["eta_kinetic_deg"].append(out["eta_kinetic_deg_norm"].cpu().detach().numpy().flatten())
                norm["R_ohmic_deg"].append(out["R_ohmic_deg_norm"].cpu().detach().numpy().flatten())
            except KeyError:
                zero = np.zeros_like(pred_arr)
                norm["eta_kinetic_deg"].append(zero)
                norm["R_ohmic_deg"].append(zero)

            try:
                norm["G_pred"].append(out["G_pred_norm"].cpu().detach().numpy().flatten())
            except KeyError:
                norm["G_pred"].append(np.zeros_like(pred_arr))

            try:
                norm["u_deg_t"].append(out["u_deg_t_norm"].cpu().detach().numpy().flatten())
            except KeyError:
                norm["u_deg_t"].append(np.zeros_like(pred_arr))

        # 3) flatten and sort by t
        flat = {k: np.concatenate(v) for k, v in norm.items()}
        order = np.argsort(flat["t"])
        for k in flat:
            flat[k] = flat[k][order]

        # helper to invert one key via output_scaler
        def inv_out(key):
            arr = flat[key].reshape(-1, 1)
            return scaler.output_scaler.inverse_transform(arr).flatten()

        # 4) undo MinMax scaling for features
        X_norm = np.stack([flat[f] for f in feature_names], axis=1)
        X_orig = scaler.input_scaler.inverse_transform(X_norm)
        t = X_orig[:, 0]
        j = X_orig[:, 1]
        T = X_orig[:, 2] if "T" in feature_names else np.zeros_like(t)

        # 5) decide if we really have true targets (not all NaN)
        first_targ_key = f"{target_names[0]}_targ"
        has_any_truth = not np.all(np.isnan(flat[first_targ_key]))

        if has_any_truth:
            try:
                u_deg_targ = inv_out("U_deg_targ")
                u_deg_pred = inv_out("U_deg_pred")
            except:
                u_deg_targ = inv_out("U_targ") - df["U_perf"].values
                u_deg_pred = inv_out("U_pred") - df["U_perf"].values
        else:
            pred_key = f"{target_names[0]}_pred"
            u_deg_pred = inv_out(pred_key)
            u_deg_targ = np.full_like(u_deg_pred, np.nan)

        # physics outputs (still normalized units for f_pde)
        f_pde = flat["f_pde"]
        eta_kinetic_deg = inv_out("eta_kinetic_deg")
        R_ohmic_deg = inv_out("R_ohmic_deg")
        G_pred = inv_out("G_pred")
        u_deg_t = inv_out("u_deg_t")

        # 6) assemble the DataFrame
        N = len(u_deg_pred)

        output_data = {
            "t": t,
            "j": j,
            "T": T,
            "u_deg_targ": u_deg_targ,
            "u_deg_pred": u_deg_pred,
            "f_pde": f_pde,
            "eta_kinetic_deg": eta_kinetic_deg,
            "R_ohmic_deg": R_ohmic_deg,
            "G_pred": G_pred,
            "u_deg_t": u_deg_t
        }

        # Dynamically add features and performance columns
        for feat in feature_names:
            output_data[feat] = X_orig[:, feature_names.index(feat)]

        # Detect and add performance columns
        if "eta_kinetic_perf" in df.columns:
            output_data["eta_kinetic_perf"] = df["eta_kinetic_perf"].values
        else:
            output_data["eta_kinetic_perf"] = np.full(N, np.nan)

        if "R_ohmic_perf" in df.columns:
            output_data["R_ohmic_perf"] = df["R_ohmic_perf"].values
        else:
            output_data["R_ohmic_perf"] = np.full(N, np.nan)

        if "U_perf" in df.columns:
            output_data["u_perf"] = df["U_perf"].values
        else:
            output_data["u_perf"] = np.full(N, np.nan)

        # Handle dynamic voltage columns
        if "U" in df.columns:
            output_data["u_cell_targ"] = df["U"].values
        else:
            output_data["u_cell_targ"] = np.full(N, np.nan)

        if "U_perf" in df.columns:
            output_data["u_cell_pred"] = df["U_perf"].values + u_deg_pred
        else:
            output_data["u_cell_pred"] = u_deg_pred.copy()

        output_df = pd.DataFrame(output_data)
        save_path = os.path.join(save_folder, "pinn_evaluation.csv")
        output_df.to_csv(save_path, index=False)
        print(f"Saved evaluation DataFrame to: {save_path}")

        return output_df

    @staticmethod
    def save_model(model, filepath=None, metadata=None):
        """
        Save model weights and metadata to disk.

        Args:
            model: A BasePINN instance
            filepath: Path to save the model. If None, a timestamped path is generated.
            metadata: Dict of additional information to store with the model
                     (e.g., training params, performance metrics)

        Returns:
            Path to the saved model
        """
        # Create directory for model if it doesn't exist
        if filepath is None:
            # Generate timestamped directory name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = model.__class__.__name__
            filepath = f"saved_models/{model_name}_{timestamp}"

        os.makedirs(filepath, exist_ok=True)

        # Build complete metadata
        model_metadata = {
            "model_type": model.__class__.__name__,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "hyperparameters": {
                "f_hidden_dim": model.f_hidden_dim,
                "g_hidden_dim": model.g_hidden_dim,
                "f_layers": model.f_layers,
                "g_layers": model.g_layers,
                "lr": model.lr,
                "pde_weight": model.alpha
            },
            "training_history": {
                "train_losses": model.epoch_losses,
                "val_losses": model.val_losses,
                "best_model_index": model.best_model_index,
                "best_training_loss": model.best_training_loss,
                "best_val_loss": model.best_val_loss if hasattr(model, "best_val_loss") else None
            }
        }

        # Add custom metadata if provided
        if metadata:
            model_metadata.update(metadata)

        # Save metadata to JSON
        with open(os.path.join(filepath, "metadata.json"), "w") as f:
            json.dump(model_metadata, f, indent=2, default=lambda x: x.tolist() if isinstance(x, torch.Tensor) else x)

        # Save F network weights
        if hasattr(model, "F_nn") and model.best_F_state is not None:
            torch.save(model.best_F_state, os.path.join(filepath, "F_nn.pth"))

        # Save G network weights if applicable
        if hasattr(model, "G_nn") and hasattr(model, "best_G_state") and model.best_G_state is not None:
            torch.save(model.best_G_state, os.path.join(filepath, "G_nn.pth"))

        print(f"Model successfully saved to {filepath}")
        return filepath



# =======================================================
# Implementation 1: Full PINN with Physics and G network
# =======================================================
class FullPINN(BasePINN):
    """
    PINN implementation incorporating both:
    1. Physics layer (u = eta_kinetic + j * R_ohmic) with F predicting eta_kinetic and R_ohmic
    2. Neural network G for degradation dynamics
    """

    def _initialize_networks(self):
        """Initialize F and G networks."""
        # F network: predicts eta_kinetic and R_ohmic
        self.F_nn = MLP_Softplus(
            input_dim=2,  # input: [t, j]
            output_dim=2,  # outputs: eta_kinetic, R_ohmic
            hidden_dim=self.f_hidden_dim,
            num_layers=self.f_layers,
            dropout=0.0
        )

        # G network: approximates degradation dynamics
        self.G_nn = MLP_Softplus(
            input_dim=5,  # input: [t, j, u, u_t, u_j]
            output_dim=1,  # outputs: approximation of u_deg_t
            hidden_dim=self.g_hidden_dim,
            num_layers=self.g_layers,
            dropout=0.0
        )

    def _initialize_optimizers(self):
        """Initialize optimizers for F and G networks."""
        self.optimizer_F = torch.optim.Adam(self.F_nn.parameters(), lr=self.lr)
        self.optimizer_G = torch.optim.Adam(self.G_nn.parameters(), lr=self.lr)

    def _zero_grad_optimizers(self):
        """Zero gradients for all optimizers."""
        self.optimizer_F.zero_grad()
        self.optimizer_G.zero_grad()

    def _step_optimizers(self):
        """Step all optimizers."""
        self.optimizer_F.step()
        self.optimizer_G.step()

    def _save_best_model_state(self):
        """Save the current model state as the best model."""
        self.best_F_state = {k: v.cpu().clone() for k, v in self.F_nn.state_dict().items()}
        self.best_G_state = {k: v.cpu().clone() for k, v in self.G_nn.state_dict().items()}

    def load_best_model(self) -> bool:
        """
        Load the best model weights.

        Returns:
            Boolean indicating if the best model was successfully loaded
        """
        if self.best_F_state is not None and self.best_G_state is not None:
            self.F_nn.load_state_dict({k: v.to(self.device) for k, v in self.best_F_state.items()})
            self.G_nn.load_state_dict({k: v.to(self.device) for k, v in self.best_G_state.items()})
            print("Best model loaded successfully.")
            return True
        else:
            print("No best model available to load.")
            return False

    def forward(self, X: torch.Tensor, return_aux: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass incorporating both physics and G network.

        Args:
            X: Input tensor of shape (N, 2) with columns [t, j]
            return_aux: Whether to return auxiliary outputs

        Returns:
            Dictionary containing model outputs
        """
        # Enable gradients on input for differentiation
        X.requires_grad = True

        # Split input
        t = X[:, 0:1]
        j = X[:, 1:2]

        # F network forward pass
        out = self.F_nn(torch.cat([t, j], dim=1))
        eta_kinetic = out[:, 0:1]
        R_ohmic = out[:, 1:2]

        # Physics layer: u = eta_kinetic + j * R_ohmic
        u = eta_kinetic + j * R_ohmic

        # Derivative calculations
        u_t = grad(outputs=u.sum(), inputs=t, create_graph=True, only_inputs=True)[0]
        u_j = grad(outputs=u.sum(), inputs=j, create_graph=True, only_inputs=True)[0]

        # G network for degradation dynamics
        phys_input = torch.cat([t, j, u, u_t, u_j], dim=1)
        G_pred = self.G_nn(phys_input)

        # PDE residual
        f_pde = u_t - G_pred

        # Prepare output
        result = {
            "u_deg_norm": u,
            "f_pde": f_pde,
        }

        if return_aux:
            result.update({
                "eta_kinetic_deg_norm": eta_kinetic,
                "R_ohmic_deg_norm": R_ohmic,
                "G_pred_norm": G_pred,
                "u_deg_t_norm": u_t
            })

        return result




# =======================================================
# Implementation 2: Physics-only PINN (no G network)
# =======================================================
class PhysicsPINN(BasePINN):
    """
    PINN implementation incorporating only:
    Physics layer (u = eta_kinetic + j * R_ohmic) with F predicting eta_kinetic and R_ohmic
    """

    def _initialize_networks(self):
        """Initialize F network only."""
        # F network: predicts eta_kinetic and R_ohmic
        self.F_nn = MLP_Softplus(
            input_dim=2,  # input: [t, j]
            output_dim=2,  # outputs: eta_kinetic, R_ohmic
            hidden_dim=self.f_hidden_dim,
            num_layers=self.f_layers,
            dropout=0.0
        )

    def _initialize_optimizers(self):
        """Initialize optimizer for F network."""
        self.optimizer_F = torch.optim.Adam(self.F_nn.parameters(), lr=self.lr)

    def _zero_grad_optimizers(self):
        """Zero gradients for F optimizer."""
        self.optimizer_F.zero_grad()

    def _step_optimizers(self):
        """Step F optimizer."""
        self.optimizer_F.step()

    def _save_best_model_state(self):
        """Save the current model state as the best model."""
        self.best_F_state = {k: v.cpu().clone() for k, v in self.F_nn.state_dict().items()}

    def load_best_model(self) -> bool:
        """
        Load the best model weights.

        Returns:
            Boolean indicating if the best model was successfully loaded
        """
        if self.best_F_state is not None:
            self.F_nn.load_state_dict({k: v.to(self.device) for k, v in self.best_F_state.items()})
            print("Best model loaded successfully.")
            return True
        else:
            print("No best model available to load.")
            return False

    def forward(self, X: torch.Tensor, return_aux: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass incorporating only physics layer.

        Args:
            X: Input tensor of shape (N, 2) with columns [t, j]
            return_aux: Whether to return auxiliary outputs

        Returns:
            Dictionary containing model outputs
        """
        # Enable gradients on input for differentiation
        X.requires_grad = True

        # Split input
        t = X[:, 0:1]
        j = X[:, 1:2]

        # F network forward pass
        out = self.F_nn(torch.cat([t, j], dim=1))
        eta_kinetic = out[:, 0:1]
        R_ohmic = out[:, 1:2]

        # Physics layer: u = eta_kinetic + j * R_ohmic
        u = eta_kinetic + j * R_ohmic

        # Prepare output (no PDE residual since there's no G network)
        result = {
            "u_deg_norm": u
        }

        if return_aux:
            # For physics-only model, we still compute derivatives but don't use them in loss
            u_t = grad(outputs=u.sum(), inputs=t, create_graph=True, only_inputs=True)[0]
            u_j = grad(outputs=u.sum(), inputs=j, create_graph=True, only_inputs=True)[0]

            result.update({
                "eta_kinetic_deg_norm": eta_kinetic,
                "R_ohmic_deg_norm": R_ohmic,
                "u_deg_t_norm": u_t,
                "u_deg_j_norm": u_j
            })

        return result



# =======================================================
# Implementation 3: G-only PINN (no physics layer)
# =======================================================
class GPINN(BasePINN):
    """
    PINN implementation incorporating only:
    2. Neural network G for degradation dynamics
    """

    def _initialize_networks(self):
        """ Initialize F and G networks. """

        # F network: predicts eta_kinetic and R_ohmic
        self.F_nn = MLP_Softplus(
            input_dim=2,  # input: [t, j]
            output_dim=1,  # outputs: u
            hidden_dim=self.f_hidden_dim,
            num_layers=self.f_layers,
            dropout=0.0
        )

        # G network: approximates degradation dynamics
        self.G_nn = MLP_Softplus(
            input_dim=5,  # input: [t, j, u, u_t, u_j]
            output_dim=1,  # outputs: approximation of u_deg_t
            hidden_dim=self.g_hidden_dim,
            num_layers=self.g_layers,
            dropout=0.0
        )

    def _initialize_optimizers(self):
        """ Initialize optimizers for F and G networks. """
        self.optimizer_F = torch.optim.Adam(self.F_nn.parameters(), lr=self.lr)
        self.optimizer_G = torch.optim.Adam(self.G_nn.parameters(), lr=self.lr)

    def _zero_grad_optimizers(self):
        """Zero gradients for all optimizers."""
        self.optimizer_F.zero_grad()
        self.optimizer_G.zero_grad()

    def _step_optimizers(self):
        """Step all optimizers."""
        self.optimizer_F.step()
        self.optimizer_G.step()

    def _save_best_model_state(self):
        """Save the current model state as the best model."""
        self.best_F_state = {k: v.cpu().clone() for k, v in self.F_nn.state_dict().items()}
        self.best_G_state = {k: v.cpu().clone() for k, v in self.G_nn.state_dict().items()}

    def load_best_model(self) -> bool:
        """
        Load the best model weights.
        Returns: Boolean indicating if the best model was successfully loaded
        """

        if self.best_F_state is not None and self.best_G_state is not None:
            self.F_nn.load_state_dict({k: v.to(self.device) for k, v in self.best_F_state.items()})
            self.G_nn.load_state_dict({k: v.to(self.device) for k, v in self.best_G_state.items()})
            print("Best model loaded successfully.")
            return True
        else:
            print("No best model available to load.")
            return False

    def forward(self, X: torch.Tensor, return_aux: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass incorporating both physics and G network.
        Args: X: Input tensor of shape (N, 2) with columns [t, j] return_aux: Whether to return auxiliary outputs

        Returns: Dictionary containing model outputs
        """
        # Enable gradients on input for differentiation
        X.requires_grad = True

        # Split input
        t = X[:, 0:1]
        j = X[:, 1:2]

        # F network forward pass
        u = self.F_nn(torch.cat([t, j], dim=1))

        # Derivative calculations
        u_t = grad(outputs=u.sum(), inputs=t, create_graph=True, only_inputs=True)[0]
        u_j = grad(outputs=u.sum(), inputs=j, create_graph=True, only_inputs=True)[0]

        # G network for degradation dynamics
        phys_input = torch.cat([t, j, u, u_t, u_j], dim=1)
        G_pred = self.G_nn(phys_input)

        # PDE residual
        f_pde = u_t - G_pred

        # Prepare output
        result = {
            "u_deg_norm": u,
            "f_pde": f_pde,
            }

        if return_aux:
            result.update({
                "eta_kinetic_deg_norm": 0,
                "R_ohmic_deg_norm": 0,
                "G_pred_norm": G_pred,
                "u_deg_t_norm": u_t
            })

        return result



