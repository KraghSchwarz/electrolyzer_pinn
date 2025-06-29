from elec_pinn.models import BasePINN
#from elec_pinn.models import MLP_Softplus
from torch.autograd import grad
from torch.utils.data import DataLoader
from tqdm import trange
from typing import Dict, Tuple, Optional, Any, Union
#from MLP import MLP_Softplus
from elec_pinn.data.loader import ScalerLoader
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from elec_pinn.models.MLP import MLP_Softplus

class ANN(BasePINN):

    """ ANN implementation incorporating neither G nor physics_layer """

    def _initialize_networks(self):
        """ Initialize F and G networks. """

        # F network: predicts eta_kinetic and R_ohmic
        self.F_nn = MLP_Softplus(
            input_dim=self.input_dim,  # input: [t, j]
            output_dim=1,  # outputs: u
            hidden_dim=self.f_hidden_dim,
            num_layers=self.f_layers,
            dropout=0.0
        )


    def _initialize_optimizers(self):
        """ Initialize optimizers for F and G networks. """
        self.optimizer_F = torch.optim.Adam(self.F_nn.parameters(), lr=self.lr)


    def _zero_grad_optimizers(self):
        """Zero gradients for all optimizers."""
        self.optimizer_F.zero_grad()

    def _step_optimizers(self):
        """Step all optimizers."""
        self.optimizer_F.step()

    def _save_best_model_state(self):
        """Save the current model state as the best model."""
        self.best_F_state = {k: v.cpu().clone() for k, v in self.F_nn.state_dict().items()}

    def load_best_model(self) -> bool:
        """
        Load the best model weights.
        Returns: Boolean indicating if the best model was successfully loaded
        """

        if self.best_F_state is not None and self.best_G_state is not None:
            self.F_nn.load_state_dict({k: v.to(self.device) for k, v in self.best_F_state.items()})
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
        #u_j = grad(outputs=u.sum(), inputs=j, create_graph=True, only_inputs=True)[0]



        # Prepare output
        result = {
            "u_deg_norm": u
            }

        if return_aux:
            result.update({
                "eta_kinetic_deg_norm": 0,
                "R_ohmic_deg_norm": 0,
                "G_pred_norm": 0,
                "u_deg_t_norm": u_t
            })

        return result