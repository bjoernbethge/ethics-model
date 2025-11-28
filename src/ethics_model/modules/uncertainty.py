"""
Uncertainty Quantification Module for Ethics Model

This module provides tools for quantifying uncertainty in ethical judgments,
using techniques like Monte Carlo Dropout, ensemble methods, and
calibrated uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import plotly.express as px
import plotly.graph_objects as go


class UncertaintyActivation(nn.Module):
    """
    Custom activation layer that outputs both mean and variance.
    
    This layer produces a mean value through sigmoid activation and
    a variance value through softplus activation for uncertainty estimation.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance from input.
        
        Args:
            x: Input tensor of shape (..., 2)
            
        Returns:
            Tuple of (mean, variance) tensors
        """
        mean = torch.sigmoid(x[..., 0:1])
        variance = F.softplus(x[..., 1:2])
        
        return mean, variance


class MCSamplingLayer(nn.Module):
    """
    Monte Carlo sampling layer with dropout for uncertainty estimation.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        dropout_rate: Dropout probability
        num_mc_samples: Number of Monte Carlo samples during inference
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        dropout_rate: float = 0.1,
        num_mc_samples: int = 30
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.num_mc_samples = num_mc_samples
    
    def forward(self, x: torch.Tensor, training: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x: Input tensor
            training: Whether in training mode
            
        Returns:
            During training: output tensor
            During inference: tuple of (mean, variance) tensors
        """
        if training:
            # During training, just apply dropout and linear layer
            x = self.dropout(x)
            return torch.sigmoid(self.linear(x))
        else:
            # During inference, perform Monte Carlo sampling
            self.train()  # Enable dropout
            
            samples = []
            for _ in range(self.num_mc_samples):
                x_drop = self.dropout(x)
                output = torch.sigmoid(self.linear(x_drop))
                samples.append(output)
            
            self.eval()  # Restore eval mode
            
            # Stack samples
            stacked_samples = torch.stack(samples, dim=0)
            
            # Compute mean and variance
            mean = stacked_samples.mean(dim=0)
            variance = stacked_samples.var(dim=0)
            
            return mean, variance


class EvidentialLayer(nn.Module):
    """
    Evidential uncertainty layer based on Dirichlet distribution.
    
    Implements evidential deep learning approach for uncertainty quantification,
    where model outputs are interpreted as evidence for different outcomes.
    
    Args:
        input_dim: Input dimension
        activation: Activation function for evidence
    """
    
    def __init__(self, input_dim: int, activation: str = "softplus"):
        super().__init__()
        
        self.evidence_layer = nn.Linear(input_dim, 2)  # For 2 parameters: alpha, beta
        
        if activation == "softplus":
            self.activation = F.softplus
        elif activation == "relu":
            self.activation = F.relu
        else:
            self.activation = F.softplus
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute parameters of the Beta distribution for uncertainty.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with evidence, alpha, beta, mean, variance
        """
        # Compute evidence
        evidence = self.activation(self.evidence_layer(x))
        
        # Calculate alpha and beta
        alpha = evidence[..., 0:1] + 1.0
        beta = evidence[..., 1:2] + 1.0
        
        # Calculate mean and variance of the Beta distribution
        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        
        return {
            'evidence': evidence,
            'alpha': alpha,
            'beta': beta,
            'mean': mean,
            'variance': variance
        }
    
    def edl_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        lambda_reg: float = 0.1
    ) -> torch.Tensor:
        """
        Evidential Deep Learning loss function.
        
        Args:
            outputs: Dictionary of model outputs
            targets: Target values
            lambda_reg: Regularization weight
            
        Returns:
            Loss tensor
        """
        alpha = outputs['alpha']
        beta = outputs['beta']
        
        # Convert to one-hot if needed
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(-1)
        
        # Binary cross-entropy
        S = alpha + beta
        p = alpha / S
        
        # Negative log likelihood
        nll = -torch.log(p + 1e-10) * targets - torch.log(1 - p + 1e-10) * (1 - targets)
        
        # Regularization term
        reg = lambda_reg * (targets * (beta - 1) + (1 - targets) * (alpha - 1))
        
        return (nll + reg).mean()


class UncertaintyEthicsModel(nn.Module):
    """
    Wrapper for ethics model with uncertainty quantification.
    
    Args:
        base_model: Base ethics model
        mc_dropout_rate: Dropout rate for Monte Carlo dropout
        num_mc_samples: Number of Monte Carlo samples
        uncertainty_method: Method for uncertainty quantification
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        mc_dropout_rate: float = 0.1,
        num_mc_samples: int = 30,
        uncertainty_method: str = "mc_dropout"  # or "evidential"
    ):
        super().__init__()
        
        self.base_model = base_model
        self.uncertainty_method = uncertainty_method
        
        # Replace output projections based on uncertainty method
        if uncertainty_method == "mc_dropout":
            # Replace with MC dropout layers
            d_model = base_model.meta_cognitive.state_dict()["1.bias"].shape[0]
            
            self.ethics_projection = MCSamplingLayer(
                d_model, 1, mc_dropout_rate, num_mc_samples
            )
            
            self.manipulation_projection = MCSamplingLayer(
                d_model, 1, mc_dropout_rate, num_mc_samples
            )
        elif uncertainty_method == "evidential":
            # Replace with evidential layers
            d_model = base_model.meta_cognitive.state_dict()["1.bias"].shape[0]
            
            self.ethics_projection = EvidentialLayer(d_model)
            self.manipulation_projection = EvidentialLayer(d_model)
        else:
            raise ValueError(f"Unknown uncertainty method: {uncertainty_method}")
    
    def forward(
        self,
        training: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            training: Whether in training mode
            **kwargs: Keyword arguments for base model
            
        Returns:
            Dictionary of model outputs with uncertainty estimates
        """
        # Get base model outputs
        base_outputs = self.base_model.forward(**kwargs)
        
        # Replace ethics and manipulation scores with uncertainty estimates
        meta_features = base_outputs['meta_cognitive_features']
        
        if self.uncertainty_method == "mc_dropout":
            if training:
                # During training
                ethics_score = self.ethics_projection(meta_features, training=True)
                manipulation_score = self.manipulation_projection(meta_features, training=True)
                
                base_outputs['ethics_score'] = ethics_score
                base_outputs['manipulation_score'] = manipulation_score
            else:
                # During inference
                ethics_mean, ethics_var = self.ethics_projection(meta_features, training=False)
                manipulation_mean, manipulation_var = self.manipulation_projection(meta_features, training=False)
                
                base_outputs['ethics_score'] = ethics_mean
                base_outputs['ethics_uncertainty'] = ethics_var
                base_outputs['manipulation_score'] = manipulation_mean
                base_outputs['manipulation_uncertainty'] = manipulation_var
        
        elif self.uncertainty_method == "evidential":
            # Evidential outputs
            ethics_outputs = self.ethics_projection(meta_features)
            manipulation_outputs = self.manipulation_projection(meta_features)
            
            base_outputs['ethics_score'] = ethics_outputs['mean']
            base_outputs['ethics_uncertainty'] = ethics_outputs['variance']
            base_outputs['ethics_evidence'] = ethics_outputs['evidence']
            base_outputs['ethics_alpha'] = ethics_outputs['alpha']
            base_outputs['ethics_beta'] = ethics_outputs['beta']
            
            base_outputs['manipulation_score'] = manipulation_outputs['mean']
            base_outputs['manipulation_uncertainty'] = manipulation_outputs['variance']
            base_outputs['manipulation_evidence'] = manipulation_outputs['evidence']
            base_outputs['manipulation_alpha'] = manipulation_outputs['alpha']
            base_outputs['manipulation_beta'] = manipulation_outputs['beta']
        
        return base_outputs
    
    def loss_function(
        self,
        outputs: Dict[str, torch.Tensor],
        ethics_labels: torch.Tensor,
        manipulation_labels: torch.Tensor,
        lambda_reg: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss with uncertainty regularization.
        
        Args:
            outputs: Model outputs
            ethics_labels: Ethics labels
            manipulation_labels: Manipulation labels
            lambda_reg: Regularization weight for evidential loss
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        if self.uncertainty_method == "mc_dropout":
            # Standard BCE loss
            loss_ethics = F.binary_cross_entropy(
                outputs['ethics_score'],
                ethics_labels
            )
            
            loss_manip = F.binary_cross_entropy(
                outputs['manipulation_score'],
                manipulation_labels
            )
            
            total_loss = loss_ethics + 0.5 * loss_manip
            
            loss_dict = {
                'ethics_loss': loss_ethics,
                'manipulation_loss': loss_manip,
                'total_loss': total_loss
            }
        
        elif self.uncertainty_method == "evidential":
            # Evidential loss
            ethics_outputs = {
                'alpha': outputs['ethics_alpha'],
                'beta': outputs['ethics_beta']
            }
            
            manipulation_outputs = {
                'alpha': outputs['manipulation_alpha'],
                'beta': outputs['manipulation_beta']
            }
            
            # Use evidential loss
            edl_loss_ethics = self.ethics_projection.edl_loss(
                ethics_outputs,
                ethics_labels,
                lambda_reg
            )
            
            edl_loss_manip = self.manipulation_projection.edl_loss(
                manipulation_outputs,
                manipulation_labels,
                lambda_reg
            )
            
            total_loss = edl_loss_ethics + 0.5 * edl_loss_manip
            
            loss_dict = {
                'ethics_loss': edl_loss_ethics,
                'manipulation_loss': edl_loss_manip,
                'total_loss': total_loss
            }
        
        return total_loss, loss_dict


class UncertaintyVisualizer:
    """
    Visualizes uncertainty in model predictions.
    
    This class provides tools for creating visualizations of
    uncertainty in ethical judgments.
    """
    
    @staticmethod
    def plot_uncertainty_distribution(
        ethics_mean: Union[float, np.ndarray],
        ethics_var: Union[float, np.ndarray],
        manipulation_mean: Union[float, np.ndarray],
        manipulation_var: Union[float, np.ndarray],
        title: str = "Uncertainty in Ethical Analysis"
    ) -> go.Figure:
        """
        Create a visualization of uncertainty distributions.
        
        Args:
            ethics_mean: Mean ethics score(s)
            ethics_var: Variance of ethics score(s)
            manipulation_mean: Mean manipulation score(s)
            manipulation_var: Variance of manipulation score(s)
            title: Plot title
            
        Returns:
            Plotly figure with uncertainty distribution
        """
        # Convert to numpy if needed
        if isinstance(ethics_mean, torch.Tensor):
            ethics_mean = ethics_mean.cpu().detach().numpy()
        if isinstance(ethics_var, torch.Tensor):
            ethics_var = ethics_var.cpu().detach().numpy()
        if isinstance(manipulation_mean, torch.Tensor):
            manipulation_mean = manipulation_mean.cpu().detach().numpy()
        if isinstance(manipulation_var, torch.Tensor):
            manipulation_var = manipulation_var.cpu().detach().numpy()
        
        # Ensure 1D arrays
        ethics_mean = np.atleast_1d(ethics_mean).flatten()
        ethics_var = np.atleast_1d(ethics_var).flatten()
        manipulation_mean = np.atleast_1d(manipulation_mean).flatten()
        manipulation_var = np.atleast_1d(manipulation_var).flatten()
        
        # Create x values for plotting
        x = np.linspace(0, 1, 100)
        
        # Create figure
        fig = go.Figure()
        
        # Plot each distribution
        for i in range(len(ethics_mean)):
            # Ethics distribution
            e_mean = ethics_mean[i]
            e_std = np.sqrt(ethics_var[i])
            
            e_y = norm_pdf(x, e_mean, e_std)
            
            # Add distribution for ethics
            fig.add_trace(go.Scatter(
                x=x,
                y=e_y,
                mode='lines',
                name=f'Ethics {i+1}',
                line=dict(color='blue', width=2, dash='solid')
            ))
            
            # Add vertical line for mean
            fig.add_trace(go.Scatter(
                x=[e_mean, e_mean],
                y=[0, np.max(e_y)],
                mode='lines',
                name=f'Ethics Mean {i+1}',
                line=dict(color='blue', width=2, dash='dash'),
                showlegend=False
            ))
            
            # Manipulation distribution
            m_mean = manipulation_mean[i]
            m_std = np.sqrt(manipulation_var[i])
            
            m_y = norm_pdf(x, m_mean, m_std)
            
            # Add distribution for manipulation
            fig.add_trace(go.Scatter(
                x=x,
                y=m_y,
                mode='lines',
                name=f'Manipulation {i+1}',
                line=dict(color='red', width=2, dash='solid')
            ))
            
            # Add vertical line for mean
            fig.add_trace(go.Scatter(
                x=[m_mean, m_mean],
                y=[0, np.max(m_y)],
                mode='lines',
                name=f'Manipulation Mean {i+1}',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False
            ))
        
        # Add decision boundary at 0.5
        fig.add_trace(go.Scatter(
            x=[0.5, 0.5],
            y=[0, 1],
            mode='lines',
            name='Decision Boundary',
            line=dict(color='black', width=2, dash='dot')
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis=dict(title='Score', range=[0, 1]),
            yaxis=dict(title='Density'),
            width=800,
            height=500,
            legend=dict(x=0.01, y=0.99, bordercolor='Black', borderwidth=1)
        )
        
        return fig
    
    @staticmethod
    def plot_uncertainty_calibration(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        labels: np.ndarray,
        title: str = "Uncertainty Calibration Plot"
    ) -> go.Figure:
        """
        Create a calibration plot for uncertainty estimates.
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            labels: Ground truth labels
            title: Plot title
            
        Returns:
            Plotly figure with calibration plot
        """
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().detach().numpy()
        if isinstance(uncertainties, torch.Tensor):
            uncertainties = uncertainties.cpu().detach().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().detach().numpy()
        
        # Ensure 1D arrays
        predictions = np.atleast_1d(predictions).flatten()
        uncertainties = np.atleast_1d(uncertainties).flatten()
        labels = np.atleast_1d(labels).flatten()
        
        # Calculate errors
        errors = np.abs(predictions - labels)
        
        # Sort by uncertainty
        sorted_indices = np.argsort(uncertainties)
        sorted_errors = errors[sorted_indices]
        sorted_uncertainties = uncertainties[sorted_indices]
        
        # Calculate moving average of error
        window_size = max(len(sorted_errors) // 10, 1)
        error_avg = np.convolve(sorted_errors, np.ones(window_size) / window_size, mode='valid')
        uncertainty_avg = np.convolve(sorted_uncertainties, np.ones(window_size) / window_size, mode='valid')
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot of errors vs uncertainties
        fig.add_trace(go.Scatter(
            x=uncertainties,
            y=errors,
            mode='markers',
            name='Individual Predictions',
            marker=dict(color='blue', size=8, opacity=0.5)
        ))
        
        # Add moving average line
        fig.add_trace(go.Scatter(
            x=uncertainty_avg,
            y=error_avg,
            mode='lines',
            name='Moving Average',
            line=dict(color='red', width=3)
        ))
        
        # Add ideal calibration line (y=x)
        max_val = max(np.max(errors), np.max(uncertainties))
        fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='Ideal Calibration',
            line=dict(color='black', width=2, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis=dict(title='Uncertainty (Standard Deviation)'),
            yaxis=dict(title='Absolute Error'),
            width=800,
            height=600,
            legend=dict(x=0.01, y=0.99)
        )
        
        return fig


def norm_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Compute normal probability density function.
    
    Args:
        x: Input values
        mean: Mean of distribution
        std: Standard deviation of distribution
        
    Returns:
        PDF values
    """
    if std <= 0.01:
        std = 0.01  # Prevent division by very small values
    
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(- (x - mean)**2 / (2 * std**2))
