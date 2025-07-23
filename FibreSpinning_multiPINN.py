#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simultaneous vs Sequential Fluid-Structure Interaction Analysis

This script compares simultaneous and sequential approaches for
fluid-structure interaction problems using Physics-Informed Neural Networks (PINNs).

Original notebook: https://colab.research.google.com/drive/17iMnYvQkZLMU5cU-OJKbfYFc-BUq_hGU
"""

# Standard library imports
import os
import csv
import copy
import datetime
import time
from pathlib import Path

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
from tqdm import tqdm

# Set device for PyTorch (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)


# =====================================================================
# Data Loading Functions
# =====================================================================

def load_data(file_path='DATA.csv'):
    """
    Load and process data from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the data
        
    Returns:
    --------
    tuple
        Tuple containing tensors for time and the three variables (u1, u2, u3)
    """
    # Initialize empty arrays for each column
    numerical_t = []
    numerical_u1 = []
    numerical_u2 = []
    numerical_u3 = []
    
    # Create full path to the data file
    full_path = Path(file_path)
    if not full_path.is_absolute():
        # If relative path, look in the same directory as this script
        script_dir = Path(__file__).parent.absolute()
        full_path = script_dir / file_path
    
    print(f"Loading data from: {full_path}")
    
    # Open the CSV file
    with open(full_path, 'r') as csvfile:
        # Create a CSV reader object
        csvreader = csv.reader(csvfile)
        
        # Iterate over each row in the CSV file
        for row in csvreader:
            # Assuming the CSV file has four columns
            # Append each value to the corresponding array
            numerical_t.append([float(row[0])])
            numerical_u1.append([float(row[1])])
            numerical_u2.append([float(row[2])])
            numerical_u3.append([float(row[3])])
    
    # Convert to PyTorch tensors
    t_tensor = torch.tensor(numerical_t, dtype=torch.float32)
    u1_tensor = torch.tensor(numerical_u1, dtype=torch.float32)
    u2_tensor = torch.tensor(numerical_u2, dtype=torch.float32)
    u3_tensor = torch.tensor(numerical_u3, dtype=torch.float32)
    
    print(f"Loaded {len(numerical_t)} data points")
    
    return t_tensor, u1_tensor, u2_tensor, u3_tensor


# =====================================================================
# Neural Network Models
# =====================================================================

class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for fluid-structure interaction problems.
    """
    def __init__(self, layers=4, neurons=50):
        """
        Initialize the PINN model.
        
        Parameters:
        -----------
        layers : int
            Number of hidden layers
        neurons : int
            Number of neurons per hidden layer
        """
        super(PINN, self).__init__()
        
        # Network architecture
        self.input_layer = nn.Linear(1, neurons)  # Input: time t
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(layers)])
        self.output_layer = nn.Linear(neurons, 3)  # Output: u1, u2, u3
        
        # Activation function
        self.activation = nn.Tanh()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using Xavier initialization.
        """
        for layer in [self.input_layer, *self.hidden_layers, self.output_layer]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, t):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        t : torch.Tensor
            Input time tensor
            
        Returns:
        --------
        torch.Tensor
            Predicted values for u1, u2, u3
        """
        x = self.activation(self.input_layer(t))
        
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        return self.output_layer(x)


class FCN1(nn.Module):
    """Defines a fully-connected network for the first variable (u1)."""
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


class FCN2(nn.Module):
    """Defines a fully-connected network for the second variable (u2)."""
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.scale = 19
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


class FCN3(nn.Module):
    """Defines a fully-connected network for the third variable (u3)."""
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
    
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

# =====================================================================
# Uncertainty Quantification
# =====================================================================

def run_uncertainty_quantification(max_epochs=80000, num_runs=5, d=0.01, patience=1000):
    """
    Run uncertainty quantification for sequential model training.
    
    Parameters:
    -----------
    num_runs : int
        Number of runs with different random seeds
    d : float
        Parameter for the physics equations
        
    Returns:
    --------
    dict
        Dictionary containing statistics of the runs
    """
    # Create plots directory if it doesn't exist
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Create a subdirectory for uncertainty quantification results
    uq_dir = plots_dir / "uncertainty_quantification"
    uq_dir.mkdir(exist_ok=True)
    
    # Load data
    t_tensor, u1_tensor, u2_tensor, u3_tensor = load_data()
    
    # Lists to store metrics across runs
    r2_u1_values = []
    r2_u2_values = []
    r2_u3_values = []
    avg_r2_values = []
    models = []  # Store models for later analysis
    
    # Different seeds for uncertainty quantification
    seeds = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 1013]
    
    # Create progress bar for all runs
    with tqdm(total=num_runs, desc="Uncertainty quantification") as pbar:
        # Run training multiple times with different seeds
        for i, seed in enumerate(seeds[:num_runs]):
            pbar.set_description(f"Run {i+1}/{num_runs} with seed {seed}")
            
            # Set random seeds
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Train model with current seed (with reduced iterations for demonstration)
            print(f"\nTraining model for run {i+1} with seed {seed}...")
            data_tuple = (t_tensor, u1_tensor, u2_tensor, u3_tensor)
            results = train_sequential_model(num_iterations=max_epochs, d=d, data_tensors=data_tuple, patience=patience)  # Added early stopping
            
            # Store models and metrics
            models.append({
                'pinn1': results['pinn1'],
                'pinn2': results['pinn2'],
                'pinn3': results['pinn3']
            })
            
            # Calculate R² scores
            u1_pred = results['pinn1'](x_t).detach().cpu().numpy()
            u2_pred = results['pinn2'](x_t).detach().cpu().numpy()
            u3_pred = results['pinn3'](x_t).detach().cpu().numpy()
            
            u1_true = u1_tensor.detach().cpu().numpy()
            u2_true = u2_tensor.detach().cpu().numpy()
            u3_true = u3_tensor.detach().cpu().numpy()
            
            # Calculate R² scores using sklearn
            from sklearn.metrics import r2_score
            r2_u1 = r2_score(u1_true, u1_pred)
            r2_u2 = r2_score(u2_true, u2_pred)
            r2_u3 = r2_score(u3_true, u3_pred)
            avg_r2 = (r2_u1 + r2_u2 + r2_u3) / 3
            
            r2_u1_values.append(r2_u1)
            r2_u2_values.append(r2_u2)
            r2_u3_values.append(r2_u3)
            avg_r2_values.append(avg_r2)
            
            # Generate and save plots for this run
            print(f"\nGenerating plots for run {i+1} (seed {seed})...")
            
            # Get predictions
            u1 = results['pinn1'](x_t).detach()
            u2 = results['pinn2'](x_t).detach()
            u3 = results['pinn3'](x_t).detach()
            
            # Plot comparison
            fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            
            # Convert tensors to numpy for plotting
            t_np = x_t.detach().cpu().numpy()
            u1_np = u1.detach().cpu().numpy()
            u2_np = u2.detach().cpu().numpy()
            u3_np = u3.detach().cpu().numpy()
            u1_true_np = u1_tensor.detach().cpu().numpy()
            u2_true_np = u2_tensor.detach().cpu().numpy()
            u3_true_np = u3_tensor.detach().cpu().numpy()
            
            # Plot u1
            axs[0].plot(t_np, u1_np, 'b-', linewidth=2, label='Predicted')
            axs[0].plot(t_np, u1_true_np, 'r--', linewidth=2, label='Numerical')
            axs[0].set_ylabel('u1', fontsize=12)
            axs[0].set_title(f'Displacement (u1) - Run {i+1}, Seed {seed}', fontsize=14)
            axs[0].grid(True, alpha=0.3)
            axs[0].legend(fontsize=10)
            
            # Plot u2
            axs[1].plot(t_np, u2_np, 'b-', linewidth=2, label='Predicted')
            axs[1].plot(t_np, u2_true_np, 'r--', linewidth=2, label='Numerical')
            axs[1].set_ylabel('u2', fontsize=12)
            axs[1].set_title(f'Velocity (u2) - Run {i+1}, Seed {seed}', fontsize=14)
            axs[1].grid(True, alpha=0.3)
            axs[1].legend(fontsize=10)
            
            # Plot u3
            axs[2].plot(t_np, u3_np, 'b-', linewidth=2, label='Predicted')
            axs[2].plot(t_np, u3_true_np, 'r--', linewidth=2, label='Numerical')
            axs[2].set_xlabel('Time', fontsize=12)
            axs[2].set_ylabel('u3', fontsize=12)
            axs[2].set_title(f'Pressure (u3) - Run {i+1}, Seed {seed}', fontsize=14)
            axs[2].grid(True, alpha=0.3)
            axs[2].legend(fontsize=10)
            
            plt.tight_layout()
            plt.savefig(uq_dir / f"run_{i+1}_seed_{seed}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Update progress bar
            pbar.update(1)
    
    # Calculate statistics across runs
    r2_u1_array = np.array(r2_u1_values)
    r2_u2_array = np.array(r2_u2_values)
    r2_u3_array = np.array(r2_u3_values)
    avg_r2_array = np.array(avg_r2_values)
    
    # Print statistics
    print("\n===== Uncertainty Quantification Results =====")
    print(f"R² for u1: {np.mean(r2_u1_array):.6f} ± {np.std(r2_u1_array):.6f}")
    print(f"R² for u2: {np.mean(r2_u2_array):.6f} ± {np.std(r2_u2_array):.6f}")
    print(f"R² for u3: {np.mean(r2_u3_array):.6f} ± {np.std(r2_u3_array):.6f}")
    print(f"Average R²: {np.mean(avg_r2_array):.6f} ± {np.std(avg_r2_array):.6f}")
    
    # Save results to CSV with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame({
        'Run': range(1, len(r2_u1_values) + 1),
        'Seed': seeds[:len(r2_u1_values)],
        'R2_u1': r2_u1_values,
        'R2_u2': r2_u2_values,
        'R2_u3': r2_u3_values,
        'Avg_R2': avg_r2_values
    })
    
    csv_path = uq_dir / f"r2_results_by_seed_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved R² results for each seed to: {csv_path}")
    
    # Create summary plots
    # R² for each variable across runs
    plt.figure(figsize=(12, 8))
    
    variables = ['u1', 'u2', 'u3']
    r2_values = [r2_u1_values, r2_u2_values, r2_u3_values]
    r2_means = [np.mean(r2_u1_array), np.mean(r2_u2_array), np.mean(r2_u3_array)]
    r2_stds = [np.std(r2_u1_array), np.std(r2_u2_array), np.std(r2_u3_array)]
    
    for i, (var, values, mean, std) in enumerate(zip(variables, r2_values, r2_means, r2_stds)):
        plt.subplot(3, 1, i+1)
        plt.bar(range(1, len(values) + 1), values)
        plt.axhline(y=mean, color='r', linestyle='-', label=f'Mean: {mean:.6f}')
        plt.axhline(y=mean + std, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=mean - std, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Run')
        plt.ylabel(f'R² for {var}')
        plt.title(f'R² for {var} across runs')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(uq_dir / f"r2_across_runs_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Average R² across runs
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(avg_r2_values) + 1), avg_r2_values)
    plt.axhline(y=np.mean(avg_r2_array), color='r', linestyle='-', 
               label=f'Mean: {np.mean(avg_r2_array):.6f} ± {np.std(avg_r2_array):.6f}')
    plt.axhline(y=np.mean(avg_r2_array) + np.std(avg_r2_array), color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=np.mean(avg_r2_array) - np.std(avg_r2_array), color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Run')
    plt.ylabel('Average R²')
    plt.title('Average R² across runs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(uq_dir / f"avg_r2_across_runs_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a more detailed CSV with additional statistics
    detailed_stats_df = pd.DataFrame({
        'Variable': ['u1', 'u2', 'u3', 'Average'],
        'Mean_R2': [np.mean(r2_u1_array), np.mean(r2_u2_array), np.mean(r2_u3_array), np.mean(avg_r2_array)],
        'Std_R2': [np.std(r2_u1_array), np.std(r2_u2_array), np.std(r2_u3_array), np.std(avg_r2_array)],
        'Min_R2': [np.min(r2_u1_array), np.min(r2_u2_array), np.min(r2_u3_array), np.min(avg_r2_array)],
        'Max_R2': [np.max(r2_u1_array), np.max(r2_u2_array), np.max(r2_u3_array), np.max(avg_r2_array)]
    })
    
    detailed_stats_path = uq_dir / f"r2_detailed_stats_{timestamp}.csv"
    detailed_stats_df.to_csv(detailed_stats_path, index=False)
    print(f"Saved detailed R² statistics to: {detailed_stats_path}")
    
    # Plot mean of all results
    # Collect predictions from all models
    all_u1_preds = []
    all_u2_preds = []
    all_u3_preds = []
    
    for model_set in models:
        u1_pred = model_set['pinn1'](x_t).detach().cpu().numpy()
        u2_pred = model_set['pinn2'](x_t).detach().cpu().numpy()
        u3_pred = model_set['pinn3'](x_t).detach().cpu().numpy()
        
        all_u1_preds.append(u1_pred)
        all_u2_preds.append(u2_pred)
        all_u3_preds.append(u3_pred)
    
    # Calculate mean and std of predictions
    mean_u1 = np.mean(all_u1_preds, axis=0)
    mean_u2 = np.mean(all_u2_preds, axis=0)
    mean_u3 = np.mean(all_u3_preds, axis=0)
    
    std_u1 = np.std(all_u1_preds, axis=0)
    std_u2 = np.std(all_u2_preds, axis=0)
    std_u3 = np.std(all_u3_preds, axis=0)
    
    # Plot mean results with confidence intervals
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    t_np = x_t.detach().cpu().numpy()
    u1_true_np = u1_tensor.detach().cpu().numpy()
    u2_true_np = u2_tensor.detach().cpu().numpy()
    u3_true_np = u3_tensor.detach().cpu().numpy()
    
    # Plot u1
    axs[0].plot(t_np, mean_u1, 'b-', linewidth=2, label='Mean Prediction')
    axs[0].fill_between(t_np.flatten(), (mean_u1 - std_u1).flatten(), (mean_u1 + std_u1).flatten(), 
                       color='b', alpha=0.2, label='±1 Std Dev')
    axs[0].plot(t_np, u1_true_np, 'r--', linewidth=2, label='Numerical')
    axs[0].set_ylabel('u1', fontsize=12)
    axs[0].set_title(f'Mean Displacement (u1) - R²: {np.mean(r2_u1_array):.6f} ± {np.std(r2_u1_array):.6f}', fontsize=14)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=10)
    
    # Plot u2
    axs[1].plot(t_np, mean_u2, 'b-', linewidth=2, label='Mean Prediction')
    axs[1].fill_between(t_np.flatten(), (mean_u2 - std_u2).flatten(), (mean_u2 + std_u2).flatten(), 
                       color='b', alpha=0.2, label='±1 Std Dev')
    axs[1].plot(t_np, u2_true_np, 'r--', linewidth=2, label='Numerical')
    axs[1].set_ylabel('u2', fontsize=12)
    axs[1].set_title(f'Mean Velocity (u2) - R²: {np.mean(r2_u2_array):.6f} ± {np.std(r2_u2_array):.6f}', fontsize=14)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=10)
    
    # Plot u3
    axs[2].plot(t_np, mean_u3, 'b-', linewidth=2, label='Mean Prediction')
    axs[2].fill_between(t_np.flatten(), (mean_u3 - std_u3).flatten(), (mean_u3 + std_u3).flatten(), 
                       color='b', alpha=0.2, label='±1 Std Dev')
    axs[2].plot(t_np, u3_true_np, 'r--', linewidth=2, label='Numerical')
    axs[2].set_xlabel('Time', fontsize=12)
    axs[2].set_ylabel('u3', fontsize=12)
    axs[2].set_title(f'Mean Pressure (u3) - R²: {np.mean(r2_u3_array):.6f} ± {np.std(r2_u3_array):.6f}', fontsize=14)
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(uq_dir / f"mean_results_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'r2_mean': {
            'u1': np.mean(r2_u1_array),
            'u2': np.mean(r2_u2_array),
            'u3': np.mean(r2_u3_array),
            'avg': np.mean(avg_r2_array)
        },
        'r2_std': {
            'u1': np.std(r2_u1_array),
            'u2': np.std(r2_u2_array),
            'u3': np.std(r2_u3_array),
            'avg': np.std(avg_r2_array)
        },
        'models': models,
        'csv_path': csv_path
    }
# =====================================================================
# Plotting Functions
# =====================================================================

def plot_comparison(x_p, x_b_0, x_b_1, x_t, predicted, numerical, variable_name, save_path=None):
    """
    Plot comparison between predicted and numerical values.
    
    Parameters:
    -----------
    x_p : torch.Tensor
        Collocation points
    x_b_0 : torch.Tensor
        Boundary points at t=0
    x_b_1 : torch.Tensor
        Boundary points at t=1
    x_t : torch.Tensor
        Time points for evaluation
    predicted : torch.Tensor
        Predicted values from the model
    numerical : torch.Tensor
        Numerical reference values
    variable_name : str
        Name of the variable being plotted (e.g., 'u1', 'u2', 'u3')
    save_path : Path or str, optional
        Path to save the figure, if None, the figure is not saved
    """
    plt.figure(figsize=(10, 6))
    
    # Plot collocation and boundary points
    plt.scatter(x_p.detach()[:, 0], torch.zeros_like(x_p)[:, 0], s=20, lw=0, 
                color="tab:green", alpha=0.6, label="Collocation Points")
    plt.scatter(x_b_0.detach()[:, 0], torch.zeros_like(x_b_0)[:, 0], s=20, lw=0, 
                color="tab:red", alpha=0.6, label="Boundary Points (t=0)")
    plt.scatter(x_b_1.detach()[:, 0], torch.zeros_like(x_b_1)[:, 0], s=20, lw=0, 
                color="tab:purple", alpha=0.6, label="Boundary Points (t=1)")
    
    # Plot predicted and numerical values
    plt.plot(x_t[:, 0], predicted, label=f"Predicted {variable_name}", color="tab:blue", linewidth=2)
    plt.plot(x_t[:, 0], numerical, label=f"Numerical {variable_name}", color="tab:orange", linewidth=2, linestyle='--')
    
    # Add labels and legend
    plt.xlabel('Time', fontsize=12)
    plt.ylabel(variable_name, fontsize=12)
    plt.title(f"Comparison of Predicted vs Numerical {variable_name}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Close the figure instead of showing it to avoid pop-up windows
    plt.close()


def plot_loss_history(loss_history, title="Training Loss", save_path=None):
    """
    Plot training loss history.
    
    Parameters:
    -----------
    loss_history : list
        List of loss values during training
    title : str
        Title for the plot
    save_path : Path or str, optional
        Path to save the figure, if None, the figure is not saved
    """
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history, linewidth=2)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Close the figure instead of showing it to avoid pop-up windows
    plt.close()


def plot_all_variables(t_tensor, u1_pred, u2_pred, u3_pred, u1_true, u2_true, u3_true, save_path=None):
    """
    Plot all variables (u1, u2, u3) in a single figure with subplots.
    
    Parameters:
    -----------
    t_tensor : torch.Tensor
        Time points
    u1_pred, u2_pred, u3_pred : torch.Tensor
        Predicted values for u1, u2, u3
    u1_true, u2_true, u3_true : torch.Tensor
        True/numerical values for u1, u2, u3
    save_path : Path or str, optional
        Path to save the figure, if None, the figure is not saved
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Convert tensors to numpy for plotting if they are not already
    t_np = t_tensor.detach().cpu().numpy() if isinstance(t_tensor, torch.Tensor) else t_tensor
    u1_pred_np = u1_pred.detach().cpu().numpy() if isinstance(u1_pred, torch.Tensor) else u1_pred
    u2_pred_np = u2_pred.detach().cpu().numpy() if isinstance(u2_pred, torch.Tensor) else u2_pred
    u3_pred_np = u3_pred.detach().cpu().numpy() if isinstance(u3_pred, torch.Tensor) else u3_pred
    u1_true_np = u1_true.detach().cpu().numpy() if isinstance(u1_true, torch.Tensor) else u1_true
    u2_true_np = u2_true.detach().cpu().numpy() if isinstance(u2_true, torch.Tensor) else u2_true
    u3_true_np = u3_true.detach().cpu().numpy() if isinstance(u3_true, torch.Tensor) else u3_true
    
    # Plot u1
    axs[0].plot(t_np, u1_pred_np, 'b-', linewidth=2, label='Predicted')
    axs[0].plot(t_np, u1_true_np, 'r--', linewidth=2, label='Numerical')
    axs[0].set_ylabel('u1', fontsize=12)
    axs[0].set_title('Displacement (u1)', fontsize=14)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=10)
    
    # Plot u2
    axs[1].plot(t_np, u2_pred_np, 'b-', linewidth=2, label='Predicted')
    axs[1].plot(t_np, u2_true_np, 'r--', linewidth=2, label='Numerical')
    axs[1].set_ylabel('u2', fontsize=12)
    axs[1].set_title('Velocity (u2)', fontsize=14)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=10)
    
    # Plot u3
    axs[2].plot(t_np, u3_pred_np, 'b-', linewidth=2, label='Predicted')
    axs[2].plot(t_np, u3_true_np, 'r--', linewidth=2, label='Numerical')
    axs[2].set_xlabel('Time', fontsize=12)
    axs[2].set_ylabel('u3', fontsize=12)
    axs[2].set_title('Pressure (u3)', fontsize=14)
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Close the figure instead of showing it to avoid pop-up windows
    plt.close()
    
    # Calculate and return metrics
    metrics = {
        'u1_r2': r2_score(u1_true_np.flatten(), u1_pred_np.flatten()),
        'u2_r2': r2_score(u2_true_np.flatten(), u2_pred_np.flatten()),
        'u3_r2': r2_score(u3_true_np.flatten(), u3_pred_np.flatten()),
        'u1_mse': np.mean((u1_true_np.flatten() - u1_pred_np.flatten())**2),
        'u2_mse': np.mean((u2_true_np.flatten() - u2_pred_np.flatten())**2),
        'u3_mse': np.mean((u3_true_np.flatten() - u3_pred_np.flatten())**2)
    }
    
    return metrics


# =====================================================================
# Training Implementation
# =====================================================================

# Define boundary points for the boundary loss
x_b_0 = torch.tensor(0.).view(-1,1).requires_grad_(True)
x_b_1 = torch.tensor(1.).view(-1,1).requires_grad_(True)

# Define training points over the entire domain for the physics loss
x_p = torch.linspace(0,1,301).view(-1,1).requires_grad_(True)

# Test points
x_t = torch.linspace(0,1,301).view(-1,1)

def train_sequential_model(num_iterations=10000, d=0.01, data_tensors=None, patience=1000):
    """
    Train the sequential model for fluid-structure interaction.
    
    Parameters:
    -----------
    num_iterations : int
        Number of training iterations
    d : float
        Parameter for the physics equations
    data_tensors : tuple, optional
        Tuple of (t_tensor, u1_tensor, u2_tensor, u3_tensor) for evaluation
    patience : int, optional
        Number of iterations to wait for improvement before early stopping
        
    Returns:
    --------
    dict
        Dictionary containing the trained models and metrics
    """
    # Initialize models
    pinn1 = FCN1(1, 1, 32, 4)
    pinn2 = FCN2(1, 1, 32, 4)
    pinn3 = FCN3(1, 1, 32, 4)
    
    # Initialize optimizers
    optimiser1 = torch.optim.Adam(pinn1.parameters(), lr=1e-3)
    optimiser2 = torch.optim.Adam(pinn2.parameters(), lr=1e-3)
    optimiser3 = torch.optim.Adam(pinn3.parameters(), lr=1e-3)
    
    # Loss weights
    lambdas = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.000000001]
    
    # Initialize tracking arrays
    physics_u1s = []
    physics_u2s = []
    physics_u3s = []
    boundary_u1s = []
    boundary_u2s = []
    u1u2 = []
    u1u3 = []
    
    # For early stopping
    best_loss = float('inf')
    best_models = None
    counter = 0
    
    # Initial forward pass
    u1_physics = pinn1(x_p)
    du1dx = torch.autograd.grad(u1_physics, x_p, torch.ones_like(u1_physics), create_graph=True)[0]
    u2_physics = pinn2(x_p)
    du2dx = torch.autograd.grad(u2_physics, x_p, torch.ones_like(u2_physics), create_graph=True)[0]
    u3_physics = pinn3(x_p)
    du3dx = torch.autograd.grad(u3_physics, x_p, torch.ones_like(u3_physics), create_graph=True)[0]
    
    # Training loop
    i = 0
    with tqdm(total=num_iterations, desc="Sequential Training") as pbar:
        while i <= num_iterations:
            # Number of inner iterations for each PINN (same for all three)  
            inner_iterations = 30
            
            # Train u1
            for j in range(inner_iterations):
                optimiser1.zero_grad()
                u1_b_0 = pinn1(x_b_0)  # (1, 1)
                u1_b_1 = pinn1(x_b_1)  # (1, 1)
                u1_physics = pinn1(x_p)  # (300,1)
                du1dx = torch.autograd.grad(u1_physics, x_p, torch.ones_like(u1_physics), create_graph=True)[0]
                
                u1_loss1 = (torch.squeeze(u1_b_0) - 1)**2
                u1_loss2 = torch.mean((u1_physics * du2dx.detach() + u2_physics.detach() * du1dx)**2)
                u1_loss3 = torch.mean((u3_physics.detach() * du1dx + u1_physics * du3dx.detach())**2)
                physics_loss_u1 = u1_loss3 + u1_loss2
                boundary_loss_u1 = u1_loss1
                
                u1_loss = lambdas[0] * u1_loss1 + lambdas[3] * (u1_loss2 + u1_loss3)
                u1_loss.backward()
                optimiser1.step()
            
            # Train u2
            for j in range(inner_iterations):
                optimiser2.zero_grad()
                u2_b_0 = pinn2(x_b_0)  # (1, 1)
                u2_b_1 = pinn2(x_b_1)  # (1, 1)
                u2_physics = pinn2(x_p)  # (300,1)
                du2dx = torch.autograd.grad(u2_physics, x_p, torch.ones_like(u2_physics), create_graph=True)[0]
                
                u2_loss1 = torch.mean((torch.squeeze(u2_b_0))**2) + torch.mean((torch.squeeze(u2_b_1))**2)
                u2_loss2 = torch.mean((u1_physics.detach() * du2dx + u2_physics * du1dx.detach())**2)
                u2_loss3 = torch.mean((2 * (u3_physics.detach() + 1/d) * du2dx - u2_physics * du3dx.detach() - (1/d) * u3_physics.detach())**2)
                physics_loss_u2 = u2_loss2 + u2_loss3
                boundary_loss_u2 = u2_loss1
                
                u2_loss = lambdas[1] * u2_loss1 + lambdas[4] * u2_loss2 + lambdas[5] * u2_loss3
                u2_loss.backward()
                optimiser2.step()
            
            # Train u3
            for j in range(inner_iterations):
                optimiser3.zero_grad()
                u3_physics = pinn3(x_p)
                du3dx = torch.autograd.grad(u3_physics, x_p, torch.ones_like(u3_physics), create_graph=True)[0]
                
                u3_loss1 = torch.mean((u3_physics * du1dx.detach() + u1_physics.detach() * du3dx)**2)
                u3_loss2 = torch.mean((u2_physics.detach() * du3dx - 2 * (u3_physics + 1/d) * du2dx.detach() + (1/d) * u3_physics)**2)
                physics_loss_u3 = u3_loss1 + u3_loss2
                
                u3_loss = lambdas[3] * u3_loss1 + lambdas[7] * u3_loss2
                u3_loss.backward()
                optimiser3.step()
            
            # Store metrics
            boundary_u1s.append(boundary_loss_u1.detach())
            physics_u1s.append(physics_loss_u1.detach())
            u1u2.append(torch.mean((u1_physics * u2_physics)).detach())
            u1u3.append(torch.mean((u1_physics * u3_physics)).detach())
            boundary_u2s.append(boundary_loss_u2.detach())
            physics_u2s.append(physics_loss_u2.detach())
            physics_u3s.append(physics_loss_u3.detach())
            
            # Calculate total loss for early stopping
            total_loss = physics_loss_u1.item() + physics_loss_u2.item() + physics_loss_u3.item() + \
                        boundary_loss_u1.item() + boundary_loss_u2.item()
            
            # Early stopping check
            if total_loss < best_loss:
                best_loss = total_loss
                # Save best models
                best_models = {
                    'pinn1': copy.deepcopy(pinn1),
                    'pinn2': copy.deepcopy(pinn2),
                    'pinn3': copy.deepcopy(pinn3)
                }
                counter = 0
            else:
                counter += 1
                
            # Check if we should stop early
            if counter >= patience:
                print(f"\nEarly stopping triggered after {i} iterations")
                if best_models is not None:
                    pinn1 = best_models['pinn1']
                    pinn2 = best_models['pinn2']
                    pinn3 = best_models['pinn3']
                break
                
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'u1_phys': f"{physics_loss_u1.item():.2e}",
                'u2_phys': f"{physics_loss_u2.item():.2e}",
                'u3_phys': f"{physics_loss_u3.item():.2e}",
                'no_improve': counter
            })
            
            i += 1
    
    # Evaluate results
    u1 = pinn1(x_t).detach()
    u2 = pinn2(x_t).detach()
    u3 = pinn3(x_t).detach()
    
    # Calculate MSE using the loaded data tensors if provided
    if data_tensors is not None:
        _, u1_true, u2_true, u3_true = data_tensors
        mse_u1 = torch.mean(((u1_true) - (u1)) ** 2).detach()
        mse_u2 = torch.mean(((u2_true) - (u2)) ** 2).detach()
        mse_u3 = torch.mean(((u3_true) - (u3)) ** 2).detach()
    else:
        # If no data tensors provided, just use zeros as placeholder
        mse_u1 = torch.tensor(0.0)
        mse_u2 = torch.tensor(0.0)
        mse_u3 = torch.tensor(0.0)
    
    print(f"MSE values: u1={mse_u1.item():.6e}, u2={mse_u2.item():.6e}, u3={mse_u3.item():.6e}")
    print(f"Total MSE: {(mse_u1 + mse_u2 + mse_u3).item():.6e}")
    
    # Calculate and print R² values if data tensors are provided
    if data_tensors is not None:
        _, u1_true, u2_true, u3_true = data_tensors
        u1_np = u1.detach().cpu().numpy()
        u2_np = u2.detach().cpu().numpy()
        u3_np = u3.detach().cpu().numpy()
        u1_true_np = u1_true.detach().cpu().numpy()
        u2_true_np = u2_true.detach().cpu().numpy()
        u3_true_np = u3_true.detach().cpu().numpy()
        
        r2_u1 = r2_score(u1_true_np, u1_np)
        r2_u2 = r2_score(u2_true_np, u2_np)
        r2_u3 = r2_score(u3_true_np, u3_np)
        avg_r2 = (r2_u1 + r2_u2 + r2_u3) / 3
        
        print(f"R² values: u1={r2_u1:.6f}, u2={r2_u2:.6f}, u3={r2_u3:.6f}")
        print(f"Average R²: {avg_r2:.6f}")
    
    # Plot results if data tensors are provided
    if data_tensors is not None:
        _, u1_true, u2_true, u3_true = data_tensors
        plot_comparison(x_p, x_b_0, x_b_1, x_t, u1, u1_true, "u1", None)
        plot_comparison(x_p, x_b_0, x_b_1, x_t, u2, u2_true, "u2", None)
        plot_comparison(x_p, x_b_0, x_b_1, x_t, u3, u3_true, "u3", None)
    # Skip plotting if no data tensors are provided
    
    # Plot loss histories
    plot_loss_history(physics_u1s, "Physics Loss for u1")
    plot_loss_history(physics_u2s, "Physics Loss for u2")
    plot_loss_history(physics_u3s, "Physics Loss for u3")
    plot_loss_history(boundary_u1s, "Boundary Loss for u1")
    plot_loss_history(boundary_u2s, "Boundary Loss for u2")
    
    return {
        'pinn1': pinn1,
        'pinn2': pinn2,
        'pinn3': pinn3,
        'mse': {
            'u1': mse_u1.item(),
            'u2': mse_u2.item(),
            'u3': mse_u3.item(),
            'total': (mse_u1 + mse_u2 + mse_u3).item()
        },
        'loss_history': {
            'physics_u1': physics_u1s,
            'physics_u2': physics_u2s,
            'physics_u3': physics_u3s,
            'boundary_u1': boundary_u1s,
            'boundary_u2': boundary_u2s
        }
    }


def train_simultaneous_model():
    """Placeholder for simultaneous model training implementation."""
    # TODO: Implement simultaneous training approach
    pass


# =====================================================================
# Hyperparameter Grid Search Implementation
# =====================================================================

def run_grid_search(max_iterations=10000, patience=1000):
    """
    Run grid search for hyperparameter optimization.
    
    Parameters:
    -----------
    max_iterations : int
        Maximum number of training iterations per model
    patience : int
        Number of iterations to wait for improvement before early stopping
        
    Returns:
    --------
    dict
        Dictionary containing the best hyperparameters and model performance
    """
    print("\n" + "=" * 50)
    print("Starting Hyperparameter Grid Search")
    print("=" * 50)
    
    # Define hyperparameter grid (reduced options as requested)
    hyperparameter_grid = {
        'n_hidden': [16, 32, 64, 128],          # Number of neurons per hidden layer
        'n_layers': [4, 5, 6, 7],             # Number of hidden layers
        'learning_rate': [1e-3, 5e-4, 1e-4]  # Learning rate for Adam optimizer
    }
    
    # Previous values used in the original implementation:
    print("\nPrevious hyperparameter values:")
    print("  - Number of hidden neurons: 50 (default in PINN class)")
    print("  - Number of hidden layers: 4 (default in PINN class)")
    print("  - Learning rate: Not explicitly set (PyTorch default: 1e-3)")
    
    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent / "hyperparam_results"
    results_dir.mkdir(exist_ok=True)
    
    # Load data
    t_tensor, u1_tensor, u2_tensor, u3_tensor = load_data()
    data_tensors = (t_tensor, u1_tensor, u2_tensor, u3_tensor)
    
    # Generate all combinations of hyperparameters
    import itertools
    param_combinations = list(itertools.product(
        hyperparameter_grid['n_hidden'],
        hyperparameter_grid['n_layers'],
        hyperparameter_grid['learning_rate']
    ))
    
    print(f"\nTotal number of combinations to test: {len(param_combinations)}")
    
    # Store results
    all_results = []
    best_result = None
    best_mse = float('inf')
    
    # Test each combination
    for idx, (n_hidden, n_layers, learning_rate) in enumerate(param_combinations):
        print(f"\nCombination {idx+1}/{len(param_combinations)}:")
        print(f"  - Number of hidden neurons: {n_hidden}")
        print(f"  - Number of hidden layers: {n_layers}")
        print(f"  - Learning rate: {learning_rate}")
        
        # Create models with specified hyperparameters
        pinn1 = FCN1(1, 1, n_hidden, n_layers)
        pinn2 = FCN2(1, 1, n_hidden, n_layers)
        pinn3 = FCN3(1, 1, n_hidden, n_layers)
        
        # Move models to device
        pinn1 = pinn1.to(device)
        pinn2 = pinn2.to(device)
        pinn3 = pinn3.to(device)
        
        # Create optimizers with specified learning rate
        optimiser1 = torch.optim.Adam(pinn1.parameters(), lr=learning_rate)
        optimiser2 = torch.optim.Adam(pinn2.parameters(), lr=learning_rate)
        optimiser3 = torch.optim.Adam(pinn3.parameters(), lr=learning_rate)
        
        # Define boundary points and collocation points
        x_b_0 = torch.tensor(0.).view(-1,1).requires_grad_(True).to(device)
        x_b_1 = torch.tensor(1.).view(-1,1).requires_grad_(True).to(device)
        x_p = torch.linspace(0,1,301).view(-1,1).requires_grad_(True).to(device)
        x_t = torch.linspace(0,1,301).view(-1,1).to(device)
        
        # Initialize variables
        i = 0
        d = 0.01  # Physics parameter
        lambdas = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Loss weights
        
        # For early stopping
        best_total_mse = float('inf')
        patience_counter = 0
        best_models = None
        
        # Training loop
        start_time = time.time()
        
        # Loss history
        loss_history = []
        
        # Training loop (sequential training approach)
        with tqdm(total=max_iterations, desc=f"Training combination {idx+1}") as pbar:
            while i <= max_iterations:
                # u1 training
                optimiser1.zero_grad()
                u1_b_0 = pinn1(x_b_0)
                u1_b_1 = pinn1(x_b_1)
                u1_physics = pinn1(x_p)
                du1dx = torch.autograd.grad(u1_physics, x_p, torch.ones_like(u1_physics), create_graph=True)[0]
                
                # u2 training
                optimiser2.zero_grad()
                u2_b_0 = pinn2(x_b_0)
                u2_b_1 = pinn2(x_b_1)
                u2_physics = pinn2(x_p)
                du2dx = torch.autograd.grad(u2_physics, x_p, torch.ones_like(u2_physics), create_graph=True)[0]
                
                # u3 training
                optimiser3.zero_grad()
                u3_physics = pinn3(x_p)
                du3dx = torch.autograd.grad(u3_physics, x_p, torch.ones_like(u3_physics), create_graph=True)[0]
                
                # Calculate losses for u1
                u1_loss1 = (torch.squeeze(u1_b_0) - 1)**2
                u1_loss2 = torch.mean((u1_physics* du2dx.detach() + u2_physics.detach() * du1dx)**2)
                u1_loss3 = torch.mean((u3_physics.detach() *du1dx + u1_physics*du3dx.detach() )**2)
                physics_loss_u1 = u1_loss3 + u1_loss2
                boundary_loss_u1 = u1_loss1
                u1_loss = lambdas[0]*u1_loss1 + lambdas[3]*(u1_loss2 + u1_loss3)
                
                # Calculate losses for u2
                u2_loss1 = (torch.squeeze(u2_b_0) - 1)**2
                u2_loss2 = (torch.squeeze(u2_b_1) - 20)**2
                u2_loss3 = torch.mean((u1_physics.detach() * du2dx + u2_physics * du1dx.detach())**2)
                u2_loss4 = torch.mean((u2_physics *du3dx.detach() - 2 * (u3_physics.detach() + 1/d)*du2dx + (1/d)*u3_physics.detach())**2)
                physics_loss_u2 = u2_loss3 + u2_loss4
                boundary_loss_u2 = u2_loss1 + u2_loss2
                u2_loss = lambdas[0]*(u2_loss1+u2_loss2) + lambdas[3]*u2_loss3 + lambdas[3] * u2_loss4
                
                # Calculate losses for u3
                u3_loss1 = torch.mean((u3_physics*du1dx.detach() + u1_physics.detach()*du3dx )**2)
                u3_loss2 = torch.mean((u2_physics.detach() *du3dx - 2 * (u3_physics + 1/d)*du2dx.detach() + (1/d)*u3_physics)**2)
                physics_loss_u3 = u3_loss1 + u3_loss2
                u3_loss = lambdas[3] * u3_loss1 + lambdas[6] * u3_loss2
                
                # Backpropagation and optimization step
                u1_loss.backward()
                optimiser1.step()
                
                u2_loss.backward()
                optimiser2.step()
                
                u3_loss.backward()
                optimiser3.step()
                
                # Store total loss
                total_loss = u1_loss.item() + u2_loss.item() + u3_loss.item()
                loss_history.append(total_loss)
                
                # Calculate MSE against numerical solution every 100 iterations
                if i % 100 == 0:
                    # Calculate MSE against numerical data
                    mse_u1 = torch.mean(((u1_tensor.to(device)) - (u1_physics)) ** 2).detach()
                    mse_u2 = torch.mean(((u2_tensor.to(device)) - (u2_physics)) ** 2).detach()
                    mse_u3 = torch.mean(((u3_tensor.to(device)) - (u3_physics)) ** 2).detach()
                    total_mse = mse_u1 + mse_u2 + mse_u3
                    
                    # Early stopping check
                    if total_mse < best_total_mse:
                        best_total_mse = total_mse
                        patience_counter = 0
                        # Save best models
                        best_models = {
                            'pinn1': copy.deepcopy(pinn1.state_dict()),
                            'pinn2': copy.deepcopy(pinn2.state_dict()),
                            'pinn3': copy.deepcopy(pinn3.state_dict())
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"\nEarly stopping at iteration {i}")
                            break
                
                # Update progress bar
                if i % 10 == 0:
                    pbar.update(10)
                    pbar.set_postfix({"loss": total_loss, "best_mse": best_total_mse.item()})
                
                i += 1
        
        # Calculate final MSE
        mse_u1 = torch.mean(((u1_tensor.to(device)) - (u1_physics)) ** 2).detach()
        mse_u2 = torch.mean(((u2_tensor.to(device)) - (u2_physics)) ** 2).detach()
        mse_u3 = torch.mean(((u3_tensor.to(device)) - (u3_physics)) ** 2).detach()
        total_mse = mse_u1 + mse_u2 + mse_u3
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Store result
        result = {
            'n_hidden': n_hidden,
            'n_layers': n_layers,
            'learning_rate': learning_rate,
            'mse_u1': mse_u1.item(),
            'mse_u2': mse_u2.item(),
            'mse_u3': mse_u3.item(),
            'total_mse': total_mse.item(),
            'iterations': i,
            'training_time': training_time,
            'best_models': best_models
        }
        
        all_results.append(result)
        
        # Check if this is the best result so far
        if total_mse < best_mse:
            best_mse = total_mse
            best_result = result
            
            # Save best models
            if best_models is not None:
                # Create models with best hyperparameters
                best_pinn1 = FCN1(1, 1, n_hidden, n_layers).to(device)
                best_pinn2 = FCN2(1, 1, n_hidden, n_layers).to(device)
                best_pinn3 = FCN3(1, 1, n_hidden, n_layers).to(device)
                
                # Load best weights
                best_pinn1.load_state_dict(best_models['pinn1'])
                best_pinn2.load_state_dict(best_models['pinn2'])
                best_pinn3.load_state_dict(best_models['pinn3'])
                
                # Save models
                torch.save(best_pinn1.state_dict(), results_dir / 'best_pinn1.pt')
                torch.save(best_pinn2.state_dict(), results_dir / 'best_pinn2.pt')
                torch.save(best_pinn3.state_dict(), results_dir / 'best_pinn3.pt')
                
                # Generate plots for best model
                print("\nGenerating plots for best model...")
                u1_pred = best_pinn1(x_t)
                u2_pred = best_pinn2(x_t)
                u3_pred = best_pinn3(x_t)
                
                # Plot all variables
                plot_all_variables(
                    x_t.cpu(), 
                    u1_pred.detach().cpu(), 
                    u2_pred.detach().cpu(), 
                    u3_pred.detach().cpu(),
                    u1_tensor, 
                    u2_tensor, 
                    u3_tensor,
                    save_path=results_dir / "best_model_predictions.png"
                )
    
    # Save all results to JSON file
    import json
    results_for_json = []
    for result in all_results:
        result_copy = result.copy()
        # Remove model state dictionaries as they can't be serialized to JSON
        if 'best_models' in result_copy:
            del result_copy['best_models']
        results_for_json.append(result_copy)
    
    with open(results_dir / 'grid_search_results.json', 'w') as f:
        json.dump(results_for_json, f, indent=4)
    
    # Print best hyperparameters
    print("\n" + "=" * 50)
    print("Grid Search Complete!")
    print("=" * 50)
    print(f"Best hyperparameters:")
    print(f"  - Number of hidden neurons: {best_result['n_hidden']}")
    print(f"  - Number of hidden layers: {best_result['n_layers']}")
    print(f"  - Learning rate: {best_result['learning_rate']}")
    print(f"Best total MSE: {best_result['total_mse']:.6f}")
    print(f"Individual MSEs: u1={best_result['mse_u1']:.6f}, u2={best_result['mse_u2']:.6f}, u3={best_result['mse_u3']:.6f}")
    print(f"Training time: {best_result['training_time']:.2f} seconds")
    print(f"Iterations: {best_result['iterations']}")
    print(f"\nBest models saved to '{results_dir}'")
    
    return best_result


# =====================================================================
# Main Function
# =====================================================================

def main():
    """
    Main function to execute the script.
    """
    # Load data
    t_tensor, u1_tensor, u2_tensor, u3_tensor = load_data()
    
    # Create a plots directory if it doesn't exist
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot the data
    plot_all_variables(
        t_tensor, 
        u1_tensor, u2_tensor, u3_tensor,  # Using same data for predicted and true for initial visualization
        u1_tensor, u2_tensor, u3_tensor,
        save_path=plots_dir / "data_visualization.png"
    )
    
    print(f"\nData visualization saved to {plots_dir / 'data_visualization.png'}")
    
    # Uncomment one of the following options to run:
    
    # Option 1: Train a single sequential model
    # results = train_sequential_model(num_iterations=100, patience=1000, data_tensors=(t_tensor, u1_tensor, u2_tensor, u3_tensor))
    # print(f"\nTraining completed with MSE: {results['mse']['total']:.6e}")
    
    # Option 2: Run uncertainty quantification
    # uq_results = run_uncertainty_quantification(num_runs=5, patience=1000)  # Added early stopping with patience
    # print(f"\nUncertainty quantification completed with average R²: {uq_results['r2_mean']['avg']:.6f} ± {uq_results['r2_std']['avg']:.6f}")
    # print(f"Higher R² values indicate better model performance (closer to 1.0 is better)")
    
    # Option 3: Run hyperparameter grid search
    best_result = run_grid_search(max_iterations=40000, patience=1000)
    print(f"\nGrid search completed. Best model saved with MSE: {best_result['total_mse']:.6e}")
    print(f"Best hyperparameters: {best_result['n_hidden']} neurons, {best_result['n_layers']} layers, learning rate {best_result['learning_rate']}")
    print(f"Check the 'hyperparam_results' directory for detailed results and model files.")
    
    # Option 3: Train a simultaneous model (placeholder)
    # Implement simultaneous model training
    
    print("\nAll tasks completed successfully.")

if __name__ == "__main__":
    main()
