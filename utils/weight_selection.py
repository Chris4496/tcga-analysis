import pandas as pd
import numpy as np
import warnings
from utils.caching_script import cache_result

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)


@cache_result(verbose=2)
def cox_lasso_group_weight_selection(X, y, group_indices, group_weights=None, gamma=1.0):
    """
    Implement adaptive Lasso for Cox regression with grouped features
    
    Parameters:
    -----------
    X : array-like, shape = (n_samples, n_features)
        Training data
    y : array-like, shape = (n_samples,)
        Target variable
    group_indices : dict
        Dictionary with group names as keys and lists of feature indices as values
        Example: {
            'group1': [0, 1, 2],
            'group2': [3, 4],
            'group3': [5, 6, 7],
            'group4': [8, 9]
        }
    group_weights : dict, optional
        Initial weights for each group. If None, uses sqrt(group_size)
        Example: {
            'group1': 1.0,
            'group2': 0.5,
            'group3': 2.0,
            'group4': 1.5
        }
    gamma : float, default=1.0
        Power parameter for adaptive weights
    n_alphas : int, default=100
        Number of alpha values to try
    l1_ratio : float, default=1.0
        Elastic net mixing parameter (1.0 = Lasso)
    
    Returns:
    --------
    model : CoxnetSurvivalAnalysis
        Fitted model
    weights : array
        Final adaptive weights
    group_specific_weights : dict
        Dictionary containing the weights for each group
    """
    
    # Step 1: Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 2: Initial Cox regression without penalty
    initial_model = CoxPHSurvivalAnalysis(alpha=0.01)
    initial_model.fit(X_scaled, y)

    
    # Step 3: Calculate initial group-specific weights
    n_features = X.shape[1]
    weights = np.ones(n_features)

    
    # If group_weights not provided, use sqrt(group_size) as default
    if group_weights is None:
        group_weights = {
            group: np.sqrt(len(indices)) 
            for group, indices in group_indices.items()
        }
        
    # Step 4: Calculate adaptive weights within each group
    group_specific_weights = {}
    for group, indices in group_indices.items():
        # Get initial coefficients for this group
        group_coef = np.abs(initial_model.coef_[indices])
        
        # Calculate single adaptive weight for entire group using mean
        group_adaptive_weights = 1 / (np.mean(group_coef) ** gamma)

        # Handle zero coefficients
        if np.isinf(group_adaptive_weights) or np.isnan(group_adaptive_weights):
            group_adaptive_weights = np.finfo(float).max
            
        # Scale by group weight
        group_adaptive_weights *= group_weights[group]
        
        # Store group-specific weights (same weight repeated for all features)
        group_specific_weights[group] = np.repeat(group_adaptive_weights, len(indices))
                                                  
        # Assign to overall weights array
        weights[indices] = group_adaptive_weights

    
    # Normalize weights to sum to number of features
    weights = weights * n_features / np.sum(weights)
    
    return weights, group_specific_weights