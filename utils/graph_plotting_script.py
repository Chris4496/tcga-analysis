from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sksurv.nonparametric import kaplan_meier_estimator

def penalty_factors_plot(group_indices, weights, show_plot=True, output_path=None):
    # Create a bar plot of the penalty factors for each group
    group_names = list(group_indices.keys())
    group_weights = pd.Series(pd.unique(weights), index=group_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    group_weights.plot(kind="bar", ax=ax)
    ax.set_ylabel("penalty factor")
    ax.set_xlabel("group")
    ax.set_title("Penalty Factors for Group Lasso Regularization")

    if show_plot:
        plt.show()
    elif output_path:
        plt.savefig(output_path)

    # clear the current figure
    plt.close()


def c_index_vs_alpha_parameter_tuning_plot(cv_results, gcv, show_plot=True, output_path=None):
    alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel("concordance index")
    ax.set_xlabel("alpha")
    ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)

    if show_plot:
        plt.show()
    elif output_path:
        plt.savefig(output_path)

    # clear the current figure
    plt.close()


def top_features_plot(gcv, Xt, show_plot=True, output_path=None):
    best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
    best_coefs = pd.DataFrame(best_model.coef_, index=Xt.columns, columns=["coefficient"])

    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    print(f"Number of non-zero coefficients: {non_zero}")

    non_zero_coefs = best_coefs.query("coefficient != 0")
    coef_order = non_zero_coefs.abs().sort_values("coefficient").index

    _, ax = plt.subplots(figsize=(10, 14))
    non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
    ax.set_xlabel("coefficient")
    ax.grid(True)

    if show_plot:
        plt.show()
    elif output_path:
        plt.savefig(output_path)

    # clear the current figure
    plt.close()


def kaplan_meier_plot(data, time_col, event_col, feature_name, show_plot=True, output_path=None):
    """
    Plots Kaplan-Meier survival curves for a feature comparing all four quartiles using sksurv.
    
    Parameters:
    data (pd.DataFrame): The dataset containing the survival data.
    time_col (str): The name of the column representing the time to event or censoring.
    event_col (str): The name of the column representing the event occurrence (1 if event occurred, 0 if censored).
    feature_name (str): The name of the feature used to compare.
    show_plot (bool, optional): If True, displays the plot. Default is True.
    output_path (str, optional): If provided, saves the plot to the specified file path. Default is None.
    
    Returns:
    None
    """


    # Calculate quartile thresholds
    q1, q2, q3 = np.percentile(data[feature_name], [25, 50, 75])
    
    # Split data into quartiles
    q1_group = data[data[feature_name] <= q1]
    q2_group = data[(data[feature_name] > q1) & (data[feature_name] <= q2)]
    q3_group = data[(data[feature_name] > q2) & (data[feature_name] <= q3)]
    q4_group = data[data[feature_name] > q3]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Colors for each quartile
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot each quartile
    for idx, (group, label) in enumerate([
        (q1_group, f"Q1 ({feature_name} ≤ {q1:.2f})"),
        (q2_group, f"Q2 ({q1:.2f} < {feature_name} ≤ {q2:.2f})"),
        (q3_group, f"Q3 ({q2:.2f} < {feature_name} ≤ {q3:.2f})"),
        (q4_group, f"Q4 ({feature_name} > {q3:.2f})")
    ]):
        try:
            # Calculate KM estimate with confidence intervals
            time, survival, conf_int = kaplan_meier_estimator(
                event=group[event_col].astype(bool),  # sksurv expects boolean
                time_exit=group[time_col],
                conf_type="log-log"
            )
        except ValueError:
            return
        
        # Plot survival curve
        plt.step(time, survival, where="post", color=colors[idx], label=label)
        
        # Plot confidence intervals
        plt.fill_between(
            time,
            conf_int[0],
            conf_int[1],
            alpha=0.2,
            step="post",
            color=colors[idx]
        )
    
    plt.xlabel("Time (days)")
    plt.ylabel("Survival probability")
    plt.title(f"Kaplan-Meier Survival Curves by {feature_name} Quartiles")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    elif output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    # clear the current figure
    plt.close()