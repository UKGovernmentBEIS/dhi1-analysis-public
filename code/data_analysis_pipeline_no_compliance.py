#!/usr/bin/env python3
"""
Full GLM pipeline for main outcome analyses.

This is a variant of data_analysis_pipeline.py that EXCLUDES the subject-level
compliance score (sub_compliance) as a covariate from all GLMs.
The original pipeline retaining sub_compliance is preserved in data_analysis_pipeline.py.

This is the CANONICAL pipeline: its outputs are the ones in ../parameter_estimates/,
and they are the results reported in the paper. The sub_compliance variant's outputs
were moved to ../parameter_estimates_compliance_cov/ by hand -- note that both scripts
write to ../parameter_estimates/, so running the other one overwrites these.

Run one combination:
    python data_analysis_pipeline_no_compliance.py --dataset trust --prompting 0

Run all eight combinations:
    bash run_pipeline.sh
"""

# For a long unattended run:
# tmux new -s pipeline
# cd code/
# bash run_pipeline.sh 2>&1 | tee pipeline_output.log
#  Ctrl+B, then D to detach
# tmux attach -t pipeline

import os
import sys
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

import argparse

# Parse arguments BEFORE any JAX imports
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True,
                    choices=['misinfo', 'trust', 'private', 'extreme'])
parser.add_argument('--prompting', type=int, required=True, choices=[0, 1])
args = parser.parse_args()


# Set environment before importing JAX
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

# Non-interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import jax
import jax.numpy as jnp
import jax.random as random

logger.info(f"Devices: {jax.devices()}")
logger.info(f"Default backend: {jax.default_backend()}")

import numpyro
import numpyro.distributions as dist
from numpyro import sample
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.infer.reparam import TransformReparam
from numpyro.handlers import reparam
from numpyro.distributions import transforms
from numpyro.infer.initialization import init_to_feasible
from numpyro.infer import Predictive

numpyro.set_platform("cuda")

from itertools import combinations
from statsmodels.stats.outliers_influence import variance_inflation_factor
import arviz as az

rng_key = jax.random.PRNGKey(1)
num_chains = 4

rope = 0.05
rope_interval = np.array([-rope, rope])

logger.info(f"Local device count: {jax.local_device_count()}")

# Ensure output directories exist
os.makedirs('../plots', exist_ok=True)
os.makedirs('../parameter_estimates', exist_ok=True)


# ============================================================
# FUNCTIONS
# ============================================================

import gc

def clear_gpu_memory():
    """Force clear GPU memory between model fits."""
    gc.collect()
    jax.clear_caches()
    logger.info("Cleared GPU memory caches")


def prepare_data(df, subset_fraction=0.01, use_subset=True, interaction_level=4, dataset="misinfo", exclude_models=True, prompting=0):
    logger.info("Distribution in raw data:")
    logger.info(f"ground_truth values: {df['ground_truth'].value_counts().to_dict()}")
    logger.info(f"researched values: {df['researched'].value_counts().to_dict()}")
    logger.info(f"iscontrol values: {df['iscontrol'].value_counts().to_dict()}")

    if prompting == 0:
        df = df[~df['model'].isin(['GPT4o_sycophancy_both', 'GPT4o_persuasion'])].copy()
    elif prompting == 1:
        df = df[df['model'].isin(['GPT4o_sycophancy_both', 'GPT4o_persuasion'])].copy()

    df['time'] = (df['presented'] == 'post').astype(float) - 0.5

    if dataset == "private":
        df['ground_truth_num'] = (df['ground_truth'] == True).astype(float) - 0.5
        df['researched_num'] = (df['researched'] == 'yes').astype(float) - 0.5
        df['istest_num'] = (df['iscontrol'] == 0).astype(float) - 0.5
    else:
        df['ground_truth_num'] = (df['ground_truth'] == True).astype(float) - 0.5
        df['researched_num'] = (df['researched'] == 'yes').astype(float) - 0.5
        df['istest_num'] = (df['iscontrol'] == 'test').astype(float) - 0.5

    if use_subset:
        true_subjects = df[df['ground_truth'] == True]['subject'].unique()
        false_subjects = df[df['ground_truth'] == False]['subject'].unique()
        
        logger.info(f"Total unique subjects: {len(df['subject'].unique())}")
        
        subject_fraction = subset_fraction
        true_sample_size = min(max(int(len(true_subjects) * subject_fraction), 3), len(true_subjects))
        false_sample_size = min(max(int(len(false_subjects) * subject_fraction), 3), len(false_subjects))
        
        sampled_true_subjects = np.random.choice(true_subjects, true_sample_size, replace=False)
        sampled_false_subjects = np.random.choice(false_subjects, false_sample_size, replace=False)
        sampled_subjects = np.concatenate([sampled_true_subjects, sampled_false_subjects])
        df = df[df['subject'].isin(sampled_subjects)]
        
        logger.info(f"Using subset of data: {len(df)} observations from {len(sampled_subjects)} subjects")

    # NOTE: sub_compliance (post-treatment covariate) deliberately omitted here.

    df['model_subject'] = df['model'] + '_' + df['subject'].astype(str)
    subject_ids = {subj: i for i, subj in enumerate(df['model_subject'].unique())}
    df['subject_id'] = df['model_subject'].map(subject_ids)

    logger.info("Recoded variable statistics:")
    for var in ['ground_truth_num', 'researched_num', 'istest_num']:
        unique_vals = np.unique(df[var])
        logger.info(f"{var}: unique values = {unique_vals}, variance = {df[var].var():.6f}")

    main_vars = ['ground_truth', 'researched', 'istest']

    selected_vars = []
    for var in main_vars:
        var_data = df[f"{var}_num"]
        if var_data.var() > 0:
            selected_vars.append(var)
        else:
            logger.warning(f"{var} has no variance in the subset - excluding from model")

    X_dict = {'Intercept': np.ones(len(df)), 'time': df['time']}
    for var in selected_vars:
        X_dict[var] = df[f"{var}_num"]
    # sub_compliance covariate removed (post-treatment variable).

    if not exclude_models:
        if prompting == 0:
            df['model_gpt4o'] = np.where(df['model'] == 'GPT4o', 0.5, np.where(df['model'] == 'claude', -0.25, -0.25))
            df['model_claude'] = np.where(df['model'] == 'claude', 0.5, np.where(df['model'] == 'GPT4o', -0.25, -0.25))
            df['model_mistral'] = np.where(df['model'] == 'mistral', 0.5, np.where(df['model'] == 'GPT4o', -0.25, -0.25))
            X_dict['model_gpt4o'] = df['model_gpt4o']
            X_dict['model_claude'] = df['model_claude']
            X_dict['model_mistral'] = df['model_mistral']
        elif prompting == 1:
            df['model_sycophancy'] = np.where(df['model'] == 'GPT4o_sycophancy_both', 0.5, -0.5)
            df['model_persuasion'] = np.where(df['model'] == 'GPT4o_persuasion', 0.5, -0.5)
            X_dict['model_sycophancy'] = df['model_sycophancy']
            X_dict['model_persuasion'] = df['model_persuasion']
    
    X = pd.DataFrame(X_dict)

    def add_interactions(X, order, vars_to_interact, model_vars=None):
        logger.info(f"Adding {order}-way interactions...")
        for var_combo in combinations(vars_to_interact, order):
            interaction_name = '_'.join(var_combo)
            X[interaction_name] = 1.0
            for var in var_combo:
                X[interaction_name] *= X[var]
            if model_vars:
                for model_var in model_vars:
                    model_interaction_name = f"{interaction_name}_{model_var}"
                    X[model_interaction_name] = X[interaction_name] * X[model_var]
        return X

    if len(selected_vars) >= 2:
        if not exclude_models:
            if prompting == 0:
                model_vars = ['model_gpt4o', 'model_claude', 'model_mistral']
            elif prompting == 1:
                model_vars = ['model_sycophancy', 'model_persuasion']
        else:
            model_vars = []
                
        max_order = min(interaction_level, len(selected_vars))
        for order in range(2, max_order + 1):
            X = add_interactions(X, order, selected_vars, model_vars)
        
        logger.info("Adding time interactions...")
        for var in selected_vars:
            X[f"time_{var}"] = X['time'] * X[var]
        if model_vars:
            for model_var in model_vars:
                X[f"time_{model_var}"] = X['time'] * X[model_var]
        if model_vars:
            for var in selected_vars:
                for model_var in model_vars:
                    X[f"time_{var}_{model_var}"] = X['time'] * X[var] * X[model_var]
        if interaction_level >= 3 and len(selected_vars) >= 2:
            for order in range(2, min(interaction_level, len(selected_vars)) + 1):
                logger.info(f"Adding {order+1}-way time interactions (time x {order} vars)...")
                for var_combo in combinations(selected_vars, order):
                    interaction_name = 'time_' + '_'.join(var_combo)
                    X[interaction_name] = X['time']
                    for var in var_combo:
                        X[interaction_name] *= X[var]
                    if model_vars:
                        for model_var in model_vars:
                            model_interaction_name = f"{interaction_name}_{model_var}"
                            X[model_interaction_name] = X[interaction_name] * X[model_var]
    else:
        logger.warning("Not enough variables with variance to create interactions.")

    logger.info("Checking for numerical issues in design matrix...")
    var_threshold = 1e-10
    low_var_cols = [(col, X[col].var()) for col in X.columns if X[col].var() < var_threshold and col != 'Intercept']
    if low_var_cols:
        logger.warning(f"{len(low_var_cols)} columns with very low variance")
        cols_to_keep = [col for col in X.columns if X[col].var() >= var_threshold or col == 'Intercept']
        X = X[cols_to_keep]
        logger.info(f"Removed {len(low_var_cols)} columns with no variance.")

    try:
        from numpy.linalg import svd
        _, s, _ = svd(X.values, full_matrices=False)
        condition_number = s[0] / s[-1]
        logger.info(f"Condition number of design matrix: {condition_number:.4e}")
    except Exception as e:
        logger.error(f"Could not compute condition number: {e}")

    y = df['response'].values - 1 
    if dataset != "trust":
        y = 6 - (df['response'].values - 1)

    logger.info(f"Response variable stats: Range={np.min(y)}-{np.max(y)}, Mean={np.mean(y):.4f}, Std={np.std(y):.4f}")

    subject_ids_array = df['subject_id'].values
    num_subjects = len(np.unique(subject_ids_array))
    logger.info(f"Number of subjects: {num_subjects}")

    X_array = jnp.array(X.values, dtype=jnp.float32)
    y_array = jnp.array(y.astype(np.int32))
    subject_ids_array = jnp.array(subject_ids_array, dtype=jnp.int32)

    unique_y = np.sort(np.unique(y_array))
    num_cutpoints = len(unique_y) - 1

    logger.info(f"Final design matrix shape: {X_array.shape}")
    logger.info(f"Number of parameters: {X_array.shape[1]}")

    return X_array, y_array, subject_ids_array, num_subjects, num_cutpoints, X.columns.tolist(), unique_y


def prepare_data_extremism(df, subset_fraction=0.01, use_subset=True, interaction_level=4, exclude_models=True, prompting=0):
    logger.info("Distribution in raw data:")
    logger.info(f"ground_truth values: {df['ground_truth'].value_counts().to_dict()}")
    logger.info(f"researched values: {df['researched'].value_counts().to_dict()}")
    logger.info(f"istest values: {df['iscontrol'].value_counts().to_dict()}")

    if prompting == 0:
        df = df[~df['model'].isin(['GPT4o_sycophancy_both', 'GPT4o_persuasion'])].copy()
    elif prompting == 1:
        df = df[df['model'].isin(['GPT4o_sycophancy_both', 'GPT4o_persuasion'])].copy()

    df['ground_truth_num'] = (df['ground_truth'] == True).astype(float) - 0.5
    df['researched_num'] = (df['researched'] == 'yes').astype(float) - 0.5
    df['istest_num'] = (df['iscontrol'] == 0).astype(float) - 0.5

    if use_subset:
        true_subjects = df[df['ground_truth'] == True]['subject'].unique()
        false_subjects = df[df['ground_truth'] == False]['subject'].unique()
        true_sample_size = min(max(int(len(true_subjects) * subset_fraction), 3), len(true_subjects))
        false_sample_size = min(max(int(len(false_subjects) * subset_fraction), 3), len(false_subjects))
        sampled_subjects = np.concatenate([
            np.random.choice(true_subjects, true_sample_size, replace=False),
            np.random.choice(false_subjects, false_sample_size, replace=False)
        ])
        df = df[df['subject'].isin(sampled_subjects)]
        logger.info(f"Using subset: {len(df)} observations from {len(sampled_subjects)} subjects")

    # NOTE: sub_compliance (post-treatment covariate) deliberately omitted here.

    center_point = 4
    df['response_centered'] = df['response'] - center_point

    pivoted_df = df.pivot_table(
        index=['subject', 'model', 'topic', 'ground_truth_num', 'researched_num', 'istest_num', 'searchornot'],
        columns='presented', values='response_centered').reset_index()

    pivoted_df['sign_flip'] = (pivoted_df['pre'] * pivoted_df['post'] < 0).astype(int)

    df = pd.merge(df, pivoted_df[['subject', 'model', 'topic', 'ground_truth_num', 'researched_num', 'istest_num', 'searchornot', 'sign_flip']],
                  on=['subject', 'model', 'topic', 'ground_truth_num', 'researched_num', 'istest_num', 'searchornot'], how='left')

    df = df.loc[df['presented'] == 'post']

    df['model_subject'] = df['model'] + '_' + df['subject'].astype(str)
    subject_ids = {subj: i for i, subj in enumerate(df['model_subject'].unique())}
    df['subject_id'] = df['model_subject'].map(subject_ids)

    main_vars = ['ground_truth', 'researched', 'istest']

    selected_vars = []
    for var in main_vars:
        var_data = df[f"{var}_num"]
        if var_data.var() > 0:
            selected_vars.append(var)
        else:
            logger.warning(f"{var} has no variance - excluding")

    X_dict = {'Intercept': np.ones(len(df))}
    for var in selected_vars:
        X_dict[var] = df[f"{var}_num"]
    # sub_compliance covariate removed (post-treatment variable).

    if not exclude_models: 
        if prompting == 0:
            df['model_gpt4o'] = np.where(df['model'] == 'GPT4o', 0.5, np.where(df['model'] == 'claude', -0.25, -0.25))
            df['model_claude'] = np.where(df['model'] == 'claude', 0.5, np.where(df['model'] == 'GPT4o', -0.25, -0.25))
            df['model_mistral'] = np.where(df['model'] == 'mistral', 0.5, np.where(df['model'] == 'GPT4o', -0.25, -0.25))
            X_dict['model_gpt4o'] = df['model_gpt4o']
            X_dict['model_claude'] = df['model_claude']
            X_dict['model_mistral'] = df['model_mistral']
        elif prompting == 1:
            df['model_sycophancy'] = np.where(df['model'] == 'GPT4o_sycophancy_both', 0.5, -0.5)
            df['model_persuasion'] = np.where(df['model'] == 'GPT4o_persuasion', 0.5, -0.5)
            X_dict['model_sycophancy'] = df['model_sycophancy']
            X_dict['model_persuasion'] = df['model_persuasion']

    X = pd.DataFrame(X_dict)

    def add_interactions(X, order, vars_to_interact, model_vars=None):
        logger.info(f"Adding {order}-way interactions...")
        for var_combo in combinations(vars_to_interact, order):
            interaction_name = '_'.join(var_combo)
            X[interaction_name] = 1.0
            for var in var_combo:
                X[interaction_name] *= X[var]
            if model_vars:
                for model_var in model_vars:
                    model_interaction_name = f"{interaction_name}_{model_var}"
                    X[model_interaction_name] = X[interaction_name] * X[model_var]
        return X

    if not exclude_models:
        if prompting == 0:
            model_vars = ['model_gpt4o', 'model_claude', 'model_mistral']
        elif prompting == 1:
            model_vars = ['model_sycophancy', 'model_persuasion']
    else:
        model_vars = []
        
    if len(selected_vars) >= 2:
        max_order = min(interaction_level, len(selected_vars))
        for order in range(2, max_order + 1):
            X = add_interactions(X, order, selected_vars, model_vars)
    else:
        logger.warning("Not enough variables with variance to create interactions.")

    logger.info("Checking for numerical issues in design matrix...")
    var_threshold = 1e-10
    low_var_cols = [(col, X[col].var()) for col in X.columns if X[col].var() < var_threshold and col != 'Intercept']
    if low_var_cols:
        logger.warning(f"{len(low_var_cols)} columns with very low variance")
        cols_to_keep = [col for col in X.columns if X[col].var() >= var_threshold or col == 'Intercept']
        X = X[cols_to_keep]
        logger.info(f"Removed {len(low_var_cols)} columns.")

    try:
        from numpy.linalg import svd
        _, s, _ = svd(X.values, full_matrices=False)
        logger.info(f"Condition number: {s[0]/s[-1]:.4e}")
    except Exception as e:
        logger.error(f"Could not compute condition number: {e}")

    y = df['sign_flip'].values

    logger.info(f"Response variable stats: Range={np.min(y)}-{np.max(y)}, Mean={np.mean(y):.4f}")

    subject_ids_array = df['subject_id'].values
    num_subjects = len(np.unique(subject_ids_array))
    logger.info(f"Number of subjects: {num_subjects}")

    X_array = jnp.array(X.values, dtype=jnp.float32)
    y_array = jnp.array(y.astype(np.int32))
    subject_ids_array = jnp.array(subject_ids_array, dtype=jnp.int32)

    unique_y = np.sort(np.unique(y_array))
    num_cutpoints = len(unique_y) - 1

    logger.info(f"Final design matrix shape: {X_array.shape}")
    logger.info(f"Number of parameters: {X_array.shape[1]}")

    return X_array, y_array, subject_ids_array, num_subjects, num_cutpoints, X.columns.tolist(), unique_y


def run_inference(model, X, y, subject_ids, num_subjects, num_cutpoints, rng_key, 
                 num_warmup=500, num_samples=500, num_chains=4, 
                 target_accept_prob=0.85, max_tree_depth=8, 
                 progress_bar=True, chain_method='parallel'):
    
    rng_keys = random.split(rng_key, num_chains)
    
    kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        init_strategy=init_to_feasible
    )
    
    mcmc = MCMC(
        kernel, 
        num_warmup=num_warmup, 
        num_samples=num_samples, 
        num_chains=num_chains, 
        progress_bar=progress_bar,
        chain_method=chain_method
    )
    
    mcmc.run(rng_keys, X, y, subject_ids, num_subjects, num_cutpoints)
    
    return mcmc


def summarize_results(mcmc, feature_names, focus_on_interactions=True):
    samples = mcmc.get_samples()
    
    if 'beta' in samples:
        beta_samples = samples['beta']
        num_features = beta_samples.shape[1]
        
        if num_features < len(feature_names):
            feature_names = feature_names[:num_features]
        
        means = np.mean(beta_samples, axis=0)
        hdi_low = np.percentile(beta_samples, 2.5, axis=0)
        hdi_high = np.percentile(beta_samples, 97.5, axis=0)
    
        summary = pd.DataFrame({
            'Parameter': feature_names,
            'Mean': means,
            'HDI_2.5%': hdi_low,
            'HDI_97.5%': hdi_high,
        })
        
        if focus_on_interactions:
            interaction_mask = summary['Parameter'].str.contains('_')
            main_effects = summary[~interaction_mask].copy()
            interactions = summary[interaction_mask].copy()
            summary = pd.concat([main_effects, interactions])
        
        if 'intercept_subject' in samples:
            intercept_subject_mean = np.mean(samples['intercept_subject'])
            intercept_subject_hdi_low = np.percentile(samples['intercept_subject'], 2.5)
            intercept_subject_hdi_high = np.percentile(samples['intercept_subject'], 97.5)
            
            summary = pd.concat([
                summary,
                pd.DataFrame({
                    'Parameter': ['intercept_subject'],
                    'Mean': [intercept_subject_mean],
                    'HDI_2.5%': [intercept_subject_hdi_low],
                    'HDI_97.5%': [intercept_subject_hdi_high],
                })
            ])
        return summary


def plot_forestplot(summary_df, figsize=(10, 16), rope_interval=rope_interval, outcome="output"):
    summary_df = summary_df.sort_values('Mean', ascending=True)
    summary_df = summary_df[summary_df['Parameter'] != 'Intercept']

    xerr = np.abs(summary_df[['HDI_2.5%', 'HDI_97.5%']].T.values - summary_df['Mean'].values)

    plt.figure(figsize=figsize)
    plt.errorbar(
        summary_df['Mean'], 
        range(summary_df.shape[0]), 
        xerr=xerr,
        fmt='o', 
        color='black', 
        markersize=4,
    )
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvspan(rope_interval[0], rope_interval[1], color='gray', alpha=0.3, label='ROPE')
    plt.yticks(range(summary_df.shape[0]), summary_df['Parameter'])
    plt.xlabel('Effect Size')
    plt.savefig(f'../plots/forest_plot_full_GLM_{outcome}.png', dpi=300, bbox_inches='tight')
    plt.close()


def calculate_overlap(rope_interval, hdi_lower, hdi_upper):
    rope_lower, rope_upper = rope_interval
    overlap_lower = max(rope_lower, hdi_lower)
    overlap_upper = min(rope_upper, hdi_upper)
    if overlap_lower > overlap_upper:
        return 0.0
    overlap_length = overlap_upper - overlap_lower
    hdi_length = hdi_upper - hdi_lower
    return overlap_length / hdi_length


def compute_waic(log_likelihood_samples):
    from scipy.special import logsumexp as scipy_logsumexp
    ll = np.array(log_likelihood_samples)
    n_samples, n_obs = ll.shape
    lppd_per_obs = scipy_logsumexp(ll, axis=0) - np.log(n_samples)
    lppd = float(np.sum(lppd_per_obs))
    p_waic_per_obs = np.var(ll, axis=0)
    p_waic = float(np.sum(p_waic_per_obs))
    elpd_waic = lppd - p_waic
    elpd_per_obs = lppd_per_obs - p_waic_per_obs
    se = float(np.sqrt(n_obs * np.var(elpd_per_obs)))
    waic = -2 * elpd_waic
    return {
        'waic': waic, 'elpd_waic': elpd_waic, 'p_waic': p_waic,
        'se': se, 'lppd': lppd, 'elpd_per_obs': elpd_per_obs
    }


def compare_waic(model_dict):
    results = {}
    for name, mcmc in model_dict.items():
        ll = np.array(mcmc.get_samples()['log_likelihood'])
        results[name] = compute_waic(ll)
    
    best_elpd = max(r['elpd_waic'] for r in results.values())
    best_name = [k for k, v in results.items() if v['elpd_waic'] == best_elpd][0]
    best_elpd_per_obs = results[best_name]['elpd_per_obs']
    
    rows = []
    for name, r in results.items():
        elpd_diff = best_elpd - r['elpd_waic']
        if name == best_name:
            dse = 0.0
        else:
            diff_per_obs = best_elpd_per_obs - r['elpd_per_obs']
            n_obs = len(diff_per_obs)
            dse = float(np.sqrt(n_obs * np.var(diff_per_obs)))
        rows.append({
            'name': name, 'elpd_waic': r['elpd_waic'], 'p_waic': r['p_waic'],
            'elpd_diff': elpd_diff, 'se': r['se'], 'dse': dse,
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('elpd_diff').reset_index(drop=True)
    df['rank'] = range(len(df))
    elpd_values = df['elpd_waic'].values
    weights = np.exp(elpd_values - elpd_values.max())
    weights = weights / weights.sum()
    df['weight'] = weights
    df['warning'] = False
    df['scale'] = 'log'
    df = df.set_index('name')
    df = df[['rank', 'elpd_waic', 'p_waic', 'elpd_diff', 'weight', 'se', 'dse', 'warning', 'scale']]
    return df


def ordinal_model(X=None, y=None, subject_ids=None, num_subjects=None, num_cutpoints=None):
    beta_scale = numpyro.sample('beta_scale', dist.HalfNormal(0.1))
    beta_raw = numpyro.sample('beta_raw', dist.Normal(0, 0.1), sample_shape=(X.shape[1],))
    beta = numpyro.deterministic('beta', beta_raw * beta_scale)
    
    sigma_subject_raw = numpyro.sample('sigma_subject_raw', dist.HalfNormal(0.01))
    sigma_subject = numpyro.deterministic('sigma_subject', sigma_subject_raw * 2.0)
    
    with numpyro.plate('subjects', num_subjects):
        subject_offset_raw = numpyro.sample('subject_offset_raw', dist.Normal(0, 0.01))
        subject_intercept = subject_offset_raw * sigma_subject
    
    alpha = 1.0
    concentration = jnp.ones(num_cutpoints + 1) * alpha
    anchor_point = 0.0
    
    with reparam(config={"cutpoints": TransformReparam()}):
        cutpoints = numpyro.sample(
            "cutpoints",
            dist.TransformedDistribution(
                dist.Dirichlet(concentration),
                transforms.SimplexToOrderedTransform(anchor_point),
            ),
        )

    eta = jnp.dot(X, beta) + subject_intercept[subject_ids]
    
    numpyro.sample('y', dist.OrderedLogistic(eta, cutpoints), obs=y)
    if y is not None:
        numpyro.deterministic('log_likelihood', dist.OrderedLogistic(eta, cutpoints).log_prob(y))


def binomial_model(X=None, y=None, subject_ids=None, num_subjects=None, num_cutpoints=None, n_trials=1):
    beta_scale = numpyro.sample('beta_scale', dist.HalfNormal(.1))
    beta_raw = numpyro.sample('beta_raw', dist.Normal(0, .1), sample_shape=(X.shape[1],))
    beta = numpyro.deterministic('beta', beta_raw * beta_scale)
    
    sigma_subject_raw = numpyro.sample('sigma_subject_raw', dist.HalfNormal(.1))
    sigma_subject = numpyro.deterministic('sigma_subject', sigma_subject_raw * 2.0)
    
    with numpyro.plate('subjects', num_subjects):
        subject_offset_raw = numpyro.sample('subject_offset_raw', dist.Normal(0, .1))
        subject_intercept = subject_offset_raw * sigma_subject

    eta = jnp.dot(X, beta) + subject_intercept[subject_ids]
    p = jax.nn.sigmoid(eta)
    
    numpyro.sample('y', dist.Binomial(n_trials, p), obs=y)
    if y is not None:
        numpyro.deterministic('log_likelihood', dist.Binomial(n_trials, p).log_prob(y))


# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == '__main__':
        
    num_samples = 2000
    num_warmup = 2000

    exclude_vars = ['istest',
                    'model_gpt4o', 'model_claude', 'model_mistral',
                    'ground_truth_researched_model_gpt4o', 'ground_truth_researched_model_claude', 'ground_truth_researched_model_mistral',
                    'ground_truth_istest', 'ground_truth_istest_model_gpt4o', 'ground_truth_istest_model_claude', 'ground_truth_istest_model_mistral',
                    'researched_istest', 'researched_istest_model_gpt4o', 'researched_istest_model_claude', 'researched_istest_model_mistral',
                    'ground_truth_researched_istest', 'ground_truth_researched_istest_model_gpt4o', 'ground_truth_researched_istest_model_claude', 'ground_truth_researched_istest_model_mistral',
                    'time_istest', 'time_model_gpt4o', 'time_model_claude', 'time_model_mistral',
                    'time_ground_truth_model_gpt4o', 'time_ground_truth_model_claude', 'time_ground_truth_model_mistral',
                    'time_researched_model_gpt4o', 'time_researched_model_claude', 'time_researched_model_mistral',
                    'time_istest_model_gpt4o', 'time_istest_model_claude', 'time_istest_model_mistral',
                    'time_ground_truth_researched_model_gpt4o', 'time_ground_truth_researched_model_claude', 'time_ground_truth_researched_model_mistral',
                    'time_ground_truth_istest', 'time_ground_truth_istest_model_gpt4o', 'time_ground_truth_istest_model_claude', 'time_ground_truth_istest_model_mistral',
                    'time_researched_istest', 'time_researched_istest_model_gpt4o', 'time_researched_istest_model_claude', 'time_researched_istest_model_mistral',
                    'time_ground_truth_researched_istest', 'time_ground_truth_researched_istest_model_gpt4o', 'time_ground_truth_researched_istest_model_claude', 'time_ground_truth_researched_istest_model_mistral',
                    'model_sycophancy', 'model_persuasion',
                    'ground_truth_researched_model_sycophancy', 'ground_truth_researched_model_persuasion',
                    'ground_truth_istest_model_sycophancy', 'ground_truth_istest_model_persuasion',
                    'researched_istest_model_sycophancy', 'researched_istest_model_persuasion',
                    'ground_truth_researched_istest_model_sycophancy', 'ground_truth_researched_istest_model_persuasion',
                    'time_model_sycophancy', 'time_model_persuasion',
                    'time_ground_truth_model_sycophancy', 'time_ground_truth_model_persuasion',
                    'time_researched_model_sycophancy', 'time_researched_model_persuasion',
                    'time_istest_model_sycophancy', 'time_istest_model_persuasion',
                    'time_ground_truth_researched_model_sycophancy', 'time_ground_truth_researched_model_persuasion',
                    'time_ground_truth_istest_model_sycophancy', 'time_ground_truth_istest_model_persuasion',
                    'time_researched_istest_model_sycophancy', 'time_researched_istest_model_persuasion',
                    'time_ground_truth_researched_istest_model_sycophancy', 'time_ground_truth_researched_istest_model_persuasion']

    dataset = args.dataset
    prompting = args.prompting
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DATASET: {dataset}, PROMPTING: {prompting}")
    logger.info(f"{'='*60}")

    if dataset == "misinfo":    
        data = pd.read_csv('../data/Ztable_misinfo_combined.csv')
    elif dataset == "trust":
        data = pd.read_csv('../data/Ztable_trust_combined.csv')
    elif dataset == "private":
        data = pd.read_csv('../data/Ztable_private_combined.csv')
    elif dataset == "extreme":
        data = pd.read_csv('../data/Ztable_extreme_combined.csv')

    logger.info(f"Data size: {len(data)}")
    logger.info(f"Models: {data['model'].unique()}")

    # ---- Full GLM with and without models ----
    for exclude_models in [0, 1]:

        logger.info(f"\n--- Exclude models: {exclude_models} ---")
        
        clear_gpu_memory()

        if dataset in ["misinfo", "trust", "private"]:
            X, y, subject_ids, num_subjects, num_cutpoints, feature_names, unique_y = prepare_data(
                data, subset_fraction=0.10, use_subset=False, interaction_level=6, 
                dataset=dataset, exclude_models=exclude_models, prompting=prompting)
        elif dataset == "extreme":
            X, y, subject_ids, num_subjects, num_cutpoints, feature_names, unique_y = prepare_data_extremism(
                data, subset_fraction=0.10, use_subset=False, interaction_level=6, 
                exclude_models=exclude_models, prompting=prompting)

        # Design matrix heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(X, cmap='viridis', cbar=False)
        plt.xlabel("Predictors")
        plt.ylabel("Observations")
        plt.title(f"Design Matrix: {dataset} p{prompting} em{exclude_models}")
        plt.savefig(f'../plots/design_matrix_{dataset}_p{prompting}_em{exclude_models}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # VIF
        def calculate_vif(X):
            vif_data = pd.DataFrame()
            vif_data["feature"] = feature_names
            vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
            return vif_data
        vif_scores = calculate_vif(X)
        logger.info(f"VIF scores:\n{vif_scores}")

        # Run inference 
        if dataset in ["misinfo", "trust", "private"]:
            mcmc_full = run_inference(
                model=ordinal_model, X=X, y=y, subject_ids=subject_ids, 
                num_subjects=num_subjects, num_cutpoints=num_cutpoints, 
                rng_key=rng_key, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        elif dataset == "extreme":
            mcmc_full = run_inference(
                model=binomial_model, X=X, y=y, subject_ids=subject_ids, 
                num_subjects=num_subjects, num_cutpoints=None, 
                rng_key=rng_key, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)

        # Summarize
        summary_full_df = summarize_results(mcmc_full, feature_names, focus_on_interactions=True)
        summary_full_df = summary_full_df.sort_index()
        summary_full_df['rope_overlap'] = summary_full_df.apply(
            lambda row: calculate_overlap(rope_interval, row['HDI_2.5%'], row['HDI_97.5%']), axis=1)

        logger.info(f"\n{summary_full_df}")
        mcmc_full.print_summary(exclude_deterministic=False)

        # Save
        if exclude_models == 0:
            summary_full_df.to_csv(f'../parameter_estimates/summary_full_GLM_{dataset}_prompting_{prompting}.csv', index=False)
            plot_forestplot(summary_full_df, figsize=(10, 16), rope_interval=rope_interval, 
                          outcome=f"{dataset}_prompting_{prompting}_full_GLM")
        else:
            summary_full_df.to_csv(f'../parameter_estimates/summary_full_GLM_{dataset}_prompting_{prompting}_no_model.csv', index=False)
            plot_forestplot(summary_full_df, figsize=(6, 8), rope_interval=rope_interval, 
                          outcome=f"{dataset}_prompting_{prompting}_full_GLM_no_model")

        # Traceplots
        try:
            if dataset == "private":
                az.plot_trace(az.from_numpyro(mcmc_full), var_names=["cutpoints"])
            elif dataset in ["misinfo", "trust"]:
                az.plot_trace(az.from_numpyro(mcmc_full), var_names=["sigma_subject", "beta_scale", "cutpoints"])
            else:
                az.plot_trace(az.from_numpyro(mcmc_full), var_names=["sigma_subject", "beta_scale"])
            plt.savefig(f'../plots/traceplot_{dataset}_p{prompting}_em{exclude_models}.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Traceplot failed: {e}")

        if exclude_models == 0:
            mcmc_full_models = mcmc_full
        else:
            mcmc_full_no_models = mcmc_full
        del mcmc_full
        clear_gpu_memory()

    # ---- Control GLM ----
    logger.info(f"\n--- Control GLM: {dataset}, Prompting: {prompting} ---")
    
    clear_gpu_memory()

    if dataset in ["misinfo", "trust", "private"]:
        X, y, subject_ids, num_subjects, num_cutpoints, feature_names, unique_y = prepare_data(
            data, subset_fraction=0.10, use_subset=False, interaction_level=6, 
            dataset=dataset, exclude_models=exclude_models, prompting=prompting)
    elif dataset == "extreme":
        X, y, subject_ids, num_subjects, num_cutpoints, feature_names, unique_y = prepare_data_extremism(
            data, subset_fraction=0.10, use_subset=False, interaction_level=6, 
            exclude_models=exclude_models, prompting=prompting)

    # Filter columns
    exclude_vars_filtered = [var for var in exclude_vars if var in feature_names]
    columns_to_keep = [var not in exclude_vars_filtered for var in feature_names]
    X_filtered = X[:, columns_to_keep]
    feature_names_filtered = [var for var in feature_names if var not in exclude_vars_filtered]

    logger.info(f"Original X shape: {X.shape}, Filtered X shape: {X_filtered.shape}")
    logger.info(f"Filtered feature names: {feature_names_filtered}")

    # Design matrix heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(X_filtered, cmap='viridis', cbar=False)
    plt.xlabel("Predictors")
    plt.ylabel("Observations")
    plt.title(f"Design Matrix Control: {dataset} p{prompting}")
    plt.savefig(f'../plots/design_matrix_control_{dataset}_p{prompting}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Run inference 
    if dataset in ["misinfo", "trust", "private"]:
        mcmc_control = run_inference(
            model=ordinal_model, X=X_filtered, y=y, subject_ids=subject_ids, 
            num_subjects=num_subjects, num_cutpoints=num_cutpoints, 
            rng_key=rng_key, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    elif dataset == "extreme":
        mcmc_control = run_inference(
            model=binomial_model, X=X_filtered, y=y, subject_ids=subject_ids, 
            num_subjects=num_subjects, num_cutpoints=None, 
            rng_key=rng_key, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)

    # Summarize
    summary_control_df = summarize_results(mcmc_control, feature_names_filtered, focus_on_interactions=True)
    summary_control_df = summary_control_df.sort_index()
    summary_control_df['rope_overlap'] = summary_control_df.apply(
        lambda row: calculate_overlap(rope_interval, row['HDI_2.5%'], row['HDI_97.5%']), axis=1)

    logger.info(f"\n{summary_control_df}")
    mcmc_control.print_summary(exclude_deterministic=False)

    summary_control_df.to_csv(f'../parameter_estimates/summary_control_GLM_{dataset}_prompting_{prompting}.csv', index=False)
    plot_forestplot(summary_control_df, figsize=(6, 8), rope_interval=rope_interval, 
                    outcome=f"{dataset}_prompting_{prompting}_control_GLM")

    # Traceplots
    try:
        if dataset == "private":
            az.plot_trace(az.from_numpyro(mcmc_control), var_names=["cutpoints"])
        elif dataset in ["misinfo", "trust"]:
            az.plot_trace(az.from_numpyro(mcmc_control), var_names=["sigma_subject", "beta_scale", "cutpoints"])
        else:
            az.plot_trace(az.from_numpyro(mcmc_control), var_names=["sigma_subject", "beta_scale"])
        plt.savefig(f'../plots/traceplot_control_{dataset}_p{prompting}.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Traceplot failed: {e}")

    # ---- WAIC comparison ----
    clear_gpu_memory()
    try:
        comparison = compare_waic({
            "GLM_full_with_model": mcmc_full_models,
            "GLM_full_without_model": mcmc_full_no_models,
            "GLM_control": mcmc_control
        })
        logger.info(f"\nWAIC Comparison:\n{comparison}")
        comparison.to_csv(f'../parameter_estimates/waic_comparison_{dataset}_prompting_{prompting}.csv')
    except Exception as e:
        logger.error(f"WAIC comparison failed: {e}")

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {dataset}, prompting={prompting}")
    logger.info(f"{'='*60}")