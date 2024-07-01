import causalpy as cp
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import xarray as xr
import re
import logging

def run_casualpy_syntheticcontrol(data: pd.DataFrame, actual: str, intercept: str, treatment_time: int, covariates: List[str]):
    """
    Run a Synthetic Control analysis using CausalPy.
    
    Args:
        data (pd.DataFrame): The input data.
        actual (str): The name of the actual outcome variable.
        intercept (str): The intercept term in the formula.
        treatment_time (int): The time point when the treatment was applied.
        covariates (List[str]): List of covariate variable names.
    
    Returns:
        tuple: A tuple containing the result object and the plot figure.
    """
    formula = f"{actual} ~ {intercept}" if not covariates else f"{actual} ~ {intercept} + {' + '.join(covariates)}"
    
    result = cp.pymc_experiments.SyntheticControl(
        data,
        treatment_time,
        formula=formula,
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs={"target_accept": 0.95}),
    )

    fig, _ = result.plot()
    return result, fig

def run_causalpy_ancova(data: pd.DataFrame, outcome: str, pretreatment_variable_name: str, group_variable_name: str):
    """
    Run an ANCOVA analysis using CausalPy.
    
    Args:
        data (pd.DataFrame): The input data.
        outcome (str): The name of the outcome variable.
        pretreatment_variable_name (str): The name of the pretreatment variable.
        group_variable_name (str): The name of the group variable.
    
    Returns:
        tuple: A tuple containing the result object and the plot figure.
    """
    formula = f"{outcome} ~ 1 + C({group_variable_name}) + {pretreatment_variable_name}"

    result = cp.pymc_experiments.PrePostNEGD(
        data,
        formula,
        group_variable_name,
        pretreatment_variable_name,
        model=cp.pymc_models.LinearRegression(),
    )

    fig, _ = result.plot()
    return result, fig

def run_casualpy_regressiondiscontinuity(
    data: pd.DataFrame,
    outcome: str,
    running_variable_name: str,
    treatment_threshold: float = None,
    bandwidth: float = None,
    use_splines: bool = False,
    spline_df: int = 6,
    epsilon: float = 0.01
):
    """
    Run a Regression Discontinuity analysis using CausalPy.
    
    Args:
        data (pd.DataFrame): The input data.
        outcome (str): The name of the outcome variable.
        running_variable_name (str): The name of the running variable.
        treatment_threshold (float, optional): The threshold for treatment assignment.
        bandwidth (float, optional): The bandwidth for local regression.
        use_splines (bool): Whether to use splines in the formula.
        spline_df (int): Degrees of freedom for the spline, if used.
        epsilon (float): Small value to avoid division by zero.
    
    Returns:
        tuple: A tuple containing the result object and the plot figure.
    """
    if treatment_threshold is None:
        treatment_threshold = data[running_variable_name].median()

    if bandwidth is None:
        bandwidth = np.inf

    formula = (
        f"{outcome} ~ 1 + bs({running_variable_name}, df={spline_df}) + {running_variable_name}"
        if use_splines
        else f"{outcome} ~ 1 + {running_variable_name} + {running_variable_name}"
    )

    rd_result = cp.pymc_experiments.RegressionDiscontinuity(
        data,
        formula,
        running_variable_name=running_variable_name,
        model=cp.pymc_models.LinearRegression(),
        treatment_threshold=treatment_threshold,
        bandwidth=bandwidth,
        epsilon=epsilon,
    )

    rd_fig, _ = rd_result.plot()
    return rd_result, rd_fig

def run_casualpy_differenceindifferences(
    data: pd.DataFrame,
    outcome: str,
    time_variable_name: str,
    group_variable_name: str
):
    """
    Run a Difference-in-Differences analysis using CausalPy.
    
    Args:
        data (pd.DataFrame): The input data.
        outcome (str): The name of the outcome variable.
        time_variable_name (str): The name of the time variable.
        group_variable_name (str): The name of the group variable.
    
    Returns:
        tuple: A tuple containing the result object and the plot figure.
    """
    logging.info("Starting Difference-in-Differences analysis")
    
    # Ensure the group variable is boolean
    data[group_variable_name] = data[group_variable_name].astype(bool)
    
    # Ensure the time variable is named 'post_treatment' and is boolean
    data['post_treatment'] = data[time_variable_name].astype(bool)

    # The formula should use the group and post_treatment variables
    formula = f"{outcome} ~ 1 + {group_variable_name} * post_treatment"

    logging.info(f"Formula: {formula}")
    logging.info(f"Data shape: {data.shape}")
    logging.info(f"Data columns: {data.columns}")
    logging.info(f"Data types:\n{data.dtypes}")
    logging.info(f"Data head:\n{data.head()}")

    try:
        did_result = cp.pymc_experiments.DifferenceInDifferences(
            data,
            formula=formula,
            time_variable_name='post_treatment',
            group_variable_name=group_variable_name,
            model=cp.pymc_models.LinearRegression(
                sample_kwargs={
                    "target_accept": 0.98,
                    "progressbar": True,
                    "nuts": {"max_treedepth": 12}
                }
            )
        )
        
        logging.info("DifferenceInDifferences model created successfully")
        
        # Attempt to get summary and log the result
        try:
            summary = did_result.summary()
            logging.info(f"Model summary:\n{summary}")
        except Exception as e:
            logging.error(f"Error getting model summary: {str(e)}")
            summary = "Error getting model summary"
        
        causal_impact = did_result.causal_impact
        logging.info(f"Causal impact type: {type(causal_impact)}")
        logging.info(f"Causal impact: {causal_impact}")
        
        # Extract causal impact value and confidence interval
        if isinstance(causal_impact, xr.DataArray):
            causal_impact_value = float(causal_impact.mean().values)
            ci_low, ci_high = np.percentile(causal_impact.values, [2.5, 97.5])
        elif isinstance(causal_impact, (float, int)):
            causal_impact_value = float(causal_impact)
            ci_low, ci_high = None, None
        else:
            causal_impact_value = None
            ci_low, ci_high = None, None
            logging.warning(f"Unexpected type for causal_impact: {type(causal_impact)}")
        
        logging.info(f"Causal impact value: {causal_impact_value}")
        logging.info(f"Confidence Interval: ({ci_low}, {ci_high})")
        
        # Extract coefficients from the summary string
        coefficients = {}
        summary_str = str(summary)
        coef_pattern = r'(\w+(?:\[T\.True\])?(?::\w+(?:\[T\.True\])?)?)(?:\s+)([-]?\d+\.\d+)(?:\s+)([-]?\d+\.\d+)(?:\s+)([-]?\d+\.\d+)'
        for match in re.finditer(coef_pattern, summary_str):
            coefficients[match.group(1)] = {
                'mean': float(match.group(2)),
                'sd': float(match.group(3)),
                'mc_error': float(match.group(4))
            }
        
        # Extract diagnostic information
        diagnostics = {
            "max_treedepth_warning": "max_treedepth" in summary_str,
            "rhat_warning": "rhat statistic is larger than 1.01" in summary_str,
            "ess_warning": "effective sample size per chain is smaller than 100" in summary_str
        }
        
        # Attempt to get p-value
        try:
            p_value = did_result.p_value
            logging.info(f"P-value: {p_value}")
        except AttributeError:
            p_value = None
            logging.warning("Unable to extract p-value")
        
        return {
            'effect': causal_impact_value,
            'confidence_interval': (ci_low, ci_high) if ci_low is not None and ci_high is not None else None,
            'p_value': p_value,
            'coefficients': coefficients,
            'summary': summary_str,
            'diagnostics': diagnostics
        }
    except Exception as e:
        logging.error(f"Error in DifferenceInDifferences: {str(e)}", exc_info=True)
        return {
            'error': str(e),
            'data_info': {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': str(data.dtypes),
                'head': data.head().to_dict()
            }
        }

def run_causalpy_iv(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    sample_kwargs: Dict[str, Any] = None
):
    """
    Run an Instrumental Variable analysis using CausalPy.
    
    Args:
        data (pd.DataFrame): The input data.
        outcome (str): The name of the outcome variable.
        treatment (str): The name of the treatment variable.
        instruments (List[str]): List of instrument variable names.
        sample_kwargs (Dict[str, Any], optional): Additional sampling parameters.
    
    Returns:
        object: The result object from the IV analysis.
    """
    formula = f"{outcome} ~ 1 + {treatment}"
    instruments_formula = f"{treatment} ~ 1 + {' + '.join(instruments)}"

    iv_result = cp.pymc_experiments.InstrumentalVariable(
        instruments_data=data[[treatment, *instruments]],
        data=data[[outcome, treatment]],
        instruments_formula=instruments_formula,
        formula=formula,
        model=cp.pymc_models.InstrumentalVariableRegression(sample_kwargs=sample_kwargs),
    )

    return iv_result