from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import logging

# Import the functions from the new file locations
from app.core.utils.general_utils import create_dot_graph, parse_knowledge_file, get_dataset_for_casualpy
from app.core.models.casualpy_models import (
    run_casualpy_syntheticcontrol, run_causalpy_ancova, 
    run_casualpy_regressiondiscontinuity, run_casualpy_differenceindifferences,
    run_causalpy_iv
)
from app.core.models.dowhy_models import run_causal_analysis
from app.core.models.semopy_models import run_semopy_analysis

class CausalInferenceInput(BaseModel):
    data: Dict[str, List[Any]]
    treatment: str
    outcome: str
    model: str = Field(..., description="Model to use: 'dowhy', 'synthetic_control', 'ancova', 'regression_discontinuity', 'difference_in_differences', 'instrumental_variable', 'semopy")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters specific to each model")

class CausalInferenceOutput(BaseModel):
    effect: Optional[float]
    confidence_interval: Optional[tuple]
    p_value: Optional[float]
    selected_model_used: str
    execution_time: float
    plot: Optional[str]
    additional_info: Dict[str, Any] = Field(default_factory=dict)

class CausalInferencePipeline:
    def __init__(self):
        self.models = {
            "dowhy": self._run_dowhy,
            "synthetic_control": self._run_synthetic_control,
            "ancova": self._run_ancova,
            "regression_discontinuity": self._run_regression_discontinuity,
            "difference_in_differences": self._run_difference_in_differences,
            "instrumental_variable": self._run_instrumental_variable,
            "semopy": self._run_semopy
        }

    def run(self, input_data: CausalInferenceInput) -> CausalInferenceOutput:
        import time
        start_time = time.time()
        print("Causal Inference Pipeline")

        # 1. Data Ingestion
        df = pd.DataFrame(input_data.data)
        # 2. Data Preprocessing
        df_processed = self._preprocess_data(df)
        # 3. Causal Model Selection
        model_func = self.models.get(input_data.model)
        if model_func is None:
            raise ValueError(f"Unknown model: {input_data.model}")
        # 4. Causal Inference Execution
        result = model_func(df_processed, input_data.treatment, input_data.outcome, input_data.additional_params)
        # 5. Result Interpretation
        interpreted_result = self._interpret_result(result, input_data.model)
        execution_time = time.time() - start_time

        logging.info(f"Pipeline finished in {execution_time} seconds")

        # 6. API Response Formation
        return CausalInferenceOutput(
            effect=interpreted_result.get('effect'),
            confidence_interval=interpreted_result.get('confidence_interval'),
            p_value=interpreted_result.get('p_value'),
            selected_model_used=input_data.model,
            execution_time=execution_time,
            plot=interpreted_result.get('plot'),
            additional_info=interpreted_result.get('additional_info', {})
        )

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Handle missing values
        df = df.dropna()
        
        # Convert categorical variables to numeric
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = pd.Categorical(df[column]).codes
        
        return df

    def _run_dowhy(self, df: pd.DataFrame, treatment: str, outcome: str, params: Dict[str, Any]) -> Dict[str, Any]:
        graph = params.get('graph')
        target_units = params.get('target_units', 'ate')
        methods = params.get('methods', ["backdoor.linear_regression"])

        identified_estimand, estimates, refutes = run_causal_analysis(df, outcome, graph, target_units, methods)

        if estimates:
            effect = estimates[0].value
            ci_low, ci_high = estimates[0].get_confidence_intervals()[0]
            p_value = estimates[0].get_significance_test_results()[0]['p_value']
        else:
            effect = ci_low = ci_high = p_value = None

        return {
            'effect': effect,
            'confidence_interval': (ci_low, ci_high),
            'p_value': p_value,
            'additional_info': {
                'identified_estimand': str(identified_estimand),
                'estimates': [str(est) for est in estimates],
                'refutations': [str(ref) for ref in refutes]
            }
        }

    def _run_synthetic_control(self, df: pd.DataFrame, treatment: str, outcome: str, params: Dict[str, Any]) -> Dict[str, Any]:
        actual = params.get('actual', outcome)
        intercept = params.get('intercept', '1')
        treatment_time = params.get('treatment_time')
        covariates = params.get('covariates', [])

        result, fig = run_casualpy_syntheticcontrol(df, actual, intercept, treatment_time, covariates)

        plot = self._fig_to_base64(fig)

        return {
            'effect': result.treatment_effect,
            'plot': plot,
            'additional_info': {
                'model_summary': str(result.summary())
            }
        }

    def _run_ancova(self, df: pd.DataFrame, treatment: str, outcome: str, params: Dict[str, Any]) -> Dict[str, Any]:
        pretreatment_variable = params.get('pretreatment_variable')
        group_variable = params.get('group_variable', treatment)

        result, fig = run_causalpy_ancova(df, outcome, pretreatment_variable, group_variable)

        plot = self._fig_to_base64(fig)

        return {
            'effect': result.treatment_effect,
            'plot': plot,
            'additional_info': {
                'model_summary': str(result.summary())
            }
        }

    def _run_regression_discontinuity(self, df: pd.DataFrame, treatment: str, outcome: str, params: Dict[str, Any]) -> Dict[str, Any]:
        running_variable = params.get('running_variable')
        treatment_threshold = params.get('treatment_threshold')
        bandwidth = params.get('bandwidth')
        use_splines = params.get('use_splines', False)
        spline_df = params.get('spline_df', 6)
        epsilon = params.get('epsilon', 0.01)

        result, fig = run_casualpy_regressiondiscontinuity(
            df, outcome, running_variable, treatment_threshold, bandwidth, use_splines, spline_df, epsilon
        )

        plot = self._fig_to_base64(fig)

        return {
            'effect': result.treatment_effect,
            'plot': plot,
            'additional_info': {
                'model_summary': str(result.summary())
            }
        }

    def _run_difference_in_differences(self, df: pd.DataFrame, treatment: str, outcome: str, params: Dict[str, Any]) -> Dict[str, Any]:
        time_variable = params.get('time_variable', 'post_treatment')
        group_variable = params.get('group_variable', treatment)

        result = run_casualpy_differenceindifferences(df, outcome, time_variable, group_variable)

        if 'error' in result:
            logging.error(f"Error in Difference-in-Differences analysis: {result['error']}")
            return result
        else:
            return {
                'effect': float(result['effect']) if result['effect'] is not None else None,
                'confidence_interval': result['confidence_interval'],
                'p_value': result['p_value'],
                'selected_model_used': "difference_in_differences",
                'additional_info': {
                    'coefficients': result['coefficients'],
                    'model_summary': result['summary'],
                    'diagnostics': result['diagnostics']
                }
            }

    def _run_instrumental_variable(self, df: pd.DataFrame, treatment: str, outcome: str, params: Dict[str, Any]) -> Dict[str, Any]:
        instruments = params.get('instruments', [])
        sample_kwargs = params.get('sample_kwargs')

        result = run_causalpy_iv(df, outcome, treatment, instruments, sample_kwargs)

        return {
            'effect': result.treatment_effect,
            'additional_info': {
                'model_summary': str(result.summary())
            }
        }

    def _interpret_result(self, result: Dict[str, Any], model: str) -> Dict[str, Any]:
        # This method can be expanded to provide more sophisticated interpretation
        return result

    def _fig_to_base64(self, fig):
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def _run_semopy(self, df: pd.DataFrame, treatment: str, outcome: str, params: Dict[str, Any]) -> Dict[str, Any]:
        model_spec = params.get('model_spec')
        if not model_spec:
            raise ValueError("Model specification is required for Semopy analysis")

        result = run_semopy_analysis(df, model_spec)

        return {
            'effect': None,  # Semopy doesn't provide a single "effect" value
            'plot': result['plot'],
            'additional_info': {
                'summary': result['summary'],
                'fit_measures': result['fit_measures']
            }
        }

pipeline = CausalInferencePipeline()