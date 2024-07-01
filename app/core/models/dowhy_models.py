from dowhy import CausalModel
import pandas as pd
from typing import Optional, List

def run_causal_analysis(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    graph: Optional[object] = None,
    target_units: str = 'ate',
    methods: List[str] = ["backdoor.linear_regression"]
):
    """
    Run a causal analysis using the DoWhy library.
    
    Args:
        data (pd.DataFrame): The input data.
        treatment (str): The name of the treatment variable.
        outcome (str): The name of the outcome variable.
        graph (object, optional): The causal graph, if available.
        target_units (str): The target units for estimation (default: 'ate').
        methods (List[str]): List of estimation methods to use.
    
    Returns:
        tuple: A tuple containing the identified estimand, estimates, and refutations.
        None: If an error is produced during the analysis.
    """
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=graph,
    )
    model.view_model()
    
    try:
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        print(identified_estimand)
    except Exception as e:
        print(f"No causal effect identified: {str(e)}")
        return None
    
    estimates = []
    refutes = []

    for method in methods:
        try:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name=method,
                target_units=target_units,
                confidence_intervals=True,
                method_params={"num_null_simulations": 100}
            )
            estimates.append(estimate)

            refute_results = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="random_common_cause"
            )
            refutes.append(refute_results)
        except Exception as e:
            print(f"Error with method {method}: {str(e)}")

    return identified_estimand, estimates, refutes