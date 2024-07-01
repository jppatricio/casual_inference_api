import semopy
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def run_semopy_analysis(data: pd.DataFrame, model_spec: str):
    """
    Run a Structural Equation Model analysis using Semopy.
    
    Args:
        data (pd.DataFrame): The input data.
        model_spec (str): The model specification in Semopy format.
    
    Returns:
        dict: A dictionary containing the analysis results and plots.
    """
    # Create and fit the model
    model = semopy.Model(model_spec)
    res = model.fit(data)
    
    # Get the model summary
    summary = model.inspect()
    
    # Generate path diagram
    fig, ax = plt.subplots(figsize=(12, 8))
    semopy.semplot(model, ax=ax)
    
    # Convert plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return {
        'summary': summary.to_dict(),
        'fit_measures': res,
        'plot': plot
    }