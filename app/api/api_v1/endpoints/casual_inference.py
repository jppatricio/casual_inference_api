from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from app.pipeline.causal_inference_pipeline import CausalInferenceInput, CausalInferenceOutput, pipeline

router = APIRouter()

@router.get("/")
def health_check():
    return {"status": "OK"}

class DataInput(BaseModel):
    data: dict
    treatment: str
    outcome: str
    model: str

@router.post("/infer", response_model=CausalInferenceOutput)
async def perform_causal_inference(input_data: CausalInferenceInput):
    try:
        result = pipeline.run(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models")
async def get_available_models():
    return list(pipeline.models.keys())

@router.post("/dowhy", response_model=CausalInferenceOutput)
async def run_dowhy(
    data: DataInput,
    graph: Optional[str] = None,
    target_units: str = Query("ate", description="Target units for estimation"),
    methods: List[str] = Query(["backdoor.linear_regression"], description="Estimation methods to use")
):
    input_data = CausalInferenceInput(
        data=data.data,
        treatment=data.treatment,
        outcome=data.outcome,
        model="dowhy",
        additional_params={
            "graph": graph,
            "target_units": target_units,
            "methods": methods
        }
    )
    return await perform_causal_inference(input_data)

@router.post("/synthetic_control", response_model=CausalInferenceOutput)
async def run_synthetic_control(
    data: DataInput,
    actual: str = Query(..., description="Name of the actual outcome variable"),
    intercept: str = Query("1", description="Intercept term in the formula"),
    treatment_time: int = Query(..., description="Time point when the treatment was applied"),
    covariates: List[str] = Query([], description="List of covariate variable names")
):
    input_data = CausalInferenceInput(
        data=data.data,
        treatment=data.treatment,
        outcome=data.outcome,
        model="synthetic_control",
        additional_params={
            "actual": actual,
            "intercept": intercept,
            "treatment_time": treatment_time,
            "covariates": covariates
        }
    )
    return await perform_causal_inference(input_data)

@router.post("/ancova", response_model=CausalInferenceOutput)
async def run_ancova(
    data: DataInput,
    pretreatment_variable: str = Query(..., description="Name of the pretreatment variable"),
    group_variable: Optional[str] = Query(None, description="Name of the group variable (defaults to treatment if not provided)")
):
    input_data = CausalInferenceInput(
        data=data.data,
        treatment=data.treatment,
        outcome=data.outcome,
        model="ancova",
        additional_params={
            "pretreatment_variable": pretreatment_variable,
            "group_variable": group_variable or data.treatment
        }
    )
    return await perform_causal_inference(input_data)

@router.post("/regression_discontinuity", response_model=CausalInferenceOutput)
async def run_regression_discontinuity(
    data: DataInput,
    running_variable: str = Query(..., description="Name of the running variable"),
    treatment_threshold: Optional[float] = Query(None, description="Threshold for treatment assignment"),
    bandwidth: Optional[float] = Query(None, description="Bandwidth for local regression"),
    use_splines: bool = Query(False, description="Whether to use splines in the formula"),
    spline_df: int = Query(6, description="Degrees of freedom for the spline, if used"),
    epsilon: float = Query(0.01, description="Small value to avoid division by zero")
):
    input_data = CausalInferenceInput(
        data=data.data,
        treatment=data.treatment,
        outcome=data.outcome,
        model="regression_discontinuity",
        additional_params={
            "running_variable": running_variable,
            "treatment_threshold": treatment_threshold,
            "bandwidth": bandwidth,
            "use_splines": use_splines,
            "spline_df": spline_df,
            "epsilon": epsilon
        }
    )
    return await perform_causal_inference(input_data)

@router.post("/difference_in_differences", response_model=CausalInferenceOutput)
async def run_difference_in_differences(
    data: DataInput,
    time_variable: str = Query(..., description="Name of the time variable"),
    group_variable: Optional[str] = Query(None, description="Name of the group variable (defaults to treatment if not provided)")
):
    input_data = CausalInferenceInput(
        data=data.data,
        treatment=data.treatment,
        outcome=data.outcome,
        model="difference_in_differences",
        additional_params={
            "time_variable": time_variable,
            "group_variable": group_variable or data.treatment
        }
    )
    return await perform_causal_inference(input_data)

@router.post("/instrumental_variable", response_model=CausalInferenceOutput)
async def run_instrumental_variable(
    data: DataInput,
    instruments: List[str] = Query(..., description="List of instrument variable names"),
    sample_kwargs: Optional[dict] = None
):
    input_data = CausalInferenceInput(
        data=data.data,
        treatment=data.treatment,
        outcome=data.outcome,
        model="instrumental_variable",
        additional_params={
            "instruments": instruments,
            "sample_kwargs": sample_kwargs
        }
    )
    return await perform_causal_inference(input_data)

@router.post("/semopy", response_model=CausalInferenceOutput)
async def run_semopy(
    data: DataInput,
    model_spec: str = Query(..., description="Model specification for Semopy in string format")
):
    input_data = CausalInferenceInput(
        data=data.data,
        treatment=data.treatment,
        outcome=data.outcome,
        model="semopy",
        additional_params={
            "model_spec": model_spec
        }
    )
    return await perform_causal_inference(input_data)