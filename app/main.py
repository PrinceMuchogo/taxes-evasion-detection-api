from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import joblib
import numpy as np
import shap
import pandas as pd
import os

app = FastAPI(title="Window Dressing Detection API with SHAP")

# --- Load Required Artifacts ---
MODEL_PATH = "model/window_dressing_model.pkl"
SCALER_PATH = "model/scaler.pkl"
FEATURE_COLUMNS_PATH = "model/feature_columns.pkl"

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURE_COLUMNS_PATH)):
    raise RuntimeError("Required model artifacts are missing.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)  # List of 35 feature names

# Initialize SHAP Explainer
explainer = shap.TreeExplainer(model, feature_names=feature_columns)

# --- Data schema ---
# This model will accept an optional field "data"
# and also allow extra fields (in case a flat dictionary is sent).
class FinancialInput(BaseModel):
    data: Optional[List[List[float]]] = None

    class Config:
        extra = "allow"  # Allow additional keys if "data" is not provided

@app.post("/predict")
def predict(input_payload: FinancialInput):
    try:
        # If "data" is provided, use it directly.
        if input_payload.data is not None:
            input_data = np.array(input_payload.data)
        else:
            # Otherwise, assume the payload is a flat dictionary.
            # Exclude the key "data" if present (should be None anyway).
            flat_dict: Dict[str, Any] = {k: v for k, v in input_payload.dict().items() if k != "data"}
            
            # Ensure that all required feature keys are present
            missing_features = [feat for feat in feature_columns if feat not in flat_dict]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Create a list of values in the proper order
            sample_list = [float(flat_dict[feat]) for feat in feature_columns]
            input_data = np.array([sample_list])
        
        if input_data.ndim != 2 or input_data.shape[1] != len(feature_columns):
            raise ValueError(
                f"Expected input shape (n_samples, {len(feature_columns)}), but got {input_data.shape}"
            )

        # Convert to DataFrame with correct columns
        df_input = pd.DataFrame(input_data, columns=feature_columns)

        # Scale features
        input_scaled = scaler.transform(df_input)

        # Predict
        preds = model.predict(input_scaled)

        # Explain
        shap_values = explainer(input_scaled)
        explanations = []
        for i, row_vals in enumerate(shap_values.values):
            top_features = sorted(
                zip(shap_values.feature_names, row_vals),
                key=lambda x: abs(x[1]), reverse=True
            )[:5]
            explanations.append({
                "prediction": int(preds[i]),
                "top_features": [
                    {"feature": name, "impact": float(val)} for name, val in top_features
                ]
            })

        return {"results": explanations}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
