import pandas as pd
import numpy as np

def format_submission(predictions, output_path):
    """
    Format predictions into proper submission format
    
    Args:
        predictions: List of predicted labels
        output_path: Path to save submission CSV
    """
    submission = pd.DataFrame({
        "Id": range(len(predictions)),
        "Prediction": predictions
    })
    
    # Validate submission format
    assert submission.shape[1] == 2, "Submission must have exactly 2 columns"
    assert submission.shape[0] == len(predictions), "Must have one row per prediction"
    assert all(submission.columns == ["Id", "Prediction"]), "Columns must be Id and Prediction"
    
    # Save submission
    submission.to_csv(output_path, index=False)
    return submission
