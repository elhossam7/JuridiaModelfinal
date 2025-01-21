import pandas as pd

def validate_submission(submission_path):
    """
    Validates a submission file meets all requirements
    
    Args:
        submission_path: Path to submission CSV file
    Returns:
        bool: True if valid, raises exception if invalid
    """
    submission = pd.read_csv(submission_path)
    
    # Check basic format
    if submission.shape[1] != 2:
        raise ValueError(f"Submission has {submission.shape[1]} columns, expected 2")
        
    if not all(submission.columns == ["Id", "Prediction"]):
        raise ValueError("Columns must be exactly 'Id' and 'Prediction'")
    
    # Check predictions
    valid_classes = {"Dispositions", "Convention", "Loi", "Dahir"}
    invalid_preds = set(submission["Prediction"]) - valid_classes
    if invalid_preds:
        raise ValueError(f"Invalid prediction classes found: {invalid_preds}")
        
    return True
