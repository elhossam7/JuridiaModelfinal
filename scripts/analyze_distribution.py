import pandas as pd
import matplotlib.pyplot as plt

def analyze_class_distribution(data_path):
    """
    Analyzes and plots class distribution in dataset
    
    Args:
        data_path: Path to data CSV file
    Returns:
        dict: Class distribution statistics
    """
    data = pd.read_csv(data_path)
    
    # Get distribution
    dist = data["doc_type"].value_counts()
    
    # Plot distribution
    plt.figure(figsize=(10,6))
    dist.plot(kind='bar')
    plt.title('Class Distribution')
    plt.ylabel('Count')
    plt.xlabel('Document Type')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    
    # Calculate statistics
    total = len(data)
    stats = {
        "distribution": dist.to_dict(),
        "percentages": (dist/total * 100).to_dict(),
        "total_samples": total
    }
    
    return stats
