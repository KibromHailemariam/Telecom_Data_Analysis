import pandas as pd
import numpy as np
from user_overview_analysis import UserOverviewAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load the telecom XDR data efficiently."""
    try:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'xdr_data.parquet')
        
        # Read only required columns
        required_cols = [
            'MSISDN/Number', 'Handset Type', 'Handset Manufacturer', 'Bearer Id',
            'Dur. (ms)', 'HTTP DL (Bytes)', 'HTTP UL (Bytes)',
            'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
            'Google DL (Bytes)', 'Google UL (Bytes)',
            'Email DL (Bytes)', 'Email UL (Bytes)',
            'YouTube DL (Bytes)', 'YouTube UL (Bytes)',
            'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
            'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
            'Other DL (Bytes)', 'Other UL (Bytes)'
        ]
        
        # Use memory-efficient dtypes
        dtype_dict = {
            'MSISDN/Number': 'category',
            'Handset Type': 'category',
            'Handset Manufacturer': 'category',
            'Bearer Id': 'int32'
        }
        
        df = pd.read_parquet(
            data_path,
            columns=required_cols,
            engine='fastparquet'  # Use fastparquet engine for better performance
        )
        
        # Convert categorical columns
        for col in ['MSISDN/Number', 'Handset Type', 'Handset Manufacturer']:
            df[col] = df[col].astype('category')
        
        print(f"Successfully loaded {len(df)} records.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*50)
    print(f" {title} ")
    print("="*50)

def main():
    # Set style for plots
    sns.set_theme(style="whitegrid")
    
    # Load data
    print_section_header("Loading Data")
    df = load_data()
    
    if df is None:
        print("Failed to load data. Please check the data path.")
        return
    
    # Initialize analyzer
    analyzer = UserOverviewAnalyzer(df)
    
    # Task 1.1: Handset Analysis
    print_section_header("Task 1.1 - Handset Analysis")
    handset_results = analyzer.analyze_handsets()
    
    print("\n1. Top 10 Handsets:")
    print(handset_results['top_handsets'])
    
    print("\n2. Top 3 Manufacturers:")
    print(handset_results['top_manufacturers'])
    
    print("\n3. Top 5 Handsets per Top 3 Manufacturers:")
    for manufacturer, handsets in handset_results['top_handsets_per_manufacturer'].items():
        print(f"\n{manufacturer}:")
        print(handsets)
    
    # Task 1.2: User Behavior Analysis
    print_section_header("Task 1.2 - User Behavior Analysis")
    user_metrics = analyzer.analyze_user_behavior()
    
    print("\n1. Basic Statistics:")
    print(user_metrics.describe())
    
    print("\n2. Data Usage by Duration Decile:")
    decile_stats = user_metrics.groupby('duration_decile', observed=True)['total_data'].agg(['mean', 'count'])
    print(decile_stats)
    
    # Application Analysis
    print_section_header("Application Usage Analysis")
    app_analysis = analyzer.analyze_applications()
    
    print("\nApplication Usage Summary:")
    for app, metrics in app_analysis.items():
        print(f"\n{app}:")
        for metric, value in metrics.items():
            if metric == 'percentage_of_total':
                print(f"  {metric}: {value:.2f}%")
            else:
                print(f"  {metric}: {value:,} bytes")
    
    # Generate Visualizations
    print_section_header("Generating Visualizations")
    analyzer.generate_visualizations()
    
    # PCA Analysis
    print_section_header("PCA Analysis")
    pca_results = analyzer.perform_pca()
    
    print("\nExplained Variance Ratios:")
    for i, ratio in enumerate(pca_results['explained_variance_ratio'], 1):
        print(f"Component {i}: {ratio:.4f} ({ratio*100:.2f}%)")
    
    print("\nCumulative Variance Explained:")
    for i, ratio in enumerate(pca_results['cumulative_variance_ratio'], 1):
        print(f"Components 1-{i}: {ratio:.4f} ({ratio*100:.2f}%)")
    
    print_section_header("Analysis Complete")
    print("""
Key Findings:
1. Handset Analysis: See 'plots/top_handsets.png' for visualization
2. User Behavior: Check 'plots/data_usage_by_decile.png' for usage patterns
3. Application Usage: View 'plots/app_usage_distribution.png' for distribution
4. Correlation Analysis: Examine 'plots/correlation_matrix.png' for relationships
5. PCA Results: See 'plots/pca_variance_ratio.png' for dimensionality reduction analysis

All visualizations have been saved in the 'plots' directory.
""")

if __name__ == "__main__":
    main()
