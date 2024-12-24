import pandas as pd
import numpy as np
from user_overview_analysis import UserOverviewAnalyzer
from user_engagement_analysis import UserEngagementAnalyzer
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
            'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
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
        print("All requested columns:", required_cols)
        try:
            # Try to read all columns to show what's available
            df_test = pd.read_parquet(data_path)
            print("Available columns:", df_test.columns.tolist())
        except Exception as e2:
            print(f"Error reading data file: {e2}")
        return None

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*50)
    print(f" {title} ")
    print("="*50)

def main():
    # Load data
    print_section_header("Loading Data")
    df = load_data()
    
    if df is None:
        print("Failed to load data. Please check the data path.")
        return
    
    # Task 1: User Overview Analysis
    print_section_header("Task 1: User Overview Analysis")
    overview_analyzer = UserOverviewAnalyzer(df)
    
    # Task 1.1: Handset Analysis
    print_section_header("Task 1.1 - Handset Analysis")
    handset_results = overview_analyzer.analyze_handsets()
    
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
    user_metrics = overview_analyzer.analyze_user_behavior()
    
    print("\n1. Basic Statistics:")
    print(user_metrics.describe())
    
    print("\n2. Data Usage by Duration Decile:")
    decile_stats = user_metrics.groupby('duration_decile', observed=True)['total_data'].agg(['mean', 'count'])
    print(decile_stats)
    
    # Task 2: User Engagement Analysis
    print_section_header("Task 2: User Engagement Analysis")
    engagement_analyzer = UserEngagementAnalyzer(df)
    
    # Task 2.1: Top Users and Clustering
    print_section_header("Task 2.1 - User Engagement Metrics")
    
    # Get top users per metric
    top_users = engagement_analyzer.get_top_users()
    print("\n1. Top 10 Users per Engagement Metric:")
    for metric, users in top_users.items():
        print(f"\n{metric}:")
        print(users)
    
    # Get cluster statistics
    print("\n2. Engagement Cluster Statistics:")
    cluster_stats = engagement_analyzer.cluster_users()
    print(cluster_stats)
    
    # Get top users per application
    print("\n3. Top 10 Most Engaged Users per Application:")
    top_app_users = engagement_analyzer.get_top_users_per_app()
    for app, users in top_app_users.items():
        print(f"\n{app}:")
        print(users)
    
    # Find optimal number of clusters
    optimal_k, _, _ = engagement_analyzer.find_optimal_k()
    print(f"\n4. Optimal number of clusters (k): {optimal_k}")
    
    # Generate visualizations
    print("\n5. Generating engagement analysis visualizations...")
    engagement_analyzer.plot_engagement_analysis()
    
    print("\nAnalysis complete! Check the 'plots' directory for visualizations.")

if __name__ == "__main__":
    main()
