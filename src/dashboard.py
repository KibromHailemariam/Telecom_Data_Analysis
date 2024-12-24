import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from user_engagement_analysis import UserEngagementAnalyzer
import os

# Set page config
st.set_page_config(
    page_title="Telecom User Engagement Analysis",
    page_icon="📱",
    layout="wide"
)

# Initialize data loading
@st.cache_data
def load_data():
    """Load and prepare data."""
    try:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'xdr_data.parquet')
        
        # Read required columns
        required_cols = [
            'MSISDN/Number', 'Bearer Id', 'Dur. (ms)',
            'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
            'Google DL (Bytes)', 'Google UL (Bytes)',
            'Email DL (Bytes)', 'Email UL (Bytes)',
            'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
            'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
            'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
            'Other DL (Bytes)', 'Other UL (Bytes)'
        ]
        
        df = pd.read_parquet(data_path, columns=required_cols)
        
        # Convert numeric columns
        numeric_cols = [col for col in df.columns if 'Bytes' in col or 'ms' in col]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def plot_engagement_metrics(engagement_analyzer):
    """Plot engagement metrics visualizations."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Session Frequency Distribution")
        fig = px.histogram(
            engagement_analyzer.user_metrics,
            x='session_count',
            title='Distribution of Session Counts per User',
            labels={'session_count': 'Number of Sessions', 'count': 'Number of Users'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Total Traffic Distribution")
        fig = px.histogram(
            engagement_analyzer.user_metrics,
            x='total_traffic',
            title='Distribution of Total Traffic per User',
            labels={'total_traffic': 'Total Traffic (bytes)', 'count': 'Number of Users'}
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_app_usage(engagement_analyzer):
    """Plot application usage visualizations."""
    st.subheader("Application Usage Analysis")
    
    # Calculate total traffic per app
    app_traffic = {}
    for app, cols in engagement_analyzer.app_cols.items():
        app_traffic[app] = engagement_analyzer.user_metrics[f'{app}_traffic'].sum()
    
    # Create bar chart
    fig = px.bar(
        x=list(app_traffic.keys()),
        y=list(app_traffic.values()),
        title='Total Traffic by Application',
        labels={'x': 'Application', 'y': 'Total Traffic (bytes)'}
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_clustering_results(engagement_analyzer):
    """Plot clustering analysis visualizations."""
    st.subheader("User Clustering Analysis")
    
    # Get optimal k
    optimal_k, k_values, inertias = engagement_analyzer.find_optimal_k()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot elbow curve
        fig = px.line(
            x=k_values, y=inertias,
            title=f'Elbow Method (Optimal k={optimal_k})',
            labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'}
        )
        fig.add_vline(x=optimal_k, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Perform clustering with optimal k
        cluster_stats = engagement_analyzer.cluster_users(k=optimal_k)
        
        # Plot cluster characteristics
        fig = go.Figure()
        metrics = ['session_count', 'total_duration', 'total_traffic']
        for metric in metrics:
            fig.add_trace(go.Box(
                y=engagement_analyzer.user_metrics[metric],
                x=engagement_analyzer.user_metrics['cluster'],
                name=metric
            ))
        fig.update_layout(title='Cluster Characteristics')
        st.plotly_chart(fig, use_container_width=True)

def show_top_users(engagement_analyzer):
    """Display top users analysis."""
    st.subheader("Top Users Analysis")
    
    # Get top users per metric
    top_users = engagement_analyzer.get_top_users()
    
    # Display in tabs
    tabs = st.tabs(list(top_users.keys()))
    for tab, (metric, users) in zip(tabs, top_users.items()):
        with tab:
            st.dataframe(users)

def plot_advanced_analysis(engagement_analyzer):
    """Plot advanced analysis visualizations."""
    st.subheader("Advanced Engagement Analysis")
    
    # 1. Correlation Analysis
    st.write("#### Correlation Analysis")
    # Calculate correlation matrix for engagement metrics
    metrics = ['session_count', 'total_duration', 'avg_duration', 'total_traffic', 'avg_traffic']
    corr_matrix = engagement_analyzer.user_metrics[metrics].corr()
    
    # Plot correlation heatmap
    fig = px.imshow(
        corr_matrix,
        title='Correlation Matrix of Engagement Metrics',
        labels=dict(color="Correlation"),
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Time-based Analysis
    st.write("#### Time-based Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Session frequency by hour
        hourly_sessions = engagement_analyzer.xdr_data.groupby(
            pd.to_datetime(engagement_analyzer.xdr_data['Start']).dt.hour
        )['Bearer Id'].count()
        
        fig = px.line(
            x=hourly_sessions.index,
            y=hourly_sessions.values,
            title='Session Frequency by Hour',
            labels={'x': 'Hour of Day', 'y': 'Number of Sessions'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Traffic volume by hour
        hourly_traffic = engagement_analyzer.xdr_data.groupby(
            pd.to_datetime(engagement_analyzer.xdr_data['Start']).dt.hour
        )['total_traffic'].sum()
        
        fig = px.line(
            x=hourly_traffic.index,
            y=hourly_traffic.values,
            title='Traffic Volume by Hour',
            labels={'x': 'Hour of Day', 'y': 'Total Traffic (bytes)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 3. User Segmentation Analysis
    st.write("#### User Segmentation Analysis")
    
    # Create user segments based on engagement metrics
    engagement_analyzer.user_metrics['traffic_segment'] = pd.qcut(
        engagement_analyzer.user_metrics['total_traffic'],
        q=5,
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    engagement_analyzer.user_metrics['session_segment'] = pd.qcut(
        engagement_analyzer.user_metrics['session_count'],
        q=5,
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Traffic segments distribution
        segment_counts = engagement_analyzer.user_metrics['traffic_segment'].value_counts()
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='User Distribution by Traffic Volume'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Session segments distribution
        segment_counts = engagement_analyzer.user_metrics['session_segment'].value_counts()
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='User Distribution by Session Frequency'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 4. Application Usage Patterns
    st.write("#### Application Usage Patterns")
    
    # Calculate app usage percentages
    app_cols = [col for col in engagement_analyzer.user_metrics.columns if '_traffic' in col]
    app_usage = engagement_analyzer.user_metrics[app_cols].sum()
    app_usage = app_usage / app_usage.sum() * 100
    
    # Create treemap
    fig = px.treemap(
        names=app_usage.index,
        parents=['Applications'] * len(app_usage),
        values=app_usage.values,
        title='Application Usage Distribution (%)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 5. User Behavior Patterns
    st.write("#### User Behavior Patterns")
    
    # Calculate average metrics per cluster
    cluster_metrics = engagement_analyzer.user_metrics.groupby('cluster').agg({
        'session_count': 'mean',
        'total_duration': 'mean',
        'total_traffic': 'mean'
    }).round(2)
    
    # Create radar chart
    categories = ['Session Count', 'Duration', 'Traffic']
    fig = go.Figure()
    
    for cluster in cluster_metrics.index:
        values = cluster_metrics.loc[cluster].values.tolist()
        # Normalize values for better visualization
        values = (values - np.min(values)) / (np.max(values) - np.min(values))
        values = values.tolist()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            name=f'Cluster {cluster}'
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='Cluster Behavior Patterns'
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("📱 Telecom User Engagement Analysis")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is not None:
        # Initialize engagement analyzer
        engagement_analyzer = UserEngagementAnalyzer(df)
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Select a page",
            ["Overview", "User Engagement", "Application Usage", "Clustering Analysis", "Advanced Analysis"]
        )
        
        if page == "Overview":
            st.header("Overview")
            st.write("""
            This dashboard analyzes user engagement in telecom data using various metrics:
            - Session frequency
            - Session duration
            - Total traffic
            
            Use the sidebar to navigate through different analyses.
            """)
            
            # Display basic statistics
            st.subheader("Basic Statistics")
            st.dataframe(engagement_analyzer.user_metrics.describe())
        
        elif page == "User Engagement":
            st.header("User Engagement Analysis")
            plot_engagement_metrics(engagement_analyzer)
            show_top_users(engagement_analyzer)
        
        elif page == "Application Usage":
            st.header("Application Usage Analysis")
            plot_app_usage(engagement_analyzer)
            
            # Show top users per application
            st.subheader("Top Users per Application")
            top_app_users = engagement_analyzer.get_top_users_per_app()
            app_tabs = st.tabs(list(top_app_users.keys()))
            for tab, (app, users) in zip(app_tabs, top_app_users.items()):
                with tab:
                    st.dataframe(users)
        
        elif page == "Clustering Analysis":
            st.header("User Clustering Analysis")
            plot_clustering_results(engagement_analyzer)
        
        elif page == "Advanced Analysis":
            st.header("Advanced Analysis")
            plot_advanced_analysis(engagement_analyzer)
    
    else:
        st.error("Failed to load data. Please check if the data file exists and is accessible.")

if __name__ == "__main__":
    main()
