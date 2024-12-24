import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from user_overview_analysis import UserOverviewAnalyzer
from user_engagement_analysis import UserEngagementAnalyzer
from experience_analytics import ExperienceAnalyzer
from satisfaction_analysis import SatisfactionAnalyzer

def load_data():
    """Load and preprocess the XDR data."""
    try:
        data = pd.read_parquet('data/xdr_data.parquet')
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    st.set_page_config(page_title="Telecom Analytics Dashboard", layout="wide")
    
    st.title("📱 Telecom Analytics Dashboard")
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Initialize analyzers
    overview_analyzer = UserOverviewAnalyzer(data)
    engagement_analyzer = UserEngagementAnalyzer(data)
    experience_analyzer = ExperienceAnalyzer(data)
    satisfaction_analyzer = SatisfactionAnalyzer(engagement_analyzer, experience_analyzer)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", 
                           ["Overview", "User Engagement", "Application Analysis", 
                            "Usage Patterns", "Experience", "Satisfaction"])
    
    if page == "Overview":
        show_overview_page(overview_analyzer)
    elif page == "User Engagement":
        show_engagement_page(engagement_analyzer)
    elif page == "Application Analysis":
        show_application_page(engagement_analyzer)
    elif page == "Usage Patterns":
        show_usage_patterns(engagement_analyzer)
    elif page == "Experience":
        show_experience_page(experience_analyzer)
    else:
        show_satisfaction_page(satisfaction_analyzer)

def show_overview_page(analyzer):
    st.header("📊 User Overview Analysis")
    
    col1, col2 = st.columns(2)
    
    # Get handset analysis data
    handset_data = analyzer.analyze_handsets()
    
    with col1:
        st.subheader("Top Handset Manufacturers")
        manufacturers = handset_data['top_manufacturers']
        fig = px.pie(values=manufacturers.values, names=manufacturers.index, 
                    title="Market Share by Manufacturer")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Handset Types")
        handsets = handset_data['top_handsets']
        fig = px.bar(x=handsets.index, y=handsets.values, 
                    title="Most Popular Handset Models")
        st.plotly_chart(fig, use_container_width=True)

def show_engagement_page(analyzer):
    st.header("👥 User Engagement Analysis")
    
    # Session metrics
    col1, col2, col3 = st.columns(3)
    metrics = analyzer.user_metrics
    
    with col1:
        st.metric("Avg Sessions per User", f"{metrics['session_count'].mean():.2f}")
    with col2:
        st.metric("Avg Duration (min)", f"{metrics['total_duration'].mean():.2f}")
    with col3:
        st.metric("Avg Data Usage (MB)", f"{metrics['total_traffic'].mean()/1e6:.2f}")
    
    # Traffic distribution
    st.subheader("📈 Traffic Distribution by Application")
    app_metrics = {
        'Social Media': metrics['Social Media_traffic'].sum(),
        'Google': metrics['Google_traffic'].sum(),
        'Email': metrics['Email_traffic'].sum(),
        'Youtube': metrics['Youtube_traffic'].sum(),
        'Netflix': metrics['Netflix_traffic'].sum(),
        'Gaming': metrics['Gaming_traffic'].sum(),
        'Other': metrics['Other_traffic'].sum()
    }
    
    fig = px.pie(values=list(app_metrics.values()), names=list(app_metrics.keys()),
                 title="Traffic Distribution by Application Type")
    st.plotly_chart(fig, use_container_width=True)

def show_application_page(analyzer):
    st.header("📱 Application Analysis")
    
    # Top users per application
    st.subheader("Top Users by Application")
    
    top_users = analyzer.get_top_users_per_app()
    tabs = st.tabs(list(top_users.keys()))
    
    for tab, (app, users) in zip(tabs, top_users.items()):
        with tab:
            # Convert bytes to MB for readability
            users_mb = users.copy()
            traffic_col = [col for col in users.columns if 'traffic' in col][0]
            users_mb[traffic_col] = users_mb[traffic_col] / 1e6
            users_mb = users_mb.round(2)
            
            st.dataframe(users_mb.head(10), 
                        column_config={
                            'msisdn': 'User ID',
                            traffic_col: f'{app} Traffic (MB)'
                        })
    
    # Application usage patterns
    st.subheader("📊 Usage Patterns by Application")
    
    # Get hourly usage patterns
    hourly_usage = analyzer.xdr_data.groupby(
        pd.to_datetime(analyzer.xdr_data['Start Time']).dt.hour
    )[analyzer.app_cols.keys()].sum()
    
    # Create line chart
    fig = go.Figure()
    for app in analyzer.app_cols.keys():
        fig.add_trace(go.Scatter(
            x=hourly_usage.index,
            y=hourly_usage[app],
            name=app,
            mode='lines'
        ))
    
    fig.update_layout(
        title="Hourly Application Usage",
        xaxis_title="Hour of Day",
        yaxis_title="Traffic Volume (bytes)"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_usage_patterns(analyzer):
    st.header("📊 Usage Pattern Analysis")
    
    # User segments based on total traffic
    st.subheader("User Segmentation")
    
    # Create traffic segments
    metrics = analyzer.user_metrics
    metrics['traffic_segment'] = pd.qcut(
        metrics['total_traffic'],
        q=5,
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Traffic segment distribution
        segment_counts = metrics['traffic_segment'].value_counts()
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='User Distribution by Traffic Volume'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average session duration by segment
        avg_duration = metrics.groupby('traffic_segment')['total_duration'].mean()
        fig = px.bar(
            x=avg_duration.index,
            y=avg_duration.values,
            title='Average Session Duration by User Segment',
            labels={'x': 'User Segment', 'y': 'Average Duration (minutes)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # User behavior patterns
    st.subheader("📱 User Behavior Analysis")
    
    # Correlation heatmap
    behavior_metrics = ['session_count', 'total_duration', 'total_traffic']
    corr_matrix = metrics[behavior_metrics].corr()
    
    fig = px.imshow(
        corr_matrix,
        title='Correlation between User Metrics',
        labels=dict(color="Correlation"),
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_experience_page(analyzer):
    st.header("🎯 User Experience Analysis")
    
    # Throughput Analysis
    st.subheader("📶 Network Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        throughput = analyzer.analyze_throughput_distribution()
        if throughput is not None:
            # Convert to DataFrame for plotting
            df_throughput = throughput.head(10).reset_index()
            fig = px.bar(df_throughput, 
                        x='Handset Type', 
                        y='mean',
                        title="Average Throughput by Handset Type",
                        labels={'mean': 'Average Throughput (kbps)', 
                               'Handset Type': 'Device Model'})
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # User clusters
        clusters = analyzer.cluster_users(k=3)
        if clusters is not None:
            cluster_counts = pd.Series(clusters).value_counts()
            fig = px.pie(values=cluster_counts.values, 
                        names=cluster_counts.index,
                        title="User Experience Clusters",
                        labels={'index': 'Cluster', 'value': 'Number of Users'})
            st.plotly_chart(fig, use_container_width=True)

def show_satisfaction_page(analyzer):
    st.header("🎯 Customer Satisfaction Analysis")
    
    # Calculate satisfaction scores
    satisfaction_df = analyzer.calculate_satisfaction_scores()
    
    # Display top satisfied customers
    st.subheader("Top 10 Most Satisfied Customers")
    top_customers = analyzer.get_top_satisfied_customers()
    st.dataframe(top_customers)
    
    # Show satisfaction model results
    st.subheader("Satisfaction Prediction Model")
    model, scores = analyzer.build_satisfaction_model()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Score", f"{scores['train_score']:.3f}")
    with col2:
        st.metric("Test Score", f"{scores['test_score']:.3f}")
    
    # Display cluster analysis
    st.subheader("Customer Segments")
    cluster_stats = analyzer.cluster_satisfaction_scores()
    
    # Visualization of clusters
    fig = px.scatter(
        satisfaction_df,
        x='engagement_score',
        y='experience_score',
        color='cluster',
        title='Customer Segments based on Engagement and Experience',
        labels={
            'engagement_score': 'Engagement Score',
            'experience_score': 'Experience Score',
            'cluster': 'Segment'
        }
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display cluster statistics
    st.subheader("Segment Statistics")
    st.dataframe(cluster_stats)

if __name__ == "__main__":
    main()
