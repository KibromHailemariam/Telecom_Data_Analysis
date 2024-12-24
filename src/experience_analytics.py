import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

class ExperienceAnalyzer:
    """Class for analyzing user experience metrics in telecom data."""
    
    def __init__(self, data):
        """Initialize with telecom data."""
        self.xdr_data = data
        self.user_metrics = None
        self.clean_data()
        
    def clean_data(self):
        """Clean the data by handling missing values and outliers."""
        # Convert columns to numeric
        numeric_cols = ['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
                       'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
                       'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']
        
        for col in numeric_cols:
            self.xdr_data[col] = pd.to_numeric(self.xdr_data[col], errors='coerce')
        
        # Handle missing values with mean
        for col in numeric_cols:
            mean_val = self.xdr_data[col].mean()
            self.xdr_data[col].fillna(mean_val, inplace=True)
        
        # Handle missing handset types with mode
        mode_handset = self.xdr_data['Handset Type'].mode()[0]
        self.xdr_data['Handset Type'].fillna(mode_handset, inplace=True)
        
        # Handle outliers using IQR method
        for col in numeric_cols:
            Q1 = self.xdr_data[col].quantile(0.25)
            Q3 = self.xdr_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Replace outliers with mean
            mean_val = self.xdr_data[(self.xdr_data[col] >= lower_bound) & 
                                   (self.xdr_data[col] <= upper_bound)][col].mean()
            self.xdr_data.loc[self.xdr_data[col] < lower_bound, col] = mean_val
            self.xdr_data.loc[self.xdr_data[col] > upper_bound, col] = mean_val
    
    def aggregate_user_metrics(self):
        """Aggregate metrics per customer."""
        # Calculate total TCP retransmission
        self.xdr_data['Total TCP Retrans'] = (
            self.xdr_data['TCP DL Retrans. Vol (Bytes)'] + 
            self.xdr_data['TCP UL Retrans. Vol (Bytes)']
        )
        
        # Calculate average RTT
        self.xdr_data['Avg RTT'] = (
            self.xdr_data['Avg RTT DL (ms)'] + 
            self.xdr_data['Avg RTT UL (ms)']
        ) / 2
        
        # Calculate average throughput
        self.xdr_data['Avg Throughput'] = (
            self.xdr_data['Avg Bearer TP DL (kbps)'] + 
            self.xdr_data['Avg Bearer TP UL (kbps)']
        ) / 2
        
        # Aggregate per customer
        self.user_metrics = self.xdr_data.groupby('MSISDN/Number').agg({
            'Total TCP Retrans': 'mean',
            'Avg RTT': 'mean',
            'Handset Type': lambda x: x.mode()[0],  # Most frequent handset type
            'Avg Throughput': 'mean'
        }).reset_index()
        
        return self.user_metrics
    
    def get_extreme_values(self, column, n=10):
        """Get top, bottom, and most frequent values for a metric."""
        values = self.xdr_data[column].copy()
        
        return {
            'top': values.nlargest(n).tolist(),
            'bottom': values.nsmallest(n).tolist(),
            'most_frequent': values.value_counts().head(n).index.tolist()
        }
    
    def analyze_throughput_distribution(self):
        """Analyze throughput distribution per handset type."""
        # Calculate average throughput per handset
        throughput_stats = self.xdr_data.groupby('Handset Type').agg({
            'Avg Bearer TP DL (kbps)': ['mean', 'std', 'count'],
            'Avg Bearer TP UL (kbps)': ['mean', 'std', 'count']
        }).round(2)
        
        # Create visualization
        fig = go.Figure()
        
        # Add download throughput
        fig.add_trace(go.Box(
            y=self.xdr_data['Avg Bearer TP DL (kbps)'],
            x=self.xdr_data['Handset Type'],
            name='Download Throughput',
            boxmean=True
        ))
        
        # Add upload throughput
        fig.add_trace(go.Box(
            y=self.xdr_data['Avg Bearer TP UL (kbps)'],
            x=self.xdr_data['Handset Type'],
            name='Upload Throughput',
            boxmean=True
        ))
        
        fig.update_layout(
            title='Throughput Distribution by Handset Type',
            xaxis_title='Handset Type',
            yaxis_title='Throughput (kbps)',
            boxmode='group'
        )
        
        return throughput_stats, fig
    
    def analyze_tcp_retransmission(self):
        """Analyze TCP retransmission per handset type."""
        # Calculate average TCP retransmission per handset
        tcp_stats = self.xdr_data.groupby('Handset Type').agg({
            'TCP DL Retrans. Vol (Bytes)': ['mean', 'std', 'count'],
            'TCP UL Retrans. Vol (Bytes)': ['mean', 'std', 'count']
        }).round(2)
        
        # Create visualization
        fig = go.Figure()
        
        # Add download retransmission
        fig.add_trace(go.Box(
            y=self.xdr_data['TCP DL Retrans. Vol (Bytes)'],
            x=self.xdr_data['Handset Type'],
            name='Download Retransmission',
            boxmean=True
        ))
        
        # Add upload retransmission
        fig.add_trace(go.Box(
            y=self.xdr_data['TCP UL Retrans. Vol (Bytes)'],
            x=self.xdr_data['Handset Type'],
            name='Upload Retransmission',
            boxmean=True
        ))
        
        fig.update_layout(
            title='TCP Retransmission Distribution by Handset Type',
            xaxis_title='Handset Type',
            yaxis_title='Retransmission Volume (Bytes)',
            boxmode='group'
        )
        
        return tcp_stats, fig
    
    def cluster_users(self, k=3):
        """Perform k-means clustering on user experience metrics."""
        if self.user_metrics is None:
            self.aggregate_user_metrics()
        
        # Select features for clustering
        features = ['Total TCP Retrans', 'Avg RTT', 'Avg Throughput']
        X = self.user_metrics[features]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        self.user_metrics['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Calculate cluster statistics
        cluster_stats = self.user_metrics.groupby('Cluster').agg({
            'Total TCP Retrans': 'mean',
            'Avg RTT': 'mean',
            'Avg Throughput': 'mean',
            'MSISDN/Number': 'count'
        }).round(2)
        
        # Create cluster visualization
        fig = px.scatter_3d(
            self.user_metrics,
            x='Total TCP Retrans',
            y='Avg RTT',
            z='Avg Throughput',
            color='Cluster',
            title='User Experience Clusters',
            labels={
                'Total TCP Retrans': 'TCP Retransmission',
                'Avg RTT': 'Average RTT (ms)',
                'Avg Throughput': 'Average Throughput (kbps)'
            }
        )
        
        return cluster_stats, fig
    
    def get_cluster_descriptions(self):
        """Generate descriptions for each cluster based on their characteristics."""
        if 'Cluster' not in self.user_metrics.columns:
            _, _ = self.cluster_users()
        
        cluster_stats = self.user_metrics.groupby('Cluster').agg({
            'Total TCP Retrans': 'mean',
            'Avg RTT': 'mean',
            'Avg Throughput': 'mean',
            'MSISDN/Number': 'count'
        })
        
        descriptions = {}
        for cluster in cluster_stats.index:
            stats = cluster_stats.loc[cluster]
            
            # Determine relative performance for each metric
            tcp_level = "high" if stats['Total TCP Retrans'] > cluster_stats['Total TCP Retrans'].mean() else "low"
            rtt_level = "high" if stats['Avg RTT'] > cluster_stats['Avg RTT'].mean() else "low"
            throughput_level = "high" if stats['Avg Throughput'] > cluster_stats['Avg Throughput'].mean() else "low"
            
            # Generate description
            description = f"Cluster {cluster} ({stats['MSISDN/Number']} users):\n"
            description += f"- {tcp_level.title()} TCP retransmission rate ({stats['Total TCP Retrans']:.2f} bytes)\n"
            description += f"- {rtt_level.title()} average RTT ({stats['Avg RTT']:.2f} ms)\n"
            description += f"- {throughput_level.title()} average throughput ({stats['Avg Throughput']:.2f} kbps)\n"
            
            if tcp_level == "low" and rtt_level == "low" and throughput_level == "high":
                description += "→ Excellent network experience"
            elif tcp_level == "high" and rtt_level == "high" and throughput_level == "low":
                description += "→ Poor network experience"
            else:
                description += "→ Moderate network experience"
            
            descriptions[cluster] = description
        
        return descriptions
