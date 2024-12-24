import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator

class UserEngagementAnalyzer:
    """Class for analyzing user engagement patterns and clustering."""
    
    def __init__(self, xdr_data: pd.DataFrame):
        """Initialize with XDR data."""
        self.xdr_data = xdr_data.copy()
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess data for engagement analysis."""
        # Convert numeric columns to float
        numeric_cols = [col for col in self.xdr_data.columns if 'DL (Bytes)' in col or 'UL (Bytes)' in col or 'Dur. (ms)' in col]
        self.xdr_data[numeric_cols] = self.xdr_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # Calculate total traffic per session
        dl_cols = [col for col in self.xdr_data.columns if 'DL (Bytes)' in col]
        ul_cols = [col for col in self.xdr_data.columns if 'UL (Bytes)' in col]
        
        self.xdr_data['total_traffic'] = (
            self.xdr_data[dl_cols].sum(axis=1) + 
            self.xdr_data[ul_cols].sum(axis=1)
        )
        
        # Convert duration to minutes
        self.xdr_data['duration_minutes'] = self.xdr_data['Dur. (ms)'] / (1000 * 60)
        
        # Calculate per-application traffic
        self.app_cols = {
            'Social Media': ['Social Media DL (Bytes)', 'Social Media UL (Bytes)'],
            'Google': ['Google DL (Bytes)', 'Google UL (Bytes)'],
            'Email': ['Email DL (Bytes)', 'Email UL (Bytes)'],
            'Youtube': ['Youtube DL (Bytes)', 'Youtube UL (Bytes)'],
            'Netflix': ['Netflix DL (Bytes)', 'Netflix UL (Bytes)'],
            'Gaming': ['Gaming DL (Bytes)', 'Gaming UL (Bytes)'],
            'Other': ['Other DL (Bytes)', 'Other UL (Bytes)']
        }
        
        for app, cols in self.app_cols.items():
            self.xdr_data[f'{app}_total'] = self.xdr_data[cols].sum(axis=1)
        
        # Aggregate metrics per user
        self.user_metrics = self.xdr_data.groupby('MSISDN/Number').agg({
            'Bearer Id': 'count',  # Session frequency
            'duration_minutes': ['sum', 'mean'],  # Duration metrics
            'total_traffic': ['sum', 'mean']  # Traffic metrics
        }).reset_index()
        
        # Flatten column names
        self.user_metrics.columns = [
            'msisdn', 'session_count', 'total_duration', 'avg_duration',
            'total_traffic', 'avg_traffic'
        ]
        
        # Calculate per-app metrics per user
        for app in self.app_cols.keys():
            app_metrics = self.xdr_data.groupby('MSISDN/Number')[f'{app}_total'].sum()
            self.user_metrics[f'{app}_traffic'] = app_metrics.values
    
    def get_top_users(self, n=10):
        """Get top 10 users per engagement metric."""
        metrics = {
            'session_count': 'Session Frequency',
            'total_duration': 'Total Duration (minutes)',
            'total_traffic': 'Total Traffic (bytes)'
        }
        
        top_users = {}
        for metric, label in metrics.items():
            top = self.user_metrics.nlargest(n, metric)[['msisdn', metric]]
            top_users[label] = top
        
        return top_users
    
    def get_top_users_per_app(self, n=10):
        """Get top 10 most engaged users per application."""
        app_columns = [col for col in self.user_metrics.columns if '_traffic' in col]
        top_users_per_app = {}
        
        for app_col in app_columns:
            app_name = app_col.replace('_traffic', '')
            top = self.user_metrics.nlargest(n, app_col)[['msisdn', app_col]]
            top_users_per_app[app_name] = top
        
        return top_users_per_app
    
    def cluster_users(self, k=3):
        """Cluster users based on engagement metrics."""
        # Select features for clustering
        features = ['session_count', 'total_duration', 'total_traffic']
        X = self.user_metrics[features]
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        self.user_metrics['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Calculate cluster statistics
        cluster_stats = self.user_metrics.groupby('cluster').agg({
            'session_count': ['min', 'max', 'mean', 'sum'],
            'total_duration': ['min', 'max', 'mean', 'sum'],
            'total_traffic': ['min', 'max', 'mean', 'sum']
        })
        
        return cluster_stats
    
    def find_optimal_k(self, max_k=10):
        """Find optimal number of clusters using elbow method."""
        features = ['session_count', 'total_duration', 'total_traffic']
        X = self.user_metrics[features]
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate inertia for different k values
        inertias = []
        k_values = range(1, max_k + 1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        kl = KneeLocator(
            k_values, inertias, curve='convex', direction='decreasing'
        )
        
        return kl.elbow, k_values, inertias
    
    def plot_engagement_analysis(self):
        """Generate visualizations for engagement analysis."""
        # Set style
        plt.style.use('seaborn')
        
        # 1. Top 3 Most Used Applications
        app_totals = self.user_metrics[[col for col in self.user_metrics.columns 
                                      if '_traffic' in col]].sum()
        app_totals = app_totals.sort_values(ascending=False)
        top_3_apps = app_totals.head(3)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(top_3_apps)), top_3_apps.values)
        plt.title('Top 3 Most Used Applications')
        plt.xlabel('Application')
        plt.ylabel('Total Traffic (bytes)')
        plt.xticks(range(len(top_3_apps)), [col.replace('_traffic', '') 
                                           for col in top_3_apps.index])
        plt.savefig('plots/top_3_apps.png')
        plt.close()
        
        # 2. Elbow Method Plot
        optimal_k, k_values, inertias = self.find_optimal_k()
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, inertias, 'bo-')
        plt.axvline(x=optimal_k, color='r', linestyle='--', 
                   label=f'Elbow at k={optimal_k}')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.legend()
        plt.savefig('plots/elbow_method.png')
        plt.close()
        
        # 3. Cluster Characteristics
        self.cluster_users(k=optimal_k)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = ['session_count', 'total_duration', 'total_traffic']
        titles = ['Session Frequency', 'Total Duration', 'Total Traffic']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            sns.boxplot(data=self.user_metrics, x='cluster', y=metric, ax=axes[i])
            axes[i].set_title(f'{title} by Cluster')
            axes[i].set_xlabel('Cluster')
        
        plt.tight_layout()
        plt.savefig('plots/cluster_characteristics.png')
        plt.close()
        
        return {
            'optimal_k': optimal_k,
            'top_3_apps': top_3_apps
        }
