import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, Dict

class SatisfactionAnalyzer:
    def __init__(self, engagement_analyzer, experience_analyzer):
        """Initialize the SatisfactionAnalyzer with engagement and experience data."""
        self.engagement_analyzer = engagement_analyzer
        self.experience_analyzer = experience_analyzer
        self.satisfaction_data = None
        
    def calculate_engagement_score(self) -> pd.Series:
        """Calculate engagement score based on distance from least engaged cluster."""
        # Get engagement metrics
        engagement_metrics = self.engagement_analyzer.user_metrics[
            ['session_count', 'total_duration', 'total_traffic']
        ]
        
        # Standardize the features
        scaler = StandardScaler()
        scaled_metrics = scaler.fit_transform(engagement_metrics)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_metrics)
        
        # Find the least engaged cluster (cluster with lowest average metrics)
        cluster_means = pd.DataFrame(scaled_metrics).groupby(clusters).mean()
        least_engaged_cluster = cluster_means.mean(axis=1).idxmin()
        
        # Calculate Euclidean distance from least engaged cluster centroid
        least_engaged_centroid = kmeans.cluster_centers_[least_engaged_cluster]
        distances = euclidean_distances(scaled_metrics, [least_engaged_centroid])
        
        return pd.Series(distances.flatten(), index=engagement_metrics.index)
    
    def calculate_experience_score(self) -> pd.Series:
        """Calculate experience score based on distance from worst experience cluster."""
        # Get experience metrics
        experience_metrics = self.experience_analyzer.get_experience_metrics()
        
        if experience_metrics is None or experience_metrics.empty:
            # Return dummy scores if no experience metrics available
            return pd.Series(np.zeros(len(self.engagement_analyzer.user_metrics)), 
                           index=self.engagement_analyzer.user_metrics.index)
        
        # Standardize the features
        scaler = StandardScaler()
        scaled_metrics = scaler.fit_transform(experience_metrics)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_metrics)
        
        # Find the worst experience cluster (cluster with highest RTT and retransmission)
        cluster_means = pd.DataFrame(scaled_metrics).groupby(clusters).mean()
        worst_experience_cluster = cluster_means.mean(axis=1).idxmin()
        
        # Calculate Euclidean distance from worst experience cluster centroid
        worst_centroid = kmeans.cluster_centers_[worst_experience_cluster]
        distances = euclidean_distances(scaled_metrics, [worst_centroid])
        
        return pd.Series(distances.flatten(), index=experience_metrics.index)
    
    def calculate_satisfaction_scores(self) -> pd.DataFrame:
        """Calculate overall satisfaction scores and create final dataframe."""
        # Calculate scores
        engagement_scores = self.calculate_engagement_score()
        experience_scores = self.calculate_experience_score()
        
        # Create satisfaction dataframe
        satisfaction_df = pd.DataFrame({
            'user_id': engagement_scores.index,
            'engagement_score': engagement_scores,
            'experience_score': experience_scores
        })
        
        # Calculate satisfaction score as average of engagement and experience scores
        satisfaction_df['satisfaction_score'] = (
            satisfaction_df['engagement_score'] + satisfaction_df['experience_score']
        ) / 2
        
        self.satisfaction_data = satisfaction_df
        return satisfaction_df
    
    def get_top_satisfied_customers(self, n: int = 10) -> pd.DataFrame:
        """Get the top n most satisfied customers."""
        if self.satisfaction_data is None:
            self.calculate_satisfaction_scores()
        
        return self.satisfaction_data.nlargest(n, 'satisfaction_score')
    
    def build_satisfaction_model(self) -> Tuple[RandomForestRegressor, Dict]:
        """Build and evaluate a regression model to predict satisfaction scores."""
        if self.satisfaction_data is None:
            self.calculate_satisfaction_scores()
        
        # Prepare features
        X = self.satisfaction_data[['engagement_score', 'experience_score']]
        y = self.satisfaction_data['satisfaction_score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        return model, {
            'train_score': train_score,
            'test_score': test_score
        }
    
    def cluster_satisfaction_scores(self) -> pd.DataFrame:
        """Run k-means clustering on engagement and experience scores."""
        if self.satisfaction_data is None:
            self.calculate_satisfaction_scores()
        
        # Prepare data for clustering
        X = self.satisfaction_data[['engagement_score', 'experience_score']]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to satisfaction data
        self.satisfaction_data['cluster'] = clusters
        
        # Calculate aggregate scores per cluster
        cluster_stats = self.satisfaction_data.groupby('cluster').agg({
            'engagement_score': 'mean',
            'experience_score': 'mean',
            'satisfaction_score': 'mean'
        }).round(3)
        
        return cluster_stats
