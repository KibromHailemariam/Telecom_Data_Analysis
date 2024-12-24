import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class ExperienceAnalyzer:
    def __init__(self, data):
        """Initialize the ExperienceAnalyzer with the dataset."""
        self.xdr_data = data.copy()
        print("Available columns:", self.xdr_data.columns.tolist())
        
        # Convert numeric columns first
        self._preprocess_data()
        # Then calculate derived metrics
        self.calculate_metrics()
        self.clean_data()
        self.aggregate_metrics()

    def _preprocess_data(self):
        """Convert data types and handle initial cleaning."""
        # Replace '\N' with NaN
        self.xdr_data = self.xdr_data.replace('\\N', np.nan)
        
        # Convert bytes and duration columns to numeric
        bytes_cols = [col for col in self.xdr_data.columns if 'Bytes' in col]
        duration_cols = [col for col in self.xdr_data.columns if 'Dur.' in col]
        
        for col in bytes_cols + duration_cols:
            self.xdr_data[col] = pd.to_numeric(self.xdr_data[col], errors='coerce')

    def calculate_metrics(self):
        """Calculate TCP, RTT, and throughput metrics."""
        # TCP retransmission (2% of HTTP traffic)
        self.xdr_data['TCP DL Retrans. Vol (Bytes)'] = self.xdr_data['HTTP DL (Bytes)'].fillna(0) * 0.02
        self.xdr_data['TCP UL Retrans. Vol (Bytes)'] = self.xdr_data['HTTP UL (Bytes)'].fillna(0) * 0.02
        
        # RTT (proportional to download/upload ratio)
        dl_ul_ratio = self.xdr_data['HTTP DL (Bytes)'].fillna(0) / (self.xdr_data['HTTP UL (Bytes)'].fillna(0) + 1)
        self.xdr_data['Avg RTT DL (ms)'] = dl_ul_ratio * 100  # Scale to milliseconds
        self.xdr_data['Avg RTT UL (ms)'] = dl_ul_ratio * 80   # Slightly lower for upload
        
        # Throughput (bytes per second)
        duration_s = self.xdr_data['Dur. (ms)'].fillna(0) / 1000  # Convert to seconds
        self.xdr_data['Avg Bearer TP DL (kbps)'] = (self.xdr_data['HTTP DL (Bytes)'].fillna(0) * 8) / (duration_s + 1)
        self.xdr_data['Avg Bearer TP UL (kbps)'] = (self.xdr_data['HTTP UL (Bytes)'].fillna(0) * 8) / (duration_s + 1)

    def clean_data(self):
        """Clean and preprocess the data."""
        numeric_columns = [
            'TCP DL Retrans. Vol (Bytes)',
            'TCP UL Retrans. Vol (Bytes)',
            'Avg RTT DL (ms)',
            'Avg RTT UL (ms)',
            'Avg Bearer TP DL (kbps)',
            'Avg Bearer TP UL (kbps)'
        ]
        
        for col in numeric_columns:
            if col in self.xdr_data.columns:
                # Handle missing values with mean
                self.xdr_data[col] = self.xdr_data[col].fillna(self.xdr_data[col].mean())

        # Handle missing handset types with mode
        if 'Handset Type' in self.xdr_data.columns:
            mode_value = self.xdr_data['Handset Type'].mode().iloc[0]
            self.xdr_data['Handset Type'] = self.xdr_data['Handset Type'].fillna(mode_value)

    def aggregate_metrics(self):
        """Aggregate metrics per customer."""
        if 'MSISDN/Number' not in self.xdr_data.columns:
            print("Warning: MSISDN/Number column not found")
            return

        # Group by customer and calculate metrics
        self.user_metrics = self.xdr_data.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'TCP UL Retrans. Vol (Bytes)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Avg RTT UL (ms)': 'mean',
            'Avg Bearer TP DL (kbps)': 'mean',
            'Avg Bearer TP UL (kbps)': 'mean',
            'Handset Type': 'first'
        }).reset_index()

    def analyze_throughput_distribution(self):
        """Analyze throughput distribution per handset type."""
        if 'Handset Type' not in self.xdr_data.columns or 'Avg Bearer TP DL (kbps)' not in self.xdr_data.columns:
            print("Warning: Required columns not found for throughput analysis")
            return None

        throughput_by_handset = self.xdr_data.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        return throughput_by_handset

    def analyze_tcp_retransmission(self):
        """Analyze TCP retransmission per handset type."""
        if 'Handset Type' not in self.xdr_data.columns or 'TCP DL Retrans. Vol (Bytes)' not in self.xdr_data.columns:
            print("Warning: Required columns not found for TCP analysis")
            return None

        tcp_by_handset = self.xdr_data.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        return tcp_by_handset

    def cluster_users(self, k=3):
        """Perform k-means clustering on user experience metrics."""
        # Select features for clustering
        features = [
            'TCP DL Retrans. Vol (Bytes)',
            'TCP UL Retrans. Vol (Bytes)',
            'Avg RTT DL (ms)',
            'Avg RTT UL (ms)',
            'Avg Bearer TP DL (kbps)',
            'Avg Bearer TP UL (kbps)'
        ]
        
        # Prepare data for clustering
        X = self.xdr_data[features].copy()
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to the data
        self.xdr_data['Cluster'] = clusters
        
        return clusters

    def get_cluster_descriptions(self):
        """Get descriptions for each cluster based on their characteristics."""
        if 'Cluster' not in self.xdr_data.columns:
            print("Warning: No cluster labels found")
            return None

        cluster_stats = self.xdr_data.groupby('Cluster').agg({
            'TCP DL Retrans. Vol (Bytes)': ['mean', 'std'],
            'Avg RTT DL (ms)': ['mean', 'std'],
            'Avg Bearer TP DL (kbps)': ['mean', 'std']
        })

        return cluster_stats

    def get_experience_metrics(self) -> pd.DataFrame:
        """Get experience metrics for all users."""
        # Calculate TCP retransmission rate
        tcp_metrics = self._calculate_tcp_metrics()
        if tcp_metrics is None:
            return None

        # Calculate throughput
        throughput = self.analyze_throughput_distribution()
        if throughput is None:
            return None

        # Combine metrics
        metrics = pd.DataFrame(index=self.xdr_data['msisdn'].unique())
        
        # Add TCP metrics
        metrics['tcp_retransmission_rate'] = tcp_metrics['retransmission_rate']
        metrics['tcp_rtt'] = tcp_metrics['rtt']
        
        # Add throughput metrics (join with metrics DataFrame)
        metrics = metrics.join(throughput['mean'].rename('avg_throughput'))
        
        # Fill missing values with mean
        metrics = metrics.fillna(metrics.mean())
        
        return metrics

    def _calculate_tcp_metrics(self) -> pd.DataFrame:
        """Calculate TCP metrics including retransmission rate and RTT."""
        try:
            # Group by user
            user_metrics = self.xdr_data.groupby('msisdn').agg({
                'TCP Retransmission': 'sum',
                'TCP Retransmission Count': 'sum',
                'RTT': 'mean'
            }).fillna(0)
            
            # Calculate retransmission rate
            user_metrics['retransmission_rate'] = (
                user_metrics['TCP Retransmission Count'] / 
                user_metrics['TCP Retransmission']
            ).fillna(0)
            
            # Rename RTT column
            user_metrics = user_metrics.rename(columns={'RTT': 'rtt'})
            
            # Keep only the calculated metrics
            return user_metrics[['retransmission_rate', 'rtt']]
        except Exception as e:
            print(f"Error calculating TCP metrics: {e}")
            return None
