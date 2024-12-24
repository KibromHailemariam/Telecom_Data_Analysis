import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

class UserOverviewAnalyzer:
    """Class for analyzing user overview and handset data."""
    
    def __init__(self, xdr_data: pd.DataFrame):
        """Initialize with XDR data."""
        self.xdr_data = xdr_data.copy()  # Create a copy to avoid warnings
        self.clean_data()
    
    def clean_data(self):
        """Clean the data by handling missing values and outliers efficiently."""
        # Define numeric columns
        numeric_cols = [
            'Dur. (ms)', 'HTTP DL (Bytes)', 'HTTP UL (Bytes)',
            'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
            'Google DL (Bytes)', 'Google UL (Bytes)',
            'Email DL (Bytes)', 'Email UL (Bytes)',
            'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
            'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
            'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
            'Other DL (Bytes)', 'Other UL (Bytes)'
        ]
        
        # Replace '\N' with NaN for all columns at once
        self.xdr_data = self.xdr_data.replace('\\N', np.nan)
        
        # Convert all numeric columns to float64 at once
        self.xdr_data[numeric_cols] = self.xdr_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # Calculate medians for all columns at once
        medians = self.xdr_data[numeric_cols].median()
        
        # Fill NaN values for all columns at once
        self.xdr_data[numeric_cols] = self.xdr_data[numeric_cols].fillna(medians)
        
        # Handle outliers for all columns using vectorized operations
        Q1 = self.xdr_data[numeric_cols].quantile(0.25)
        Q3 = self.xdr_data[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        lower_bounds = Q1 - 1.5 * IQR
        upper_bounds = Q3 + 1.5 * IQR
        
        # Clip values outside bounds
        self.xdr_data[numeric_cols] = self.xdr_data[numeric_cols].clip(
            lower=lower_bounds, 
            upper=upper_bounds, 
            axis=1
        )
    
    def analyze_handsets(self):
        """
        Analyze handset usage and manufacturers.
        Returns detailed statistics about handset usage patterns.
        """
        # Top 10 handsets
        top_handsets = self.xdr_data['Handset Type'].value_counts().head(10)
        
        # Top 3 manufacturers
        top_manufacturers = self.xdr_data['Handset Manufacturer'].value_counts().head(3)
        
        # Top 5 handsets per manufacturer
        top_handsets_per_manufacturer = {}
        manufacturer_market_share = {}
        
        for manufacturer in top_manufacturers.index:
            # Get manufacturer's data
            manufacturer_data = self.xdr_data[
                self.xdr_data['Handset Manufacturer'] == manufacturer
            ]
            
            # Top 5 handsets
            top_handsets_per_manufacturer[manufacturer] = (
                manufacturer_data['Handset Type'].value_counts().head(5)
            )
            
            # Calculate market share
            manufacturer_market_share[manufacturer] = (
                len(manufacturer_data) / len(self.xdr_data) * 100
            )
        
        return {
            'top_handsets': top_handsets,
            'top_manufacturers': top_manufacturers,
            'top_handsets_per_manufacturer': top_handsets_per_manufacturer,
            'manufacturer_market_share': manufacturer_market_share
        }
    
    def analyze_user_behavior(self):
        """
        Analyze user behavior on different applications.
        Returns detailed metrics about user engagement and data usage patterns.
        """
        # Aggregate per user
        user_metrics = self.xdr_data.groupby('MSISDN/Number').agg({
            'Bearer Id': 'count',  # number of sessions
            'Dur. (ms)': ['sum', 'mean', 'median'],  # duration metrics
            'HTTP DL (Bytes)': 'sum',
            'HTTP UL (Bytes)': 'sum'
        })
        
        # Flatten column names
        user_metrics.columns = [
            'session_count', 'total_duration', 'avg_duration', 'median_duration',
            'total_download', 'total_upload'
        ]
        
        # Add total data volume
        user_metrics['total_data'] = (
            user_metrics['total_download'] + user_metrics['total_upload']
        )
        
        # Create duration deciles
        user_metrics['duration_decile'] = pd.qcut(
            user_metrics['total_duration'], 
            q=10, 
            labels=['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
        )
        
        # Add additional metrics
        user_metrics['avg_data_per_session'] = (
            user_metrics['total_data'] / user_metrics['session_count'].replace(0, 1)
        )
        
        user_metrics['download_upload_ratio'] = (
            user_metrics['total_download'] / user_metrics['total_upload'].replace(0, 1)
        )
        
        return user_metrics
    
    def analyze_applications(self):
        """
        Analyze application usage patterns.
        Returns detailed statistics about application usage and data consumption.
        """
        apps = ['Social Media', 'Google', 'Email', 'YouTube', 'Netflix', 'Gaming', 'Other']
        app_metrics = {}
        
        for app in apps:
            dl_col = f'{app} DL (Bytes)'
            ul_col = f'{app} UL (Bytes)'
            
            if dl_col in self.xdr_data.columns and ul_col in self.xdr_data.columns:
                # Calculate basic metrics
                total_dl = self.xdr_data[dl_col].sum()
                total_ul = self.xdr_data[ul_col].sum()
                total_data = total_dl + total_ul
                
                # Calculate user engagement
                active_users = (
                    (self.xdr_data[dl_col] > 0) | 
                    (self.xdr_data[ul_col] > 0)
                ).sum()
                
                app_metrics[app] = {
                    'total_download': total_dl,
                    'total_upload': total_ul,
                    'total_data': total_data,
                    'active_users': active_users,
                    'avg_data_per_user': total_data / active_users if active_users > 0 else 0,
                    'percentage_of_total': 0  # Will be calculated after all apps
                }
        
        # Calculate percentages
        total_data_all_apps = sum(m['total_data'] for m in app_metrics.values())
        for app in app_metrics:
            app_metrics[app]['percentage_of_total'] = (
                app_metrics[app]['total_data'] / total_data_all_apps * 100
            )
        
        return app_metrics
    
    def perform_pca(self):
        """
        Perform Principal Component Analysis on application data.
        Returns PCA results and explained variance ratios.
        """
        apps = ['Social Media', 'Google', 'Email', 'Netflix', 'Gaming', 'Other']  # Removed YouTube as it's part of streaming
        pca_data = pd.DataFrame()
        
        for app in apps:
            dl_col = f'{app} DL (Bytes)'
            ul_col = f'{app} UL (Bytes)'
            if dl_col in self.xdr_data.columns and ul_col in self.xdr_data.columns:
                pca_data[app] = (
                    self.xdr_data[dl_col].fillna(0) + 
                    self.xdr_data[ul_col].fillna(0)
                )
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_data)
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Calculate cumulative variance ratio
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        
        # Calculate component loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(len(pca.components_))],
            index=apps
        )
        
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': cumulative_variance_ratio,
            'components': pca.components_,
            'loadings': loadings,
            'feature_names': apps
        }
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations for the analysis."""
        # Create visualizations directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        # Set style for all plots
        sns.set_theme(style="whitegrid")  # Use seaborn's set_theme instead of plt.style
        
        # 1. Handset Analysis Visualizations
        self._plot_handset_analysis()
        
        # 2. User Behavior Visualizations
        self._plot_user_behavior()
        
        # 3. Application Usage Visualizations
        self._plot_application_usage()
        
        # 4. Correlation Analysis
        self._plot_correlation_matrix()
        
        # 5. PCA Visualization
        self._plot_pca_analysis()
    
    def _plot_handset_analysis(self):
        """Generate handset analysis visualizations."""
        handset_data = self.analyze_handsets()
        
        # Plot top handsets
        plt.figure(figsize=(15, 8))
        ax = handset_data['top_handsets'].plot(kind='bar')
        plt.title('Top 10 Handsets by Usage', fontsize=14, pad=20)
        plt.xlabel('Handset Type', fontsize=12)
        plt.ylabel('Number of Users', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(handset_data['top_handsets']):
            ax.text(i, v, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('plots/top_handsets.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot manufacturer market share
        plt.figure(figsize=(10, 8))
        market_share = pd.Series(handset_data['manufacturer_market_share'])
        plt.pie(market_share, labels=market_share.index, autopct='%1.1f%%')
        plt.title('Manufacturer Market Share', fontsize=14, pad=20)
        plt.axis('equal')
        plt.savefig('plots/manufacturer_market_share.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_user_behavior(self):
        """Generate user behavior visualizations."""
        user_metrics = self.analyze_user_behavior()
        
        # Plot data usage by decile
        plt.figure(figsize=(12, 6))
        decile_data = user_metrics.groupby('duration_decile', observed=True)['total_data'].mean()
        ax = decile_data.plot(kind='bar')
        plt.title('Average Data Usage by Duration Decile', fontsize=14, pad=20)
        plt.xlabel('Duration Decile', fontsize=12)
        plt.ylabel('Average Total Data (Bytes)', fontsize=12)
        
        # Add value labels
        for i, v in enumerate(decile_data):
            ax.text(i, v, f'{v/1e6:.1f}M', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('plots/data_usage_by_decile.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot session distribution
        plt.figure(figsize=(10, 6))
        user_metrics['session_count'].hist(bins=30)
        plt.title('Distribution of Sessions per User', fontsize=14, pad=20)
        plt.xlabel('Number of Sessions', fontsize=12)
        plt.ylabel('Number of Users', fontsize=12)
        plt.savefig('plots/session_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_application_usage(self):
        """Generate application usage visualizations."""
        app_metrics = self.analyze_applications()
        
        # Plot application data distribution
        plt.figure(figsize=(12, 8))
        apps = list(app_metrics.keys())
        percentages = [app_metrics[app]['percentage_of_total'] for app in apps]
        
        plt.pie(percentages, labels=apps, autopct='%1.1f%%')
        plt.title('Application Data Usage Distribution', fontsize=14, pad=20)
        plt.axis('equal')
        plt.savefig('plots/app_usage_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot active users per application
        plt.figure(figsize=(12, 6))
        active_users = [app_metrics[app]['active_users'] for app in apps]
        plt.bar(apps, active_users)
        plt.title('Active Users per Application', fontsize=14, pad=20)
        plt.xlabel('Application', fontsize=12)
        plt.ylabel('Number of Active Users', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/app_active_users.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_matrix(self):
        """Generate correlation matrix visualization."""
        apps = ['Social Media', 'Google', 'Email', 'YouTube', 'Netflix', 'Gaming', 'Other']
        corr_data = pd.DataFrame()
        
        for app in apps:
            dl_col = f'{app} DL (Bytes)'
            ul_col = f'{app} UL (Bytes)'
            if dl_col in self.xdr_data.columns and ul_col in self.xdr_data.columns:
                corr_data[app] = (
                    self.xdr_data[dl_col].fillna(0) + 
                    self.xdr_data[ul_col].fillna(0)
                )
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_data.corr()))
        sns.heatmap(corr_data.corr(), 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   mask=mask,
                   fmt='.2f')
        plt.title('Application Usage Correlation Matrix', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig('plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pca_analysis(self):
        """Generate PCA analysis visualization."""
        pca_results = self.perform_pca()
        
        # Plot explained variance ratio
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(pca_results['explained_variance_ratio']) + 1),
                pca_results['cumulative_variance_ratio'], 'bo-')
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Cumulative Explained Variance Ratio', fontsize=12)
        plt.title('PCA Explained Variance Ratio', fontsize=14, pad=20)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/pca_variance_ratio.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot component loadings
        plt.figure(figsize=(12, 8))
        sns.heatmap(pca_results['loadings'], 
                   annot=True, 
                   cmap='coolwarm',
                   center=0,
                   fmt='.2f')
        plt.title('PCA Component Loadings', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig('plots/pca_loadings.png', dpi=300, bbox_inches='tight')
        plt.close()
