# Telecom User Engagement Analysis Dashboard

A comprehensive dashboard for analyzing telecom user engagement patterns and behaviors.

## Features

### 1. Overview
- Basic statistics about user engagement
- Summary metrics and data distribution

### 2. User Engagement Analysis
- Session frequency distribution
- Total traffic distribution
- Top users analysis

### 3. Application Usage Analysis
- Total traffic by application
- Top users per application
- Application usage patterns

### 4. Clustering Analysis
- K-means clustering of users
- Cluster characteristics visualization
- User distribution across clusters

### 5. Advanced Analysis
- Correlation analysis between engagement metrics
- Time-based usage patterns
- User segmentation analysis
- Application usage patterns
- User behavior patterns through radar charts

### 6. Experience Analytics
- User experience metrics analysis
- Performance analysis
- User segmentation based on experience

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/telecom-analysis.git
cd telecom-analysis
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your telecom data file (xdr_data.parquet) in the `data` directory

2. Run the Streamlit dashboard:
```bash
streamlit run src/dashboard.py
```

3. Access the dashboard in your web browser at http://localhost:8501

## Project Structure

```
telecom-analysis/
├── data/
│   └── xdr_data.parquet
├── src/
│   ├── dashboard.py
│   ├── main.py
│   ├── user_engagement_analysis.py
│   ├── user_overview_analysis.py
│   └── experience_analytics.py
├── requirements.txt
└── README.md
```

## Dependencies

- pandas
- numpy
- streamlit
- plotly
- scikit-learn
- matplotlib
- seaborn
- kneed

## License

MIT License
