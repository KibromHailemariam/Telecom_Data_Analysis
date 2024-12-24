# Telecom User Overview Analysis

## Project Overview
This project analyzes telecom user behavior and handset usage patterns, focusing on understanding user preferences and data consumption patterns.

## Task 1: User Overview Analysis

### Analysis Components

1. **Handset Analysis**
   - Identifies top 10 handsets used by customers
   - Analyzes top 3 handset manufacturers
   - Shows top 5 handsets per manufacturer
   - Calculates market share for manufacturers

2. **User Behavior Analysis**
   - Session count and duration analysis
   - Data consumption patterns
   - Usage trends across different applications
   - User segmentation by duration deciles

3. **Application Usage Analysis**
   - Data usage distribution across applications
   - Active users per application
   - Upload/download patterns
   - Application correlations

4. **Statistical Analysis**
   - Descriptive statistics
   - Correlation analysis
   - Principal Component Analysis (PCA)
   - Data distribution analysis

### Visualizations Generated
- Top handsets distribution
- Manufacturer market share
- Data usage by duration decile
- Session distribution
- Application usage distribution
- Active users per application
- Application correlation matrix
- PCA analysis results

## Setup and Installation

### Prerequisites
- Python 3.8+
- Required packages listed in requirements.txt

### Installation
```bash
# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation
1. Place your telecom XDR data file in the `data` directory
2. Ensure the data file is named `xdr_data.parquet`

### Running the Analysis
```bash
python src/main.py
```

### Output
The analysis will generate:
1. Detailed statistics in the console output
2. Visualizations in the `plots` directory

## Project Structure
```
.
├── data/               # Data directory
│   └── xdr_data.parquet
├── plots/              # Generated visualizations
├── src/               
│   ├── main.py              # Main script to run analysis
│   └── user_overview_analysis.py  # Analysis implementation
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Results
The analysis provides insights into:
1. Most popular handsets and manufacturers
2. User behavior patterns and preferences
3. Application usage distribution
4. Data consumption patterns
5. Statistical relationships between different metrics

Check the generated plots in the `plots` directory for visual representations of the analysis results.

## License
MIT License
