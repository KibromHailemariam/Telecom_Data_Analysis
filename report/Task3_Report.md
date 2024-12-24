# Task 3: Experience Analytics Report

## Overview
This report analyzes user experience in the telecom industry by focusing on network parameters and device characteristics. The analysis provides insights into user experience through TCP retransmission, Round Trip Time (RTT), throughput metrics, and handset types.

## Task 3.1: Customer Metrics Aggregation

### Data Preprocessing
- Missing values were handled by replacing them with the mean of the corresponding variable
- Outliers were identified using the IQR method and replaced with the mean
- Handset type missing values were replaced with the mode

### Aggregated Metrics per Customer
1. Average TCP Retransmission
   - Calculated from both upload and download retransmission volumes
   - Provides insight into network reliability

2. Average RTT (Round Trip Time)
   - Combined upload and download RTT
   - Indicates network latency and responsiveness

3. Handset Type
   - Most frequently used device per customer
   - Important for device-specific optimizations

4. Average Throughput
   - Combined upload and download throughput
   - Measures actual data transfer performance

## Task 3.2: Extreme Value Analysis

### TCP Retransmission Values
- Top 10: Identifies customers experiencing significant network issues
- Bottom 10: Shows customers with optimal network conditions
- Most Frequent: Reveals typical retransmission patterns

### RTT Values
- Top 10: Highlights high-latency scenarios
- Bottom 10: Shows low-latency connections
- Most Frequent: Indicates typical network response times

### Throughput Values
- Top 10: Best performing connections
- Bottom 10: Connections requiring attention
- Most Frequent: Common throughput levels

## Task 3.3: Distribution Analysis

### Throughput Distribution per Handset Type
#### Findings
1. Performance Variation
   - Different handset types show distinct throughput patterns
   - Some handsets consistently achieve higher throughput
   - Variations might be due to device capabilities

2. Impact Factors
   - Device hardware capabilities
   - Network technology support (3G, 4G, etc.)
   - User behavior patterns

### TCP Retransmission per Handset Type
#### Findings
1. Device Impact
   - Certain handset types show higher retransmission rates
   - May indicate device-specific network handling issues
   - Could be related to device firmware or network stack implementation

2. Performance Patterns
   - Some handsets show more consistent performance
   - Others exhibit higher variability in retransmission rates

## Task 3.4: User Experience Clustering

### Methodology
- K-means clustering with k=3
- Features used:
  - TCP retransmission rates
  - RTT values
  - Throughput measurements

### Cluster Descriptions

#### Cluster 0: Optimal Experience Users
- Low TCP retransmission rates
- Low RTT values
- High throughput
- Characteristics:
  - Stable network connections
  - Modern devices
  - Good network coverage areas

#### Cluster 1: Average Experience Users
- Moderate TCP retransmission
- Medium RTT values
- Average throughput
- Characteristics:
  - Typical urban users
  - Mixed device types
  - Variable network conditions

#### Cluster 2: Challenged Experience Users
- High TCP retransmission rates
- High RTT values
- Low throughput
- Characteristics:
  - Network congestion areas
  - Older devices
  - Coverage edge areas

## Recommendations

1. Device-specific Optimization
   - Optimize network parameters for popular handset types
   - Consider device capabilities in network planning

2. Network Improvements
   - Focus on areas with high RTT and retransmission rates
   - Upgrade infrastructure in challenged experience clusters

3. User Experience Enhancement
   - Provide device-specific recommendations to users
   - Implement targeted network optimizations for different user segments

4. Monitoring and Maintenance
   - Regular monitoring of experience metrics
   - Proactive identification of degrading performance

## Conclusion
The analysis reveals significant variations in user experience across different handset types and network conditions. By understanding these patterns, the telecom provider can implement targeted improvements to enhance overall service quality and user satisfaction.
