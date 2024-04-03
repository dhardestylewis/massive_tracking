# Exploring Human Mobility Patterns using Call Detail Records (CDRs)

This repository contains code and data for exploring human mobility patterns in Milan, Italy using Call Detail Records (CDRs) from December 3, 2013. The project aims to gain insights into urban dynamics and inform transportation planning decisions through the analysis of large-scale, anonymized mobile phone data.

## Dataset

The primary dataset used in this project is the Telecom Italia Big Data Challenge dataset, which includes:

- Call Detail Records (CDRs) for Milan on December 3, 2013 (`sms-call-internet-mi-2013-12-03.txt`)
- Milano Grid file (`milano-grid.geojson`) for spatial aggregation

The CDR data contains information about SMS messages, calls, and internet usage for each anonymized user, along with timestamps and approximate locations based on cell tower IDs.

## Repository Structure

```
.
├── data/
│   ├── administrative_regions_Milano.json
│   ├── milano-grid.geojson
│   └── sms-call-internet-mi-2013-12-03.txt
├── docs/
│   └── sdata201555.pdf
├── results/
│   ├── aggregate_statistics.csv
│   ├── descriptive_statistics.csv
│   ├── hourly_internet_activity.png
│   ├── internet_activity_stats.csv
│   ├── internet_traffic_volume.csv
│   ├── temporal_patterns.png
│   ├── temporal_patterns_excluding_internet.png
│   └── user_activity_distribution.png
└── src/
    ├── analyze_cdr_data.py
    └── methodology.py
```

- `data/`: Contains the input datasets (CDRs, Milano Grid) and additional data files
- `docs/`: Includes relevant documentation, such as the research paper describing the dataset
- `results/`: Stores the output files generated during the analysis, including statistics, visualizations, and processed data
- `src/`: Contains the source code for data preprocessing, analysis, and visualization

## Analysis

The main analysis steps performed in this project include:

1. Data preprocessing:
   - Cleaning and filtering the CDR data
   - Ensuring format consistency and handling missing values
   - Merging CDR data with cell tower locations

2. Descriptive statistics:
   - Calculating aggregate statistics for SMS, call, and internet usage
   - Analyzing the distribution of user activity levels

3. Temporal pattern analysis:
   - Visualizing hourly variations in SMS, call, and internet activity
   - Identifying peak hours and comparing different activity types

4. Spatial analysis (partially completed):
   - Identifying users' home locations based on nighttime activity
   - Segmenting CDRs into trips (work in progress)
   - Analyzing internet traffic volume patterns spatially and animating them over time (planned)

The code for these analysis steps can be found in the `src/` directory, with `analyze_cdr_data.py` being the main script.

## Results

The `results/` directory contains the output files generated during the analysis, including:

- Descriptive statistics and aggregate statistics for the CDR data
- Visualizations of temporal patterns in user activity
- Preliminary results on internet traffic volume analysis

## Future Work

This project demonstrates the potential of CDR data for understanding human mobility patterns and urban dynamics. However, there are several areas for future improvement and expansion:

- Refining the trip segmentation algorithm to better capture individual trips
- Completing the spatial analysis of internet traffic volume patterns and creating animations
- Integrating additional data sources, such as transportation networks or land use data, to enrich the analysis
- Extending the analysis to cover a longer time period or compare different cities

## References

- Barlacchi, G., De Nadai, M., Larcher, R., Casella, A., Chitic, C., Torrisi, G., ... & Lepri, B. (2015). A multi-source dataset of urban life in the city of Milan and the Province of Trentino. Scientific data, 2(1), 1-15.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
