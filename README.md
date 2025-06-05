# Fraud Network Analysis

This project explores how network analysis can be applied to detect fraud within a synthetic transaction dataset. The analysis walks through how to generate a scale-free network, inject a collusive fraud ring, and use graph-based metrics to identify suspicious activity.

## What’s Included

- **`fraud_network_analysis_modular.py`**: A fully modular Python script that runs the entire fraud detection workflow. Includes functions for graph generation, risk scoring, community detection, and visualization.
- **`fraud_network_analysis_final_full_explained_expanded.ipynb`**: A Jupyter notebook version with detailed step-by-step explanations of the analysis, including the reasoning behind each method used.
- **PDF Report**: A PDF export of the notebook that’s suitable for sharing or presentation purposes.

## How It Works

1. **Network Generation**: A synthetic transaction network is built using the Barabási–Albert model, simulating the natural emergence of hubs seen in real-world financial data.
2. **Fraud Injection**: A dense fraud ring is embedded into the network to simulate collusive behavior.
3. **Centrality Metrics**: Each node is scored using a blend of degree, betweenness, closeness, and eigenvector centrality to estimate its potential fraud risk.
4. **Community Detection**: The network is segmented into communities using label propagation to identify suspicious clusters.
5. **Visualization**: The final output includes an interactive graph colored by fraud risk and tabular summaries highlighting high-risk nodes and communities.

## How to Run

To run the analysis from the Python script:

```bash
python fraud_network_analysis_modular.py
```

This will print out the top risky nodes and communities, and launch an interactive Plotly visualization in your browser.

## Requirements

- Python 3.7+
- networkx
- pandas
- numpy
- plotly
- scikit-learn

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Notes

This is a synthetic analysis built to test fraud detection methods using graph theory. If applied to real data, parameters and thresholds should be adjusted based on the domain context.
