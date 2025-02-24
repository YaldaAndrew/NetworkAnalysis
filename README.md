# Network Fraud Detection using Graph Analysis

## Overview

This project applies network analysis techniques to detect potential fraud in transaction data. It builds a transaction graph, identifies suspicious clusters, and applies community detection methods to reveal hidden fraud rings. Centrality measures are also used to highlight influential nodes in the network.

### Features

Graph Construction: Builds a directed transaction network from synthetic data.

Fraud Simulation: Generates normal and fraudulent transactions with a fraud ring.

Centrality Measures: Computes degree and betweenness centrality to identify key nodes.

Community Detection: Applies:

Greedy modularity-based clustering.

Label propagation for automatic community identification.

Visualization: Graph rendering with node colors based on community membership.

### Requirements

To run this project, install the following dependencies:

pip install networkx matplotlib pandas

### Usage

Run the script to generate a transaction network and analyze fraud patterns:

python fraud_network_analysis.py

Interpretation

Nodes represent accounts, and edges represent transactions.

High centrality nodes may indicate key players in a fraud ring.

Community detection methods help identify closely linked fraudulent groups.


