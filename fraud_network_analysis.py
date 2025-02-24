#!/usr/bin/env python3
"""
Fraud Network Analysis with Label Propagation

This script creates a synthetic dataset of transactions between accounts.
It simulates normal transactions among a set of accounts and introduces a fraud ring,
which is a group of accounts with high interconnectivity that might be suspicious.
The script builds a directed graph from the transactions, computes network centrality
metrics, detects communities using both a greedy modularity algorithm and a label
propagation algorithm, and visualizes the graph with nodes colored by community membership.

Requirements:
    - networkx
    - matplotlib
    - pandas
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random

def generate_normal_transactions(num_accounts=100, num_transactions=300):
    """
    Generates a list of normal (non-fraudulent) transactions.
    
    Each transaction is represented as a dictionary with:
      - source: originating account
      - target: destination account
      - amount: simulated transaction amount
      - fraud: boolean flag (False for normal transactions)
    """
    transactions = []
    for _ in range(num_transactions):
        src = random.randint(0, num_accounts - 1)
        tgt = random.randint(0, num_accounts - 1)
        if src == tgt:
            continue  # skip self-transactions
        amount = round(random.uniform(10, 1000), 2)
        transactions.append({
            'source': src,
            'target': tgt,
            'amount': amount,
            'fraud': False
        })
    return transactions

def generate_fraud_ring(ring_nodes):
    """
    Generates transactions for a simulated fraud ring.
    
    The fraud ring is a tightly connected group where most accounts interact with each other.
    Fraudulent transactions are flagged with fraud=True.
    
    Args:
        ring_nodes (list): List of account IDs that form the fraud ring.
        
    Returns:
        List of transaction dictionaries.
    """
    fraud_transactions = []
    # Create edges between every pair of nodes in the fraud ring
    for i in range(len(ring_nodes)):
        for j in range(i + 1, len(ring_nodes)):
            # To simulate high connectivity, include the edge with high probability
            if random.random() < 0.8:
                amount = round(random.uniform(500, 2000), 2)
                fraud_transactions.append({
                    'source': ring_nodes[i],
                    'target': ring_nodes[j],
                    'amount': amount,
                    'fraud': True
                })
                fraud_transactions.append({
                    'source': ring_nodes[j],
                    'target': ring_nodes[i],
                    'amount': amount,
                    'fraud': True
                })
    return fraud_transactions

def create_graph(transactions):
    """
    Creates a directed graph from the list of transactions.
    
    Each transaction is added as a directed edge with associated attributes.
    """
    G = nx.DiGraph()
    for txn in transactions:
        src = txn['source']
        tgt = txn['target']
        # Use edge attributes to store the amount and fraud flag
        G.add_edge(src, tgt, amount=txn['amount'], fraud=txn['fraud'])
    return G

def analyze_graph(G):
    """
    Performs network analysis on the graph.
    
    This function computes:
      - Degree centrality: to identify influential nodes.
      - Betweenness centrality: to detect nodes that act as bridges.
      - Community detection:
            * Greedy modularity communities (on undirected graph)
            * Asynchronous label propagation communities (on undirected graph)
    
    Returns:
        A tuple containing:
          - degree_centrality
          - betweenness_centrality
          - communities_greedy (from greedy modularity)
          - communities_label (from label propagation)
    """
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Convert to undirected graph for community detection
    G_undirected = G.to_undirected()
    
    # Community detection using greedy modularity maximization
    communities_greedy = list(nx.algorithms.community.greedy_modularity_communities(G_undirected))
    
    # Community detection using asynchronous label propagation
    communities_label = list(nx.algorithms.community.asyn_lpa_communities(G_undirected))
    
    return degree_centrality, betweenness_centrality, communities_greedy, communities_label

def visualize_graph(G, communities, title):
    """
    Visualizes the graph using matplotlib.
    
    Nodes are colored based on their community membership.
    
    Args:
        G: The graph to visualize.
        communities: A list of sets, where each set is a community of nodes.
        title: Title of the plot.
    """
    pos = nx.spring_layout(G, seed=42)  # for reproducible layout
    
    # Map each node to its community index
    community_map = {}
    for i, community in enumerate(communities):
        for node in community:
            community_map[node] = i
    
    node_colors = [community_map.get(node, 0) for node in G.nodes()]
    
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.Set1, node_size=300)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    # Generate normal transaction data
    transactions = generate_normal_transactions(num_accounts=100, num_transactions=300)
    
    # Simulate a fraud ring using a set of account IDs (e.g., 100-105)
    fraud_ring_nodes = [100, 101, 102, 103, 104, 105]
    fraud_transactions = generate_fraud_ring(fraud_ring_nodes)
    
    # Combine normal and fraud transactions
    transactions.extend(fraud_transactions)
    
    # Optionally, load into a DataFrame to inspect the sample data
    df = pd.DataFrame(transactions)
    print("Sample Transaction Data:")
    print(df.head())
    
    # Create a graph from the transactions
    G = create_graph(transactions)
    
    # Analyze the graph
    degree_centrality, betweenness_centrality, communities_greedy, communities_label = analyze_graph(G)
    
    # Display some centrality metrics (for demonstration purposes)
    print("\nDegree Centrality (first 10 nodes):")
    for node, centrality in list(degree_centrality.items())[:10]:
        print(f"Node {node}: {centrality:.3f}")
        
    print("\nBetweenness Centrality (first 10 nodes):")
    for node, centrality in list(betweenness_centrality.items())[:10]:
        print(f"Node {node}: {centrality:.3f}")
    
    # Display detected communities from Greedy Modularity
    print("\nDetected Communities (Greedy Modularity):")
    for i, community in enumerate(communities_greedy):
        print(f"Community {i}: {sorted(community)}")
    
    # Display detected communities from Label Propagation
    print("\nDetected Communities (Label Propagation):")
    for i, community in enumerate(communities_label):
        print(f"Community {i}: {sorted(community)}")
    
    # Visualize the network with communities detected by Greedy Modularity
    visualize_graph(G, communities_greedy, "Fraud Network Analysis - Greedy Modularity Communities")
    
    # Visualize the network with communities detected by Label Propagation
    visualize_graph(G, communities_label, "Fraud Network Analysis - Label Propagation Communities")

if __name__ == '__main__':
    main()
