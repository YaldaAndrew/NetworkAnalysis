"""
fraud_network_analysis.py

This script simulates and analyzes a synthetic transaction network for fraud detection. 
It generates a graph, injects a synthetic fraud ring, computes various centrality measures, 
assigns fraud risk scores, detects communities, and visualizes the network.
"""

import datetime
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from networkx.algorithms.community import label_propagation_communities


def generate_transaction_graph(n=300, m=2):
    """
    Generate a scale-free graph using the Barabási–Albert model.

    Parameters:
        n (int): Number of nodes.
        m (int): Number of edges to attach from a new node to existing nodes.

    Returns:
        networkx.Graph: Undirected graph.
    """
    return nx.barabasi_albert_graph(n=n, m=m, seed=42)


def inject_fraud_ring(G, ring_size=10):
    """
    Inject a fully connected subgraph (fraud ring) into the graph.

    Parameters:
        G (networkx.Graph): Base graph.
        ring_size (int): Number of nodes in the fraud ring.

    Returns:
        networkx.Graph: Graph with fraud ring.
        list: List of fraud ring node IDs.
    """
    start_idx = max(G.nodes) + 1
    fraud_ring = list(range(start_idx, start_idx + ring_size))
    G.add_nodes_from(fraud_ring)
    for i in fraud_ring:
        for j in fraud_ring:
            if i != j:
                G.add_edge(i, j)
    return G, fraud_ring


def to_directed_with_attributes(G):
    """
    Convert an undirected graph to a directed graph with edge attributes.

    Parameters:
        G (networkx.Graph): Undirected graph.

    Returns:
        networkx.DiGraph: Directed graph with attributes.
    """
    DG = nx.DiGraph()
    for u, v in G.edges():
        weight = np.random.uniform(10, 1000)
        timestamp = datetime.datetime.now() - datetime.timedelta(days=np.random.randint(0, 365))
        DG.add_edge(u, v, weight=weight, timestamp=timestamp)
    return DG


def compute_risk_scores(G):
    """
    Compute centrality metrics and derive a combined risk score.

    Parameters:
        G (networkx.DiGraph): Directed graph.

    Returns:
        dict: Node risk scores.
    """
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=500)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0 for n in G.nodes()}
    return {
        n: 0.25 * degree[n] + 0.25 * betweenness[n] + 0.25 * closeness[n] + 0.25 * eigenvector[n]
        for n in G.nodes()
    }


def detect_communities(G):
    """
    Detect communities using label propagation.

    Parameters:
        G (networkx.Graph): Undirected graph.

    Returns:
        dict: Mapping of node to community.
    """
    communities = list(label_propagation_communities(G))
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i
    return community_map


def summarize_node_risks(G, risk_scores, community_map):
    """
    Create a summary DataFrame of node risks.

    Parameters:
        G (networkx.DiGraph): Graph.
        risk_scores (dict): Node risk scores.
        community_map (dict): Node to community mapping.

    Returns:
        pandas.DataFrame: Summary table.
    """
    return pd.DataFrame({
        'Node': list(risk_scores.keys()),
        'RiskScore': list(risk_scores.values()),
        'Community': [community_map[n] for n in risk_scores]
    })


def summarize_community_risks(summary_df):
    """
    Summarize risk metrics by community.

    Parameters:
        summary_df (pandas.DataFrame): Node summary.

    Returns:
        pandas.DataFrame: Community summary.
    """
    grouped = summary_df.groupby("Community").agg({
        "RiskScore": ["mean", "max", "count"]
    }).sort_values(("RiskScore", "mean"), ascending=False)
    grouped.columns = ["AvgRisk", "MaxRisk", "MemberCount"]
    return grouped.reset_index()


def visualize_graph(G, risk_scores):
    """
    Visualize the graph with node color mapped to risk score.

    Parameters:
        G (networkx.DiGraph): Directed graph.
        risk_scores (dict): Risk scores.
    """
    pos = nx.spring_layout(G, seed=42)
    for n in G.nodes():
        G.nodes[n]['pos'] = pos[n]

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'),
                            hoverinfo='none', mode='lines')

    node_x, node_y, color = [], [], []
    for n in G.nodes():
        x, y = G.nodes[n]['pos']
        node_x.append(x)
        node_y.append(y)
        color.append(risk_scores[n])

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers',
                            marker=dict(size=10, color=color, colorscale='YlOrRd',
                                        colorbar=dict(title="Risk Score")),
                            text=[f"Node {n}, Risk: {risk_scores[n]:.4f}" for n in G.nodes()],
                            hoverinfo='text')

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title='Network Risk Visualization',
                                     titlefont_size=16,
                                     hovermode='closest',
                                     margin=dict(b=20, l=5, r=5, t=40),
                                     xaxis=dict(showgrid=False, zeroline=False),
                                     yaxis=dict(showgrid=False, zeroline=False)))
    fig.show()


def run_fraud_analysis():
    """
    Run the complete fraud detection workflow.
    """
    G = generate_transaction_graph()
    G, fraud_ring = inject_fraud_ring(G)
    DG = to_directed_with_attributes(G)

    risk_scores = compute_risk_scores(DG)
    nx.set_node_attributes(DG, risk_scores, 'risk_score')

    community_map = detect_communities(DG.to_undirected())
    nx.set_node_attributes(DG, community_map, 'community')

    summary_df = summarize_node_risks(DG, risk_scores, community_map)
    community_summary = summarize_community_risks(summary_df)

    print(summary_df.sort_values(by="RiskScore", ascending=False).head(10))
    print(community_summary.head(10))

    visualize_graph(DG, risk_scores)


if __name__ == "__main__":
    run_fraud_analysis()
