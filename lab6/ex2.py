import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random
def generate_merged_graph():
    regular_graph = nx.random_regular_graph(d=3, n=100)

    clique_graph = nx.connected_caveman_graph(l=10, k=20)

    merged_graph = nx.union(regular_graph, clique_graph, rename=('A', 'B'))

    regular_nodes = [n for n in merged_graph.nodes() if n.startswith('A')]
    clique_nodes = [n for n in merged_graph.nodes() if n.startswith('B')]
    while not nx.is_connected(merged_graph):
    
        node1 = random.choice(regular_nodes)
        node2 = random.choice(clique_nodes)
        if not merged_graph.has_edge(node1, node2):
            merged_graph.add_edge(node1, node2)

    return merged_graph

def generate_heavyvicinity_graph():
    G1 = nx.random_regular_graph(d=3, n=100)
    G2 = nx.random_regular_graph(d=5, n=100)
    G = nx.union(G1, G2, rename=('C', 'D'))
    nodes = list(G.nodes())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    
    random_nodes = random.sample(nodes, 2)
    
    for node in random_nodes:
        egonet = nx.ego_graph(G, node, radius=1)
        for edge in egonet.edges():
            G[edge[0]][edge[1]]['weight'] += 10
    
    return G, random_nodes

def compute_heavyvicinity_scores(G):
    scores = {}
    for node in G.nodes():
        Ei = G.nodes[node]['Ei']
        Wi = G.nodes[node]['Wi']
        scores[node] = Wi / (Ei + 1) 
    return scores
def extract_features(G):
    for node in G.nodes():
        egonet = nx.ego_graph(G, node)
        neighbor_count = len(egonet) - 1
        edges = egonet.size(weight = None)
        total_weight = egonet.size(weight = 'weight')

        adj = nx.adjacency_matrix(egonet, weight='weight').todense()
        eigenvalues = np.linalg.eigvals(adj)
        p_eig = max(abs(eigenvalues))
        nx.set_node_attributes(G, {node: {
            'Ni': neighbor_count,
            'Ei': edges,
            'Wi': total_weight,
            'lambda' : p_eig
        }})
    return G

def compute_anomaly_scores(G):
    nodes = list(G.nodes())
    X = np.array([G.nodes[n]['Ni'] for n in nodes])
    y = np.array([G.nodes[n]['Ei'] for n in nodes])

    X = np.log(X + 1).reshape(-1,1)
    y = np.log(y + 1).reshape(-1,1)
    
    lr = LinearRegression()
    lr.fit(X,y) 
    y_pred = lr.predict(X)

    scores = {}
    for i, node in enumerate(nodes):

        ratio = max(y[i], y_pred[i]) / min(y[i], y_pred[i])
        scores[node] = ratio * np.log(abs(y[i] - y_pred[i]) + 1)
    
    return scores

def visualize_graph(G, scores, anomaly_count=10):
    top_nodes = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:anomaly_count]
    
    node_colors = ['red' if node in top_nodes else 'blue' for node in G.nodes()]

    plt.figure(figsize=(10, 8))
    nx.draw(G, node_color=node_colors, node_size=100)
    plt.show()

if __name__ == '__main__':
    merged_graph = generate_merged_graph()
    merged_graph = extract_features(merged_graph)
    clique_scores = compute_anomaly_scores(merged_graph)

    visualize_graph(merged_graph, clique_scores, anomaly_count=10)

    heavy_graph, heavy_anomalies = generate_heavyvicinity_graph()
    heavy_graph = extract_features(heavy_graph)
    heavy_scores = compute_heavyvicinity_scores(heavy_graph)

    visualize_graph(heavy_graph, heavy_scores, anomaly_count=4)


