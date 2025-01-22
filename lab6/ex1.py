import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
def load_graph(path, row_count = 1500):
    G = nx.Graph()
    with open(path, 'r') as file:
        for i,line in enumerate(file):
            if line.startswith('#'):
                continue
            if i >= row_count:
                break
            n1, n2 = map(int, line.strip().split())

            if G.has_edge(n1, n2):
                G[n1][n2]['weight'] += 1
            else:
                G.add_edge(n1,n2, weight=1)
    return G

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

def compute_combined_scores(G, anomaly_dict):
    
    nodes = list(G.nodes())
    features = np.array([[G.nodes[n]['Ei'], G.nodes[n]['Ni']] for n in nodes])
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof.fit(features)
    lof_scores = -lof.score_samples(features)  
    
    anomaly_scores = np.array([anomaly_dict[n] for n in nodes])
    anomaly_scores_normalized = (anomaly_scores - anomaly_scores.mean()) / anomaly_scores.std()
    lof_scores_normalized = (lof_scores - lof_scores.mean()) / lof_scores.std()
    
    combined_scores = {}
    for i, node in enumerate(nodes):
        combined_scores[node] = anomaly_scores_normalized[i] + lof_scores_normalized[i]
    
    return combined_scores

def visualize_graph(G, scores, anomaly_count=10):

    top_nodes = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:anomaly_count]
    
    node_colors = ['red' if node in top_nodes else 'blue' for node in G.nodes()]

    plt.figure(figsize=(10, 8))
    nx.draw(G, node_color=node_colors, node_size=100)
    plt.show()
if __name__ == '__main__':
    G = load_graph('ca-AstroPh.txt')
    G = extract_features(G)
    anomaly_scores = compute_anomaly_scores(G)
    visualize_graph(G, anomaly_scores)
    combined_scores = compute_combined_scores(G, anomaly_scores)
    visualize_graph(G, combined_scores)

