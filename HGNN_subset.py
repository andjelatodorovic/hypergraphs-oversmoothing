import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from config import config
from data import data
import json

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Extract largest connected component of a hypergraph
def extract_largest_connected_hypergraph(features, labels, hypergraph):
    G = nx.Graph()
    for edge, nodes in hypergraph.items():
        node_list = list(nodes)
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                G.add_edge(node_list[i], node_list[j])

    largest_cc = max(nx.connected_components(G), key=len)
    largest_cc = sorted(list(largest_cc))

    node_index_map = {node: idx for idx, node in enumerate(largest_cc)}

    features_subset = features[largest_cc, :]
    labels_subset = labels[largest_cc]

    hypergraph_subset = {}
    for edge, nodes in hypergraph.items():
        filtered_nodes = [node_index_map[node] for node in nodes if node in node_index_map]
        if len(filtered_nodes) > 1:
            hypergraph_subset[edge] = filtered_nodes

    return features_subset, labels_subset, hypergraph_subset

# Load largest connected component of a hypergraph
def load_largest_connected_hypergraph(dataset):
    features_lcc, labels_lcc, hypergraph_lcc = extract_largest_connected_hypergraph(
        dataset['features'], dataset['labels'], dataset['hypergraph']
    )

    dataset_lcc = {
        'n': len(features_lcc),
        'hypergraph': hypergraph_lcc,
        'features': features_lcc,
        'labels': labels_lcc
    }
    return dataset_lcc

# Normalize Laplacian for hypergraph
def normalized_laplacian(V, H):
    H = np.array(H)
    n_edge = H.shape[1]
    W = np.ones(n_edge)
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)
    invDE = np.diag(np.power(DE, -1, where=DE != 0))
    DV_safe = np.where(DV > 0, DV, 1)
    DV2 = np.diag(np.power(DV_safe, -0.5))
    W = np.diag(W)
    H = np.mat(H)
    HT = H.T
    L = DV2 @ H @ W @ invDE @ HT @ DV2
    return torch.Tensor(np.eye(V) - L)

# Normalize Laplacian with exponential transformation
def normalized_laplacian_exponential(n, H, t):
    if t == 0:
        return torch.eye(n, dtype=torch.float32)
    L = normalized_laplacian(n, H)
    eigenvalues, eigenvectors = torch.linalg.eigh(L)

    eigenvalues = torch.clamp(eigenvalues, min=1e-6)
    exp_tLambda = torch.diag(torch.exp(-1 * t * eigenvalues))
    exp_tL = eigenvectors @ exp_tLambda @ eigenvectors.t()
    return exp_tL

# Create hypergraph structure matrix
def create_hypergraph_structure(hypergraph, num_nodes):
    hyperedges = list(hypergraph.values())
    num_hyperedges = len(hyperedges)
    H = torch.zeros((num_nodes, num_hyperedges), dtype=torch.float32)
    for j, edge in enumerate(hyperedges):
        edge_list = list(edge)
        for node in edge_list:
            H[node, j] = 1
    return H

# Hypergraph dataset class
class HypergraphDataset(Dataset):
    def __init__(self, features, labels, V, H, use_exponential=True, t=10):
        self.features = features
        self.labels = labels
        self.V = V
        self.H = H
        if use_exponential:
            self.L = normalized_laplacian_exponential(V, H, t)
        else:
            self.L = normalized_laplacian(V, H)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = torch.tensor(self.features, dtype=torch.float32)
        y = torch.tensor(self.labels, dtype=torch.long)
        L = self.L
        return x, y, L

# HGNN convolution layer
class HGNN_conv(nn.Module):
    def __init__(self, in_ch, out_ch, use_bias=False):
        super(HGNN_conv, self).__init__()
        self.linear = nn.Linear(in_ch, out_ch, bias=use_bias)

    def forward(self, x, L):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if L.dim() == 2:
            L = L.unsqueeze(0)

        x = torch.bmm(L, x)
        x = self.linear(x)
        return x

# HGNN model
class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, hidden_dim, num_layers, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers > 1:
            self.convs.append(HGNN_conv(in_ch, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(HGNN_conv(hidden_dim, hidden_dim))
            self.convs.append(HGNN_conv(hidden_dim, n_class))
        else:
            self.convs.append(HGNN_conv(in_ch, n_class))

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, L):
        for conv in self.convs:
            x = F.relu(conv(x, L))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.softmax(x)
        return x

# Experiment with HGNN
def hgnn_experiment(dataset, args, depth, t, use_exponential=True, patience=10, early_stopping=False):
    V, E = dataset['n'], dataset['hypergraph']
    X, Y = dataset['features'], dataset['labels']
    input_dim, output_dim = X.shape[1], Y.shape[1]
    H = create_hypergraph_structure(E, V)
    dataloader = DataLoader(HypergraphDataset(X, np.argmax(Y, axis=1), V, H, use_exponential, t), batch_size=1)
    gnn_model = HGNN(input_dim, output_dim, args.n_hid, depth, args.dropout)

    L = (normalized_laplacian_exponential(V, H, t) if use_exponential else normalized_laplacian(V, H))

    optimizer = optim.Adam(gnn_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    patience_counter = 0

    best_accuracies_per_epoch = []

    for epoch in range(args.epochs):
        gnn_model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        for x_batch, y_batch, L_batch in dataloader:
            optimizer.zero_grad()
            outputs = gnn_model(x_batch, L_batch)
            outputs = outputs.view(-1, outputs.size(-1))
            y_batch = y_batch.view(-1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            total_samples += x_batch.size(1)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples

        # Track best accuracy for the current epoch
        best_accuracies_per_epoch.append(accuracy)
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        if early_stopping:
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    return accuracy

if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    args = config.parse()
    dataset, train_indices, test_indices = data.load(args)

    dataset_lcc = load_largest_connected_hypergraph(dataset)
    results = []

    for depth in range(2, 21):  # Running experiments for layers 2 to 20
            acc = hgnn_experiment(dataset_lcc, args, depth=depth, t=0, use_exponential=False)        
            results.append(acc)


    # t_values = [0, 1, 10, 100]
    # for t in t_values:
    #     print(f"Running experiment with t={t}.")
    #     best_accuracies = []
    #     for depth in range(2, 21):  # Running experiments for layers 2 to 20
    #         acc = hgnn_experiment(dataset_lcc, args, depth=depth, t=t, use_exponential=True)        
        
    #     results[t] = acc

    # Save results in a JSON file
    with open("best_experiment_results.json", "w") as f:
        json.dump(results, f)

    # Plot the accuracies
    plt.plot(range(2, 21), results, label="use_exponential=False")

    plt.xlabel("Number of Layers")
    plt.ylabel("Accuracy Across Epochs")
    plt.title("Accuracy Across Epochs vs. Number of Layers (No Exponential)")
    plt.legend()
    plt.show()

    # # Plot the best accuracies for each t
    # for t, accuracies in results.items():
    #     plt.plot(range(2, 21), accuracies, label=f"t={t}")
    
    # plt.xlabel("Number of Layers")
    # plt.ylabel("Accuracy Across Epochs")
    # plt.title("Accuracy Across Epochs vs. Number of Layers for different t values")
    # plt.legend()
    # plt.show()
