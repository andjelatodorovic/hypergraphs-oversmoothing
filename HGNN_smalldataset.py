import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import ConvexHull
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_connected_hypergraph(num_nodes=100, num_hyperedges=30, min_nodes_per_edge=5, max_nodes_per_edge=20):
    hypergraph = {0: list(range(min_nodes_per_edge))}
    used_nodes = set(hypergraph[0])

    for i in range(1, num_hyperedges):
        overlap_sample_size = min(len(used_nodes), random.randint(1, min_nodes_per_edge))
        overlap_nodes = random.sample(used_nodes, overlap_sample_size)
        
        remaining_new_nodes_needed = random.randint(min_nodes_per_edge - len(overlap_nodes), max_nodes_per_edge - len(overlap_nodes))
        available_new_nodes = [n for n in range(num_nodes) if n not in used_nodes]
        new_nodes_needed = min(remaining_new_nodes_needed, len(available_new_nodes))
        
        new_nodes = random.sample(available_new_nodes, new_nodes_needed)
        used_nodes.update(new_nodes)
        
        hypergraph[i] = overlap_nodes + new_nodes

        if len(used_nodes) == num_nodes:
            break

    if len(used_nodes) < num_nodes:
        remaining_nodes = [n for n in range(num_nodes) if n not in used_nodes]
        hypergraph[len(hypergraph) - 1].extend(remaining_nodes)
    
    return hypergraph

def generate_connected_hypergraph_lambda(num_nodes=100, num_hyperedges=30, 
                                         min_nodes_per_edge=5, max_nodes_per_edge=20, 
                                         max_lambda=0.5, tolerance=1e-2):
    while True:
        # Generate a hypergraph
        hypergraph = {0: list(range(min_nodes_per_edge))}
        used_nodes = set(hypergraph[0])

        for i in range(1, num_hyperedges):
            overlap_sample_size = min(len(used_nodes), random.randint(1, min_nodes_per_edge))
            overlap_nodes = random.sample(used_nodes, overlap_sample_size)
            
            remaining_new_nodes_needed = random.randint(
                min_nodes_per_edge - len(overlap_nodes),
                max_nodes_per_edge - len(overlap_nodes),
            )
            available_new_nodes = [n for n in range(num_nodes) if n not in used_nodes]
            new_nodes_needed = min(remaining_new_nodes_needed, len(available_new_nodes))
            
            new_nodes = random.sample(available_new_nodes, new_nodes_needed)
            used_nodes.update(new_nodes)
            
            hypergraph[i] = overlap_nodes + new_nodes

            if len(used_nodes) == num_nodes:
                break

        if len(used_nodes) < num_nodes:
            remaining_nodes = [n for n in range(num_nodes) if n not in used_nodes]
            hypergraph[len(hypergraph) - 1].extend(remaining_nodes)

        # Create Laplacian and compute largest eigenvalue
        H = create_hypergraph_structure(hypergraph, num_nodes)
        L = normalized_laplacian(num_nodes, H).to(torch.float32)
        eigenvalues, _ = torch.linalg.eigh(L)
        largest_eigenvalue = torch.max(eigenvalues).item()

        # Check if condition is satisfied
        if largest_eigenvalue < max_lambda + tolerance:
            print(f"Condition satisfied: λ = {largest_eigenvalue:.4f}")
            return hypergraph
        else:
            print(f"Retrying: λ = {largest_eigenvalue:.4f}")



def visualize_connected_hypergraph(hypergraph, num_nodes=100):
    plt.figure(figsize=(15, 15))
    G = nx.Graph()

    connected_nodes = set()
    for edge in hypergraph.values():
        for node in edge:
            connected_nodes.add(node)
            G.add_node(node)

    for edge_nodes in hypergraph.values():
        for i in range(len(edge_nodes)):
            for j in range(i + 1, len(edge_nodes)):
                G.add_edge(edge_nodes[i], edge_nodes[j])

    positions = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos=positions, node_color='lightblue', node_size=100)
    nx.draw_networkx_labels(G, pos=positions, font_size=8)

    for i, nodes in hypergraph.items():
        node_positions = np.array([positions[node] for node in nodes if node in connected_nodes])
        if len(node_positions) > 2:  
            hull = ConvexHull(node_positions)
            polygon_points = node_positions[hull.vertices]
            plt.fill(*zip(*polygon_points), alpha=0.3, color=np.random.rand(3,), edgecolor='black', linewidth=1.5)

    nx.draw_networkx_edges(G, pos=positions, edge_color='gray', alpha=0.5)

    plt.title("Connected Hypergraph Visualization with Convex Hulls for Hyperedges")
    plt.axis('equal')
    plt.show()


def generate_synthetic_data(num_nodes, num_features, num_classes):
    X = np.random.rand(num_nodes, num_features)
    Y = np.random.randint(0, num_classes, size=(num_nodes,))
    return X, Y

class HypergraphDataset(Dataset):
    def __init__(self, features, labels, V, H):
        self.features = features
        self.labels = labels
        self.V = V
        self.H = H
        self.L = normalized_laplacian(V, H)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = torch.tensor(self.features, dtype=torch.float32)
        y = torch.tensor(self.labels, dtype=torch.long)
        L = self.L
        return x, y, L

class HGNN_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_conv, self).__init__()
        self.linear = nn.Linear(in_ch, out_ch, bias=False)

    def forward(self, x, L):
        x = torch.bmm(L, x)
        x = self.linear(x)
        return x

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, hidden_dim, num_layers, dropout=0.5):
        super(HGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(HGNN_conv(in_ch, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(HGNN_conv(hidden_dim, hidden_dim))
        self.convs.append(HGNN_conv(hidden_dim, n_class))
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = dropout

    def forward(self, x, L, H):
        self.layer_outputs = [] 
        dirichlet_energies_L = {}
        dirichlet_energies_L_prime = {}
        spectral_norm_product = 1.0
        E_X_l_minus_1 = 1
        max_spectral_norm = 0
        spectral_norms = []

        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, L))
            self.layer_outputs.append(x.clone().detach())  # Store the output after ReLU
            eigenvalues, _ = torch.linalg.eigh(L)
            #print(eigenvalues.sort())
            spectral_gap = torch.min(eigenvalues[eigenvalues > 1e-10])

            weight_matrix = conv.linear.weight
            spectral_norm = torch.linalg.norm(weight_matrix, ord=2)
            spectral_norms.append(spectral_norm)
            max_spectral_norm = max(max_spectral_norm, spectral_norm)

            product_sl_lambda = spectral_norm * ((1 - spectral_gap) ** 2)

            alternative_X_l = self.compute_dirichlet_energy_expanded(x, H)
            E_X_l = self.compute_dirichlet_energy(x, L)
            print(f"XTLX: {alternative_X_l}, Expanded Formula: {E_X_l}")
            print(f"Layer {i+1}: Energy E(X(l)) = {E_X_l.item()}")

            if i > 0:
                if E_X_l > product_sl_lambda * E_X_l_minus_1:
                    print(f"Theorem condition violated at layer {i+1}")

            E_X_l_minus_1 = E_X_l
            dirichlet_energies_L[i + 1] = E_X_l

            if i < len(self.convs) - 1:
                x = F.dropout(x, self.dropout, training=self.training)

        if max_spectral_norm * ((1 - spectral_gap) ** 2) < 1:
            print("Corollary condition satisfied")
        else:
            print("Didn't satisfy the corrolary condition", "s =",max_spectral_norm, "lambda_tilede = ", (1 - spectral_gap) ** 2)
        x = self.softmax(x)
        return x, dirichlet_energies_L, dirichlet_energies_L_prime, spectral_norm_product

    @torch.no_grad()
    def compute_dirichlet_energy(self, x, L):
        if x.dim() == 3:
            x = x.squeeze(0)
        if L.dim() == 3:
            L = L.squeeze(0)

        d_energy = torch.matmul(torch.matmul(x.T, L), x)
        energy = torch.trace(d_energy)
        normalized_energy = energy / torch.trace(x.T @ x)  # Normalized energy
        return energy

    @torch.no_grad()
    def compute_dirichlet_energy_expanded(self, x, H, epsilon=1e-10):
        H_tensor = torch.tensor(H, dtype=torch.float32) if not isinstance(H, torch.Tensor) else H
        n_nodes = H_tensor.shape[0]
        n_edge = H_tensor.shape[1]

        if x.dim() == 3:
            x = x.squeeze(0)

        W = torch.ones(n_edge)
        d_v = torch.sum(H_tensor, dim=1)
        d_e = torch.sum(H_tensor, dim=0)
        expanded_energy = 0.0

        for e in range(n_edge):
            nodes_in_edge = torch.where(H_tensor[:, e] > 0)[0]
            for i in range(len(nodes_in_edge)):
                for j in range(i + 1, len(nodes_in_edge)):
                    u, v = int(nodes_in_edge[i]), int(nodes_in_edge[j])
                    w_e = W[e].item()
                    h_u_e = H_tensor[u, e]
                    h_v_e = H_tensor[v, e]
                    delta_e = d_e[e].item() + epsilon

                    term = (
                        w_e * h_u_e * h_v_e / delta_e *
                        (x[u] / torch.sqrt(d_v[u] + epsilon) - x[v] / torch.sqrt(d_v[v] + epsilon)).pow(2).sum()
                    )
                    expanded_energy += term

        return expanded_energy


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

# Connectivity check function based on Laplacian eigenvalues
def is_connected(L):
    eigenvalues = torch.linalg.eigvalsh(L).cpu().numpy()
    zero_eigenvalues = np.isclose(eigenvalues, 0, atol=1e-6).sum()
    print(f"Number of eigenvalues close to zero: {zero_eigenvalues}")
    return zero_eigenvalues == 1



def create_hypergraph_structure(hypergraph, num_nodes):
    hyperedges = list(hypergraph.values())
    num_hyperedges = len(hyperedges)
    H = torch.zeros((num_nodes, num_hyperedges), dtype=torch.float32)
    for j, edge in enumerate(hyperedges):
        edge_list = list(edge)
        for node in edge_list:
            H[node, j] = 1
    return H



def hgnn_experiment(dataset, num_layers, n_hid=64, dropout=0.5, lr=0.01, epochs=100):
    V, E = dataset['n'], dataset['hypergraph']
    X, Y = dataset['features'], dataset['labels']
    input_dim, output_dim = X.shape[1], Y.max() + 1
    H = create_hypergraph_structure(E, V)

    dataset = HypergraphDataset(X, Y, V, H)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    gnn_model = HGNN(input_dim, output_dim, n_hid, num_layers)

    L = normalized_laplacian(V, H).to(torch.float32)

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr, weight_decay=1e-4) 
    #weight initialization or weight regularization helps here in controling the sk parameter
    #optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    last_epoch_dirichlet_energies = []  # Store energies for the last epoch


    for epoch in range(epochs):
        gnn_model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        for x_batch, y_batch, L_batch in dataloader:
            optimizer.zero_grad()
            
            outputs, dirichlet_L, _, _ = gnn_model(x_batch, L_batch, H)
            outputs = outputs.view(-1, outputs.size(-1))
            y_batch = y_batch.view(-1)
            loss = criterion(outputs, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            total_samples += x_batch.size(1)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()

            # Collect Dirichlet energies for the last epoch
            if epoch == epochs - 1:  # Only in the last epoch
                last_epoch_dirichlet_energies = [e.item() for e in dirichlet_L.values()]

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Plot Dirichlet Energies for the last epoch (excluding the last layer)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(last_epoch_dirichlet_energies)),  # Exclude the last layer
            last_epoch_dirichlet_energies[:-1],  # Slicing to exclude the last value
            marker='o', label="Standard Dirichlet Energy (Excl. Last Layer)", color='b')

    plt.xlabel("Layer")
    plt.ylabel("Dirichlet Energy")
    plt.title("Dirichlet Energies in the Last Epoch (Excluding Last Layer)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return gnn_model, accuracy


set_seed(42)
num_nodes = 100
num_features = 16
num_classes = 2
num_layers = 10

X, Y = generate_synthetic_data(num_nodes, num_features, num_classes)
hypergraph = generate_connected_hypergraph(num_nodes=num_nodes, num_hyperedges=20)


visualize_connected_hypergraph(hypergraph, num_nodes=num_nodes)

dataset = {
    'n': num_nodes,
    'hypergraph': hypergraph,
    'features': X,
    'labels': Y
}
hgnn_experiment(dataset, num_layers=num_layers)
