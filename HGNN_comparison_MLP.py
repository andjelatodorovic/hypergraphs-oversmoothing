import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import wandb
from config import config
from data import data
from torch.utils.data import DataLoader, Dataset
import networkx as nx

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        if len(filtered_nodes) > 1:  # Keep only non-trivial edges
            hypergraph_subset[edge] = filtered_nodes

    return features_subset, labels_subset, hypergraph_subset


def load_largest_connected_hypergraph(dataset):
    
    V = dataset['n']
    E = dataset['hypergraph']
    features = dataset['features']
    labels = dataset['labels']

    features_lcc, labels_lcc, hypergraph_lcc = extract_largest_connected_hypergraph(features, labels, E)

    # Update the dataset dictionary
    dataset_lcc = {
        'n': len(features_lcc),  
        'hypergraph': hypergraph_lcc,
        'features': features_lcc,
        'labels': labels_lcc
    }
    return dataset_lcc

# Updated function to handle set objects
def calculate_hypergraph_statistics(dataset):
    num_nodes = dataset['n']
    hypergraph = dataset['hypergraph']
    features = dataset['features']
    labels = dataset['labels']

    # Number of edges
    num_edges = len(hypergraph)

    # Degree (number of edges per node)
    degrees = np.zeros(num_nodes)
    for edge, nodes in hypergraph.items():
        for node in nodes:
            degrees[node] += 1

    avg_degree = degrees.mean()
    max_degree = degrees.max()
    min_degree = degrees.min()

    # Edge sizes (number of nodes per edge)
    edge_sizes = [len(nodes) for nodes in hypergraph.values()]
    avg_edge_size = np.mean(edge_sizes)
    max_edge_size = np.max(edge_sizes)
    min_edge_size = np.min(edge_sizes)

    # Build a graph representation for component analysis
    G = nx.Graph()
    for edge, nodes in hypergraph.items():
        nodes = list(nodes)  # Convert set to list for indexing
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                G.add_edge(nodes[i], nodes[j])

    num_components = nx.number_connected_components(G)

    statistics = {
        "Number of Nodes": num_nodes,
        "Number of Edges": num_edges,
        "Average Degree": avg_degree,
        "Max Degree": max_degree,
        "Min Degree": min_degree,
        "Average Edge Size": avg_edge_size,
        "Max Edge Size": max_edge_size,
        "Min Edge Size": min_edge_size,
        "Number of Components": num_components,
    }

    return statistics


def display_statistics(statistics):
    print("Dataset Statistics:")
    for key, value in statistics.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")



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

class HGNN_conv(nn.Module):
    def __init__(self, in_ch, out_ch, use_bias=False):  # added `use_bias` parameter
        super(HGNN_conv, self).__init__()
        self.linear = nn.Linear(in_ch, out_ch, bias=use_bias)

        if use_bias and self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x, L):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if L.dim() == 2:
            L = L.unsqueeze(0)

        assert L.size(1) == x.size(1), f"Dimension mismatch: L has {L.size(1)} nodes, x has {x.size(1)} nodes."
        x = torch.bmm(L, x)
        x = self.linear(x)
        return x


class HGNN(nn.Module):

    def __init__(self, in_ch, n_class, hidden_dim, num_layers, dropout=0.5, lambda_param=0.9, use_exponential = False):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.lambda_param = lambda_param
        self.use_exponential = use_exponential
        self.convs = nn.ModuleList()
        self.layer_outputs = []  
        if num_layers > 1:
            self.convs.append(HGNN_conv(in_ch, hidden_dim, use_bias=False))
            for _ in range(num_layers - 2):
                self.convs.append(HGNN_conv(hidden_dim, hidden_dim, use_bias=False))
            self.convs.append(HGNN_conv(hidden_dim, n_class, use_bias=False))
        else:
            self.convs.append(HGNN_conv(in_ch, n_class, use_bias=False))
        
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, L, H, L_prime = None):
        self.layer_outputs = [] 
        dirichlet_energies_L = {}
        dirichlet_energies_L_prime = {}
        spectral_norm_product = 1.0
        E_X_l_minus_1 = 1
        max_spectral_norm = 0
        spectral_norms = []
        H_tensor = torch.tensor(H, dtype=torch.float32) if not isinstance(H, torch.Tensor) else H

        d_v = torch.sum(H_tensor, dim=1)
        non_zero_degree_nodes = torch.where(d_v > 0)[0]
        H = H[non_zero_degree_nodes, :]

        # Determine which Laplacian to use in forward pass
        L_forward = L_prime if (self.use_exponential and L_prime is not None) else L

        for i, conv in enumerate(self.convs):
            # Use the selected Laplacian for the forward pass
            x = F.relu(conv(x, L_forward))
            self.layer_outputs.append(x.clone().detach())  # Store the output after ReLU

            eigenvalues, _ = torch.linalg.eigh(L)
            #print(eigenvalues.sort())
            spectral_gap = torch.min(eigenvalues[eigenvalues > 1e-4])
            
            weight_matrix = conv.linear.weight
            spectral_norm = torch.linalg.norm(weight_matrix, ord=2)
            spectral_norms.append(spectral_norm)
            max_spectral_norm = max(max_spectral_norm, spectral_norm)

            product_sl_lambda = spectral_norm * ((1 - spectral_gap) ** 2)

            alternative_X_l = self.compute_dirichlet_energy_expanded(x, H)
            E_X_l = self.compute_dirichlet_energy(x, L)
            #print(f"XTLX: {alternative_X_l}, Expanded Formula: {E_X_l}")
            #print(f"Layer {i+1}: Energy E(X(l)) = {E_X_l.item()}")

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
            print("Didn't satisfy the corrolary condition", "s =",max_spectral_norm, "lambda_tilde = ", (1 - spectral_gap) ** 2)
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
                    delta_e = d_e[e].item()
                    # can have zero values if hypergraph is not fully connected or has isolated components

                    term = (
                        w_e * h_u_e * h_v_e / delta_e *
                        (x[u] / torch.sqrt(d_v[u] + epsilon) - x[v] / torch.sqrt(d_v[v] + epsilon)).pow(2).sum()
                    )
                    expanded_energy += term

        return expanded_energy
    

def create_hypergraph_structure(hypergraph, num_nodes):
    hyperedges = list(hypergraph.values())
    num_hyperedges = len(hyperedges)
    H = torch.zeros((num_nodes, num_hyperedges), dtype=torch.float32)
    for j, edge in enumerate(hyperedges):
        edge_list = list(edge)
        for node in edge_list:
            H[node, j] = 1
    return H

def normalize_by_sqrt_degree(X, L):
    X = torch.tensor(X) if not isinstance(X, torch.Tensor) else X
    L = torch.tensor(L) if not isinstance(L, torch.Tensor) else L

    degree = L.sum(dim=1)  # Shape (2708,)
    sqrt_degree = torch.sqrt(degree + 1e-10).reshape(-1, 1)

    sqrt_degree_expanded = sqrt_degree.expand_as(X)  

    normalized_X = X / sqrt_degree_expanded
    return normalized_X

def normalized_laplacian_exponential(n, H, t):
    if t == 0:
        return torch.eye(n, dtype=torch.float32)
    L = normalized_laplacian(n, H)
    eigenvalues, eigenvectors = torch.linalg.eigh(L)

    count = eigenvalues.size(0)- np.count_nonzero(eigenvalues)
    eigenvalues = torch.clamp(eigenvalues, min=1e-6)
    exp_tLambda = torch.diag(torch.exp(-1 * t * eigenvalues))
    exp_tL = eigenvectors @ exp_tLambda @ eigenvectors.t()
    return exp_tL


def plot_feature_histograms(layer_outputs, feature_indices=None, bins=30):
    num_layers = len(layer_outputs)
    num_features = layer_outputs[0].shape[-1]
    
    if feature_indices is None:
        feature_indices = np.random.choice(num_features, size=min(10, num_features), replace=False)
    
    plt.figure(figsize=(15, len(feature_indices) * 4))
    
    for i, feature_idx in enumerate(feature_indices):
        for layer_idx, layer_output in enumerate(layer_outputs):
            feature_activations = layer_output[:, feature_idx].cpu().detach().numpy().flatten()
            plt.subplot(len(feature_indices), num_layers, i * num_layers + layer_idx + 1)
            plt.hist(feature_activations, bins=bins, alpha=0.7)
            plt.title(f"Layer {layer_idx + 1} - Feature {feature_idx}")
            plt.xlabel("Activation")
            plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()


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


def is_connected(L):
    eigenvalues = torch.linalg.eigvalsh(L).cpu().numpy()
    print(eigenvalues)
    zero_eigenvalues = np.isclose(eigenvalues, 0, atol=1e-3).sum()
    print(f"Number of eigenvalues close to zero: {zero_eigenvalues}")
    return zero_eigenvalues == 1

def normalized_laplacian_exponential(n, H, t):
    if t == 0:
        return torch.eye(n, dtype=torch.float32)
    L = normalized_laplacian(n, H)
    eigenvalues, eigenvectors = torch.linalg.eigh(L)

    count = eigenvalues.size(0)- np.count_nonzero(eigenvalues)
    eigenvalues = torch.clamp(eigenvalues, min=1e-6)
    exp_tLambda = torch.diag(torch.exp(-1 * t * eigenvalues))
    exp_tL = eigenvectors @ exp_tLambda @ eigenvectors.t()
    return exp_tL

def hgnn_experiment(dataset, train_indices, test_indices, args, depth, t, use_exponential = True, patience=10, early_stopping=False):
    V, E = dataset['n'], dataset['hypergraph']
    X, Y = dataset['features'], dataset['labels']
    input_dim, output_dim = X.shape[1], Y.shape[1]
    H = create_hypergraph_structure(E, V)
    dataset = HypergraphDataset(X, np.argmax(Y, axis=1), V, H)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    gnn_model = HGNN(input_dim, output_dim, args.n_hid, depth, args.dropout, use_exponential = True)

    if use_exponential:
        L = normalized_laplacian_exponential(V, H, t)
    else:
        L = normalized_laplacian(V, H).to(torch.float32)

    print("Check for connectivity", is_connected(L))
    #gnn_model.apply(lambda m: initialize_weights(m, init_type='orthogonal', gain=0.1))
    optimizer = optim.Adam(gnn_model.parameters(), lr=args.lr / 100)
    criterion = nn.CrossEntropyLoss()
    if use_exponential:
        L = normalized_laplacian_exponential(V, H, t)
    else:
        L = normalized_laplacian(V, H).to(torch.float32)

    print(L)
    all_true_labels = []
    all_predicted_labels = []
    best_accuracy = 0.0
    patience_counter = 0
    final_dirichlet_energies_L = None
    final_dirichlet_energies_L_prime = None
    for epoch in range(args.epochs):
        gnn_model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        for x_batch, y_batch, L_batch in dataloader:
            optimizer.zero_grad()
            outputs, dirichlet_energies_L, dirichlet_energies_L_prime, spectral_norm_product = gnn_model(x_batch, L_batch, H)
            outputs = outputs.view(-1, outputs.size(-1))
            y_batch = y_batch.view(-1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
            total_samples += x_batch.size(1)
            _, predicted = torch.max(outputs, 1)
            all_true_labels.extend(y_batch.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())
            correct += (predicted == y_batch).sum().item()
        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        final_dirichlet_energies_L = dirichlet_energies_L
        final_dirichlet_energies_L_prime = dirichlet_energies_L_prime
        if early_stopping:
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
                print(f"Best accuracy updated to {best_accuracy:.4f} at epoch {epoch + 1}")
            else:
                patience_counter += 1
                print(f"No improvement in accuracy. Patience counter: {patience_counter}")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in accuracy for {patience} epochs.")
                break
    print("\nAfter final epoch:")
    print("Dirichlet energies from last epoch L:", final_dirichlet_energies_L)
    print("Dirichlet energies from last epoch L prime:", final_dirichlet_energies_L_prime)
    print(f"True labels: {all_true_labels[:20]}")
    print(f"Predicted labels: {all_predicted_labels[:20]}")

    #plot_x_colorplot(gnn_model.layer_outputs, title="Color plot of X over layers")
    #plot_feature_histograms(gnn_model.layer_outputs, feature_indices=[5, 6, 7, 8, 9])  # Select features for histograms
    #plot_dirichlet_energies(final_dirichlet_energies_L, title="Final Epoch Dirichlet Energy")
    #plot_dirichlet_energies_log(final_dirichlet_energies_L, title="Final Epoch Dirichlet Energies L (Log Scale)")
    #plot_dirichlet_energies_log(final_dirichlet_energies_L_prime, title="Final Epoch Dirichlet Energies L prime (Log Scale)")
    return gnn_model, accuracy, final_dirichlet_energies_L

def plot_dirichlet_energies(energies_dict, title="Dirichlet Energies"):
    layers = sorted(energies_dict.keys())
    layers_to_plot = layers[:-1]
    energies_to_plot = [energies_dict[layer] for layer in layers_to_plot]
    plt.figure(figsize=(10, 6))
    plt.plot(layers_to_plot, energies_to_plot, marker='o', linestyle='-', color='b', label='Dirichlet Energy')
    plt.xlabel('Layer')
    plt.ylabel('Dirichlet Energy')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_dirichlet_energies_log(energies_dict, title="Dirichlet Energies (Log Scale)"):
    layers = sorted(energies_dict.keys())
    layers_to_plot = layers[:-1]
    energies_to_plot = [energies_dict[layer] for layer in layers_to_plot]
    plt.figure(figsize=(10, 6))
    plt.plot(layers_to_plot, energies_to_plot, marker='o', linestyle='-', color='b', label='Dirichlet Energy')
    plt.xlabel('Layer')
    plt.ylabel('Dirichlet Energy (Log Scale)')
    plt.yscale('log')
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()


def plot_dirichlet_energies_log_values(energies_dicts, title="Dirichlet Energies (Log Scale)"):
    plt.figure(figsize=(12, 8))

    for label, energies_dict in energies_dicts.items():
        layers = sorted(energies_dict.keys())
        energies = [energies_dict[layer] for layer in layers]
        plt.scatter(layers, energies, label=label)

    plt.xlabel('Layer')
    plt.ylabel('Dirichlet Energy (Log Scale)')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

class MLP(nn.Module):
    def __init__(self, in_ch, n_class, hidden_dim, num_layers, dropout=0.5):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.dirichlet_energies = []
        
        if num_layers > 1:
            self.layers.append(nn.Linear(in_ch, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, n_class))
        else:
            self.layers.append(nn.Linear(in_ch, n_class))

        self.softmax = nn.LogSoftmax(dim=1)

def hgnn_and_mlp_experiment(dataset, args, depth, t, use_exponential=True):
    V, E = dataset['n'], dataset['hypergraph']
    X, Y = dataset['features'], dataset['labels']
    input_dim, output_dim = X.shape[1], Y.shape[1]
    H = create_hypergraph_structure(E, V)

    # DataLoader and Models
    dataloader = DataLoader(HypergraphDataset(X, np.argmax(Y, axis=1), V, H, use_exponential, t), batch_size=1)
    hgnn_model = HGNN(input_dim, output_dim, args.n_hid, depth, args.dropout, use_exponential = True)
    mlp_model = MLP(input_dim, args.n_hid, output_dim, depth, args.dropout)

    # Laplacian
    L = (normalized_laplacian_exponential(V, H, t) if use_exponential else normalized_laplacian(V, H))

    # Optimizers and Loss
    hgnn_optimizer = optim.Adam(hgnn_model.parameters(), lr=args.lr)
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Dirichlet Energies
    hgnn_dirichlet_energies = {}
    mlp_dirichlet_energies = {}

    for epoch in range(args.epochs):
        hgnn_model.train()
        mlp_model.train()

        hgnn_total_loss = 0
        mlp_total_loss = 0
        correct_hgnn = 0
        correct_mlp = 0

        for x_batch, y_batch, L_batch in dataloader:
            # HGNN Training
            hgnn_optimizer.zero_grad()
            hgnn_outputs = hgnn_model(x_batch, L_batch, H)
            hgnn_loss = criterion(hgnn_outputs.view(-1, hgnn_outputs.size(-1)), y_batch.view(-1))
            hgnn_loss.backward()
            hgnn_optimizer.step()

            hgnn_total_loss += hgnn_loss.item() * x_batch.size(0)
            correct_hgnn += (hgnn_outputs.argmax(dim=1) == y_batch).sum().item()

            # MLP Training
            mlp_optimizer.zero_grad()
            mlp_outputs = mlp_model(x_batch.squeeze())
            mlp_loss = criterion(mlp_outputs.view(-1, mlp_outputs.size(-1)), y_batch.view(-1))
            mlp_loss.backward()
            mlp_optimizer.step()

            mlp_total_loss += mlp_loss.item() * x_batch.size(0)
            correct_mlp += (mlp_outputs.argmax(dim=1) == y_batch).sum().item()

        print(f"Epoch {epoch + 1}/{args.epochs} - HGNN Loss: {hgnn_total_loss:.4f}, MLP Loss: {mlp_total_loss:.4f}, HGNN Accuracy: {correct_hgnn / len(dataloader.dataset):.4f}, MLP Accuracy: {correct_mlp / len(dataloader.dataset):.4f}")

        hgnn_dirichlet_energies[epoch + 1] = hgnn_total_loss
        mlp_dirichlet_energies[epoch + 1] = mlp_total_loss

    # Plot Dirichlet Energies
    plot_dirichlet_energies_log_values(
        {
            "HGNN": hgnn_dirichlet_energies,
            "MLP": mlp_dirichlet_energies
        },
        "Dirichlet Energies Comparison (Log Scale)"
    )



if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    args = config.parse()
    dataset, train_indices, test_indices = data.load(args)
    num_layers = 10

    # Extract the largest connected hypergraph
    dataset_lcc = load_largest_connected_hypergraph(dataset)

    # Store results for comparison
    dirichlet_energies_experiments = {}

    hgnn_and_mlp_experiment(dataset_lcc, args, depth=10, t=0, use_exponential=True)

