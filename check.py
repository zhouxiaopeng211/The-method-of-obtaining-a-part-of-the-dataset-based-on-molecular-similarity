import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
import re
import random
import number
import similarity
import copy
import matrix_make
import torch.optim as opt
import torch as T
import os
import numpy as np
from torch_geometric.nn import GATConv
from torch.nn import Linear
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import main_train

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Define helper function to extract digits
def extract_digits(s):
    return re.sub(r'^\D+', '', s)


# Define the GNN model class (same as in training script)
class GNN(nn.Module):
    def __init__(self, INPUT):
        super(GNN, self).__init__()
        # Building INPUT
        self.INPUT = INPUT
        # Defining base variables
        self.CHECKPOINT_DIR = self.INPUT["CHECKPOINT_DIR"]
        self.CHECKPOINT_FILE = os.path.join(self.CHECKPOINT_DIR, self.INPUT["NAME"])
        self.SIZE_LAYERS = self.INPUT["SIZE_LAYERS"]
        self.initial_conv = GATConv(self.SIZE_LAYERS[0], self.SIZE_LAYERS[1])
        self.conv1 = GATConv(self.SIZE_LAYERS[1], self.SIZE_LAYERS[2])
        self.conv2 = GATConv(self.SIZE_LAYERS[2], self.SIZE_LAYERS[2])
        self.linear = Linear(self.SIZE_LAYERS[2], self.SIZE_LAYERS[3])
        self.optimizer = opt.Adam(self.parameters(), lr=self.INPUT["LR"])
        self.criterion = nn.MSELoss()

    def forward(self, x, edge_index):  # forward propagation includes defining layers
        out = F.relu(self.initial_conv(x, edge_index=edge_index))
        out = F.relu(self.conv1(out, edge_index=edge_index))
        out = F.relu(self.conv2(out, edge_index=edge_index))
        return self.linear(out)


def test_model(model_path, num_node_features, output_dim=32, test_runs=5):
    """
    Test the model with the given test data

    Args:
        listA: List of molecule identifiers marked as class 0
        listB: List of molecule identifiers marked as class 1
        model_path: Path to the saved model file
        num_node_features: Number of features per node in the graph
        output_dim: Dimension of the output embedding
        test_runs: Number of test runs with random sampling

    Returns:
        Average accuracy, precision, recall, and F1 score across all test runs
    """
    # Define model architecture (same as in training)
    hidden_dim1 = 64
    hidden_dim2 = 64

    active_values, active_0_name, active_1_name = number.extract_active_property("test.sdf")

    # Fill in listA and listB with your data

    gnn_input = {
        "LR": 0.001,  # Not used for inference
        "NAME": "Triplet_Encoder",
        "CHECKPOINT_DIR": "",
        "SIZE_LAYERS": [num_node_features, hidden_dim1, hidden_dim2, output_dim]
    }

    # Initialize models
    encoder = GNN(gnn_input).to(device)
    classifier = nn.Sequential(
        nn.Linear(output_dim, (int(output_dim / 2))),
        nn.ReLU(),
        nn.Linear((int(output_dim / 2)), (int(output_dim / 4))),
        nn.ReLU(),
        nn.Linear((int(output_dim / 4)), 2)
    ).to(device)

    # Load saved model weights
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    # 将模型设置为评估模式
    encoder.eval()
    classifier.eval()

    # 初始化度量累加器
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    # 定义样本大小，根据您的数据进行调整


    print(
        f"Running {test_runs} test iterations with {active_0_name} class 0 samples and {active_1_name} class 1 samples")

    for run in range(test_runs):
        print(f"Test run {run + 1}/{test_runs}")


        # # Randomly sample from both lists
        # A_sample = np.random.choice(listA, sample_size_A, replace=False)
        # B_sample = np.random.choice(listB, sample_size_B, replace=False)

        # Extract digits and convert to integers
        g0_indices = [main_train.extract_digits(t) for t in active_0_name]
        g1_indices = [main_train.extract_digits(t) for t in active_1_name]
        print(g0_indices,'\n',g1_indices)
        print(f"Class 0 samples: {len(g0_indices)}")
        print(f"Class 1 samples: {len(g1_indices)}")

        # Convert indices to graph objects
        g0_test, _, _ = similarity.precompute_all_graphs(g0_indices)
        g1_test, _, _ = similarity.precompute_all_graphs(g1_indices)
        test_graphs = g1_test + g0_test

        if len(test_graphs) == 0:
            print("Warning: Test batch is empty!")
            continue

        # Create batch
        test_batch = Batch.from_data_list(test_graphs).to(device)

        # Create labels (1 for g1_test, 0 for g0_test)
        test_labels = torch.zeros(len(test_graphs), dtype=torch.long, device=device)
        test_labels[:len(g1_test)] = 1

        with torch.no_grad():
            # Forward pass
            h = encoder(test_batch.x, test_batch.edge_index)
            g = global_mean_pool(h, test_batch.batch)
            out = classifier(g)

            # Calculate accuracy
            _, predicted = torch.max(out.data, 1)
            accuracy = (predicted == test_labels).sum().item() / len(test_labels)

            # Calculate precision, recall, and F1 score
            true_positive = ((predicted == 1) & (test_labels == 1)).sum().item()
            false_positive = ((predicted == 1) & (test_labels == 0)).sum().item()
            false_negative = ((predicted == 0) & (test_labels == 1)).sum().item()

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # Accumulate metrics
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            print(
                f"Run {run + 1} metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # Calculate average metrics
    avg_accuracy = total_accuracy / test_runs
    avg_precision = total_precision / test_runs
    avg_recall = total_recall / test_runs
    avg_f1 = total_f1 / test_runs

    print(f"\nFinal Test Results (Average over {test_runs} runs):")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")

    return avg_accuracy, avg_precision, avg_recall, avg_f1


# Example usage
# if __name__ == "__main__":
#     # Replace these with your actual test lists
#     listA = []  # Class 0 molecules
#     listB = []  # Class 1 molecules
#
#     # Fill in listA and listB with your data
#     # listA = ["name1", "name2", ...]
#     # listB = ["name100", "name101", ...]
#
#     # Path to saved model
#     model_path = 'encoder_classifier.pth'
#
#     # Number of features per node (should match training)
#     num_node_features = 0  # Replace with actual node feature dimension
#
#     # Run the test
#     test_model(listA, listB, model_path, num_node_features)