import networkx as nx
from node2vec import Node2Vec
import torch
import torch.nn as nn

class Node2VecGraph:
    def _init_(self, num_nodes=100, dimensions=64, walk_length=30, num_walks=200, p=1, q=1):
        self.num_nodes = num_nodes
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q

    def generate_graph(self):
        # Create a random graph with exactly 100 nodes
        graph = nx.erdos_renyi_graph(self.num_nodes, 0.5)
        
        # Precompute probabilities and generate walks
        node2vec = Node2Vec(graph, dimensions=self.dimensions, walk_length=self.walk_length, 
                            num_walks=self.num_walks, p=self.p, q=self.q, workers=4)
        
        # Embed nodes
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        # Get embeddings for each node
        embeddings = model.wv.vectors
        
        return torch.tensor(embeddings, dtype=torch.float32)

    def forward(self, x):
        relu = nn.ReLU()
        return relu(x)

# Example usage
