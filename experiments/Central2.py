from sklearn.model_selection import train_test_split
import torch
import random
import numpy as np
import networkx as nx
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx

from graphxai.datasets.dataset import NodeDataset
from graphxai.utils import Explanation

class CentralNodeDataset(NodeDataset):
    '''
    Dataset class for graphs with labeled central nodes.
    
    Args:
        num_graphs (int): Number of graphs to generate
        min_nodes (int): Minimum number of nodes in each graph
        max_nodes (int): Maximum number of nodes in each graph
        feature_dim (int): Dimension of node features
        model_layers (int): Number of layers within the GNN that will be explained.
            This defines the extent of the ground-truth explanations. (:default: :obj:`3`)
        seed (int, optional): Seed for random operations. (:default: `None`)
        make_explanations (bool, optional): Whether to generate explanations for nodes. 
            (:default: `True`)
    '''

    def __init__(self,
        num_graphs: int,
        min_nodes: int = 10,
        max_nodes: int = 50,
        feature_dim: int = 16,
        model_layers: int = 3,
        seed: Optional[int] = None,
        make_explanations: Optional[bool] = True,
        **kwargs):
        
        super().__init__(name='CentralNodeDataset', num_hops=model_layers)
        
        self.model_layers = model_layers
        self.seed = seed
        self.make_explanations = make_explanations
        
        # Set random seed if provided
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        
        # Generate graphs and central nodes
        self.graphs, self.central_nodes = self.generate_graphs(
            num_graphs=num_graphs,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            feature_dim=feature_dim
        )
        
        # Convert graphs to NetworkX format
        self.nx_graphs = [to_networkx(g, to_undirected=True) for g in self.graphs]
        
        # Generate explanations if requested
        if self.make_explanations:
            self.__generate_explanations()
    
    def generate_graphs(self, 
                       num_graphs: int,
                       min_nodes: int,
                       max_nodes: int,
                       feature_dim: int) -> Tuple[List[Data], List[int]]:
        """
        Generate a set of graphs with central nodes.
        
        Args:
            num_graphs (int): Number of graphs to generate
            min_nodes (int): Minimum number of nodes in each graph
            max_nodes (int): Maximum number of nodes in each graph
            feature_dim (int): Dimension of node features
            
        Returns:
            Tuple[List[Data], List[int]]: List of generated graphs and their central nodes
        """
        graphs = []
        central_nodes = []
        
        for _ in range(num_graphs):
            # Randomly choose number of nodes for this graph
            num_nodes = random.randint(min_nodes, max_nodes)
            
            # Generate random node features
            x = torch.randn(num_nodes, feature_dim)
            
            # Generate a random graph using preferential attachment
            # This creates a graph where some nodes are more central than others
            G = nx.barabasi_albert_graph(num_nodes, m=2)
            
            # Convert to edge index format
            edge_index = torch.tensor(list(G.edges())).t().contiguous()
            
            # Choose a central node (node with highest degree)
            degrees = dict(G.degree())
            central_node = max(degrees.items(), key=lambda x: x[1])[0]
            
            # Create PyTorch Geometric Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                y=torch.zeros(num_nodes, dtype=torch.long),  # All nodes are class 0 by default
                num_nodes=num_nodes
            )
            
            # Mark central node as class 1
            data.y[central_node] = 1
            
            graphs.append(data)
            central_nodes.append(central_node)
        
        return graphs, central_nodes
    
    def __generate_explanations(self):
        '''
        Generate explanations for each central node in the graphs.
        '''
        self.explanations = []
        
        for graph_idx, (graph, central_node) in enumerate(zip(self.graphs, self.central_nodes)):
            # Create an explanation for the current central node
            exp = self.explanation_generator(graph, central_node)
            self.explanations.append([exp])  # Must be a list
    
    def explanation_generator(self, graph: Data, node_idx: int):
        khop_info = k_hop_subgraph(
            node_idx,
            num_hops=self.model_layers,
            edge_index=graph.edge_index,
            relabel_nodes=True
        )
        
        node_imp = torch.zeros(khop_info[0].size(0), dtype=torch.double)
        center_node_idx = 0  # After relabeling, central node is always at index 0
        node_imp[center_node_idx] = 1.0
        
        # Mark direct neighbors as important
        for i in range(khop_info[1].size(1)):
            if khop_info[1][0, i] == center_node_idx or khop_info[1][1, i] == center_node_idx:
                if khop_info[1][0, i] == center_node_idx:
                    node_imp[khop_info[1][1, i]] = 0.5
                else:
                    node_imp[khop_info[1][0, i]] = 0.5
        
        # Edge importance based on direct neighbors
        edge_imp = torch.zeros(khop_info[1].size(1), dtype=torch.double)
        for i in range(khop_info[1].size(1)):
            if khop_info[1][0, i] == center_node_idx or khop_info[1][1, i] == center_node_idx:
                edge_imp[i] = 1.0
        
        # Feature importance - all features equally important
        feature_imp = torch.ones(graph.x.size(1), dtype=torch.double)
        
        # Create explanation object
        exp = Explanation(
            feature_imp=feature_imp,
            node_imp=node_imp,
            edge_imp=edge_imp,
            node_idx=node_idx
        )
        
        # Set the enclosing subgraph
        exp.set_enclosing_subgraph(khop_info)
        
        return exp
    
    def visualize(self, graph_idx: int = 0, ax=None, show=True):
        '''
        Visualize a graph with its central node highlighted.
        
        Args:
            graph_idx (int): Index of the graph to visualize. (:default: :obj:`0`)
            ax (matplotlib.axes, optional): Matplotlib axes to plot on. (:default: :obj:`None`)
            show (bool, optional): Whether to show the plot. (:default: :obj:`True`)
        '''
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        
        G = self.nx_graphs[graph_idx]
        central_node = self.central_nodes[graph_idx]
        
        # Get node positions using spring layout
        pos = nx.spring_layout(G, seed=self.seed or 42)
        
        # Draw the basic graph
        nx.draw(G, pos, node_color='lightblue', 
                node_size=50, alpha=0.8, ax=ax)
        
        # Highlight central node
        nx.draw_networkx_nodes(G, pos, nodelist=[central_node], 
                              node_color='red', node_size=100, ax=ax)
        
        ax.set_title(f'Graph {graph_idx} with Central Node')
        
        if show:
            plt.show()
            
        return ax

    def get_train_loader(self, batch_size=64):
        """
        Returns a DataLoader for training data.
        
        Args:
            batch_size (int): Batch size for the DataLoader.
            
        Returns:
            tuple: (DataLoader, explanations) - The loader and corresponding explanations
        """
        from torch_geometric.loader import DataLoader
        
        data_list = []
        exp_list = []
        
        for graph_idx, (graph, central_node) in enumerate(zip(self.graphs, self.central_nodes)):
            # Get enclosing subgraph
            node_subgraph = self.get_enclosing_subgraph(central_node, graph)
            
            # Create a Data object for this subgraph
            data = Data(
                x=graph.x[node_subgraph.nodes],
                edge_index=node_subgraph.edge_index,
                y=torch.tensor([1]),  # Central node is always class 1
                node_idx=torch.tensor([0]),  # Central node is always at index 0 after relabeling
            )
            
            data_list.append(data)
            
            if self.make_explanations:
                exp_list.append(self.explanations[graph_idx])
        
        loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
        
        return loader, exp_list
        
    def get_test_loader(self):
        """
        Returns a DataLoader for test data.
        
        Returns:
            tuple: (DataLoader, explanations) - The loader and corresponding explanations
        """
        return self.get_train_loader(batch_size=1)
        
    def get_val_loader(self):
        """
        Returns a DataLoader for validation data.
        
        Returns:
            tuple: (DataLoader, explanations) - The loader and corresponding explanations
        """
        return self.get_train_loader(batch_size=1)
    
    def get_graph(self, 
                 use_fixed_split: bool = True, 
                 split_sizes: Tuple = (0.7, 0.2, 0.1),
                 stratify: bool = True, 
                 seed: int = None) -> Data:
        
        if use_fixed_split:
            self._create_fixed_masks(split_sizes, seed, stratify)
        else:
            self._create_random_masks(split_sizes, seed, stratify)
            
        return self.graph

    def _create_fixed_masks(self, split_sizes, seed, stratify):
        """Создание фиксированных масок для разделения"""
        indices = np.arange(self.graph.y.size(0))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=split_sizes[1]+split_sizes[2],
            random_state=seed,
            stratify=self.graph.y.numpy() if stratify else None
        )
        
        val_idx, test_idx = train_test_split(
            test_idx,
            test_size=split_sizes[2]/(split_sizes[1]+split_sizes[2]),
            random_state=seed,
            stratify=self.graph.y[test_idx].numpy() if stratify else None
        )
        
        self.graph.train_mask = self._create_mask(train_idx)
        self.graph.valid_mask = self._create_mask(val_idx)
        self.graph.test_mask = self._create_mask(test_idx)

    def _create_mask(self, indices):
        """Создание бинарной маски из индексов"""
        mask = torch.zeros(self.graph.y.size(0), dtype=torch.bool)
        mask[indices] = True
        return mask