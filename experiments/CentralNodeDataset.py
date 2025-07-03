from graphxai.datasets.dataset import NodeDataset
import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple
import random

class CentralNodeDataset(NodeDataset):
    def __init__(self, 
                 name: str = "CentralNodes",
                 num_hops: int = 2,
                 num_graphs: int = 100,
                 min_nodes: int = 10,
                 max_nodes: int = 50,
                 download: Optional[bool] = True,
                 root: Optional[str] = None):
        
        self.num_graphs = num_graphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.download()
        super().__init__(name=name, num_hops=num_hops, download=download, root=root)

    def download(self):
        """Генерация графов и вычисление центральных узлов"""
        self.graphs = []
        self.explanations = []
        self.num_nodes = 0
        
        for _ in range(self.num_graphs):
            # Генерация случайного графа
            n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
            self.num_nodes += n_nodes
            
            G = self._generate_random_graph(n_nodes)
            
            # Вычисление эксцентриситета и центральных узлов
            eccentricity = nx.eccentricity(G)
            min_eccentricity = min(eccentricity.values())
            central_nodes = [n for n, e in eccentricity.items() if e == min_eccentricity]
            
            # Создание признаков и меток
            x = torch.randn(n_nodes, 5)  # 5-мерные признаки
            x[central_nodes] *= 1.5
            y = torch.zeros(n_nodes, dtype=torch.long)
            y[list(central_nodes)] = 1
            
            # Преобразование в формат PyG
            edge_index = torch.tensor(list(G.edges)).t().contiguous()
            graph = Data(x=x, edge_index=edge_index, y=y)
            
            self.graphs.append(graph)
            self.explanations.extend([
                f"Node {i}: {'Central' if i in central_nodes else 'Non-central'}, "
                f"Eccentricity: {eccentricity[i]}" for i in range(n_nodes)
            ])

        # Объединение всех графов в один большой граф
        self.graph = self._merge_all_graphs()
        
    def _generate_random_graph(self, n_nodes: int) -> nx.Graph:
        """Генерация случайного графа с центральной структурой"""
        graph_type = random.choice([
            'tree', 
            'star', 
            'wheel', 
            'barabasi_albert', 
            'geometric'
        ])
        
        if graph_type == 'tree':
            return nx.random_tree(n_nodes)
        elif graph_type == 'star':
            return nx.star_graph(n_nodes-1)
        elif graph_type == 'wheel':
            return nx.wheel_graph(n_nodes)
        elif graph_type == 'barabasi_albert':
            return nx.barabasi_albert_graph(n_nodes, m=2)
        elif graph_type == 'geometric':
            return nx.random_geometric_graph(n_nodes, radius=0.8)
        
    def _merge_all_graphs(self) -> Data:
        """Объединение всех графов в один большой несвязный граф"""
        x_list = []
        edge_index_list = []
        y_list = []
        node_offset = 0
        
        for graph in self.graphs:
            n_nodes = graph.x.size(0)
            x_list.append(graph.x)
            y_list.append(graph.y)
            
            # Смещение индексов ребер
            edges = graph.edge_index + node_offset
            edge_index_list.append(edges)
            
            node_offset += n_nodes
        
        return Data(
            x=torch.cat(x_list, dim=0),
            edge_index=torch.cat(edge_index_list, dim=1),
            y=torch.cat(y_list, dim=0)
        )

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

    def __len__(self) -> int:
        return self.num_graphs

    def __getitem__(self, idx: int) -> Tuple[Data, str]:
        return self.graphs[idx], self.explanations[idx]
