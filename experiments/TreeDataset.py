from graphxai.datasets.dataset import GraphDataset
import torch
import networkx as nx
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random 


class TreeDataset(GraphDataset):
    def __init__(self, split_sizes=(0.7, 0.2, 0.1), seed=None, device=None, num_samples_per_class=1000):
        self.num_samples = num_samples_per_class
        self.download()  # Генерация данных перед вызовом родительского конструктора
        super().__init__(name="Tree_NonTree", split_sizes=split_sizes, seed=seed, device=device)
    
    def download(self):
        """Генерация деревьев и не-деревьев с метками"""
        self.graphs = []
        self.explanations = []
        
        # Генерация деревьев (класс 1)
        for _ in range(self.num_samples):
            n_nodes = random.randint(3, 20)
            G = nx.random_tree(n_nodes)
            self._add_graph(G, label=1, node_type="tree")

        # Генерация не-деревьев (класс 0)
        for _ in range(self.num_samples):
            # Выбираем случайный тип не-дерева
            graph_type = random.choice(["cyclic", "complete", "disconnected", "random"])
            G = self._generate_non_tree(graph_type)
            self._add_graph(G, label=0, node_type=graph_type)
        
        random.shuffle(self.graphs)

    def _generate_non_tree(self, graph_type):
        """Генерация различных типов не-деревьев"""
        n_nodes = random.randint(3, 20)
        
        if graph_type == "cyclic":
            # Циклический граф с дополнительными ребрами
            G = nx.cycle_graph(n_nodes)
            if n_nodes >= 3:
                G.add_edge(0, random.randint(2, n_nodes-1))
        elif graph_type == "complete":
            # Полный граф
            G = nx.complete_graph(n_nodes)
        elif graph_type == "disconnected":
            # Несвязный граф (две компоненты)
            G = nx.Graph()
            G.add_edges_from(nx.random_tree(n_nodes//2).edges())
            G.add_edges_from(nx.random_tree(n_nodes - n_nodes//2).edges())
        else:  # random
            # Случайный граф с высокой плотностью
            G = nx.erdos_renyi_graph(n_nodes, p=0.5)
            while nx.is_tree(G):  # Проверка, что не дерево
                G = nx.erdos_renyi_graph(n_nodes, p=0.5)
        
        return G

    def _add_graph(self, G, label, node_type):
        """Добавление графа в датасет"""
        # Конвертация в формат PyTorch Geometric
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        x = torch.randn(G.number_of_nodes(), 5)  # Случайные признаки
        
        # Создание объекта Data
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([label]))
        
        self.graphs.append(data)
        self.explanations.append(
            f"{'Tree' if label==0 else 'Non-tree'} " +
            f"({node_type}) with {G.number_of_nodes()} nodes " +
            f"and {G.number_of_edges()} edges"
        )

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)