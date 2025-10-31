import json
import gzip
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset
import os
import urllib.request

# ============================================================================
# Amazon数据集下载和加载
# ============================================================================

class AmazonDataLoader:
    """Amazon数据集加载器"""
    
    def __init__(self, category='Beauty', data_dir='./tokenrec_project/data'):
        """
        Args:
            category: 数据集类别 ('Beauty', 'Clothing', etc.)
            data_dir: 数据存储目录
        """
        self.category = category
        self.data_dir = data_dir
        self.raw_data_path = os.path.join(data_dir, f'{category}_5.json.gz')
        
        os.makedirs(data_dir, exist_ok=True)
    
    def download_data(self):
        """下载Amazon数据集"""
        # Amazon数据集URL (5-core版本，每个用户和物品至少有5次交互)
        base_url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
        filename = f"reviews_{self.category}_5.json.gz"
        url = base_url + filename
        
        if os.path.exists(self.raw_data_path):
            print(f"数据集已存在: {self.raw_data_path}")
            return
        
        print(f"正在下载 {self.category} 数据集...")
        print(f"URL: {url}")
        
        try:
            urllib.request.urlretrieve(url, self.raw_data_path)
            print(f"下载完成: {self.raw_data_path}")
        except Exception as e:
            print(f"下载失败: {e}")
            print(f"请手动从以下链接下载数据集:")
            print(f"{url}")
            print(f"并保存到: {self.raw_data_path}")
    
    def parse_data(self):
        """解析JSON格式的数据"""
        print(f"正在解析数据: {self.raw_data_path}")
        
        data = []
        with gzip.open(self.raw_data_path, 'rt', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        df = pd.DataFrame(data)
        print(f"原始数据大小: {len(df)} 条评论")
        
        # 选择需要的列
        df = df[['reviewerID', 'asin', 'overall', 'unixReviewTime']]
        df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
        
        # 过滤低评分 (隐式反馈，只保留评分>=4的)
        df = df[df['rating'] >= 4.0]
        
        print(f"过滤后数据大小: {len(df)} 条交互")
        
        return df
    
    def create_mappings(self, df):
        """创建用户和物品的ID映射"""
        # 用户ID映射
        unique_users = df['user_id'].unique()
        user2id = {user: idx for idx, user in enumerate(unique_users)}
        id2user = {idx: user for user, idx in user2id.items()}
        
        # 物品ID映射
        unique_items = df['item_id'].unique()
        item2id = {item: idx for idx, item in enumerate(unique_items)}
        id2item = {idx: item for item, idx in item2id.items()}
        
        # 转换ID
        df['user_idx'] = df['user_id'].map(user2id)
        df['item_idx'] = df['item_id'].map(item2id)
        
        print(f"用户数量: {len(user2id)}")
        print(f"物品数量: {len(item2id)}")
        
        return df, user2id, id2user, item2id, id2item
    
    def split_data(self, df, test_ratio=0.1, val_ratio=0.1):
        """
        按照leave-one-out策略划分数据集
        - 每个用户的最后一次交互作为测试集
        - 倒数第二次交互作为验证集
        - 其余作为训练集
        """
        # 按时间戳排序
        df = df.sort_values(['user_idx', 'timestamp'])
        
        train_data = []
        val_data = []
        test_data = []
        
        for user_idx in df['user_idx'].unique():
            user_interactions = df[df['user_idx'] == user_idx]
            
            if len(user_interactions) < 3:
                # 如果交互太少，全部放入训练集
                train_data.extend(user_interactions.values.tolist())
            else:
                # 最后一个作为测试
                test_data.append(user_interactions.iloc[-1].values.tolist())
                # 倒数第二个作为验证
                val_data.append(user_interactions.iloc[-2].values.tolist())
                # 其余作为训练
                train_data.extend(user_interactions.iloc[:-2].values.tolist())
        
        columns = df.columns.tolist()
        train_df = pd.DataFrame(train_data, columns=columns)
        val_df = pd.DataFrame(val_data, columns=columns)
        test_df = pd.DataFrame(test_data, columns=columns)
        
        print(f"\n数据划分:")
        print(f"训练集: {len(train_df)} 条交互")
        print(f"验证集: {len(val_df)} 条交互")
        print(f"测试集: {len(test_df)} 条交互")
        
        return train_df, val_df, test_df
    
    def create_interaction_dict(self, df):
        """创建用户-物品交互字典"""
        interactions = defaultdict(list)
        
        for _, row in df.iterrows():
            user_idx = row['user_idx']
            item_idx = row['item_idx']
            timestamp = row['timestamp']
            
            interactions[user_idx].append((item_idx, timestamp))
        
        # 按时间排序
        for user_idx in interactions:
            interactions[user_idx] = sorted(interactions[user_idx], key=lambda x: x[1])
            # 只保留item_idx
            interactions[user_idx] = [item for item, _ in interactions[user_idx]]
        
        return dict(interactions)

    def create_interaction_dict_with_time(self, df):
        """保留时间戳的交互字典: {user_idx: {'items': [...], 'timestamps': [...]}}"""
        interactions = defaultdict(list)
        for _, row in df.iterrows():
            interactions[row['user_idx']].append((row['item_idx'], row['timestamp']))

        result = {}
        for user_idx, pairs in interactions.items():
            pairs = sorted(pairs, key=lambda x: x[1])
            items = [p[0] for p in pairs]
            timestamps = [p[1] for p in pairs]
            result[user_idx] = {'items': items, 'timestamps': timestamps}
        return result
    
    def load_and_preprocess(self):
        """完整的数据加载和预处理流程"""
        # 1. 下载数据
        self.download_data()
        
        # 2. 解析数据
        df = self.parse_data()
        
        # 3. 创建ID映射
        df, user2id, id2user, item2id, id2item = self.create_mappings(df)
        
        # 4. 划分数据集
        train_df, val_df, test_df = self.split_data(df)
        
        # 5. 创建交互字典
        train_interactions = self.create_interaction_dict(train_df)
        val_interactions = self.create_interaction_dict(val_df)
        test_interactions = self.create_interaction_dict(test_df)

        # 5+. 创建保留时间戳的交互字典
        train_interactions_with_time = self.create_interaction_dict_with_time(train_df)
        val_interactions_with_time = self.create_interaction_dict_with_time(val_df)
        test_interactions_with_time = self.create_interaction_dict_with_time(test_df)
        
        # 6. 保存预处理后的数据
        processed_data = {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'train_interactions': train_interactions,
            'val_interactions': val_interactions,
            'test_interactions': test_interactions,
            'train_interactions_with_time': train_interactions_with_time,
            'val_interactions_with_time': val_interactions_with_time,
            'test_interactions_with_time': test_interactions_with_time,
            'user2id': user2id,
            'id2user': id2user,
            'item2id': item2id,
            'id2item': id2item,
            'num_users': len(user2id),
            'num_items': len(item2id)
        }
        
        processed_path = os.path.join(self.data_dir, f'{self.category}_processed.pkl')
        torch.save(processed_data, processed_path)
        print(f"\n预处理数据已保存到: {processed_path}")
        
        return processed_data


# ============================================================================
# 构建用户-物品交互图
# ============================================================================

class BipartiteGraph:
    """二部图表示用户-物品交互"""
    
    def __init__(self, num_users, num_items, train_interactions):
        """
        Args:
            num_users: 用户数量
            num_items: 物品数量
            train_interactions: dict {user_idx: [item_idx1, item_idx2, ...]}
        """
        self.num_users = num_users
        self.num_items = num_items
        self.train_interactions = train_interactions
        
        # 构建边
        self.user_item_edges = []  # [(user_idx, item_idx), ...]
        self.item_user_edges = []  # [(item_idx, user_idx), ...]
        
        for user_idx, items in train_interactions.items():
            for item_idx in items:
                self.user_item_edges.append((user_idx, item_idx))
                self.item_user_edges.append((item_idx, user_idx))
        
        print(f"\n图统计信息:")
        print(f"用户节点数: {num_users}")
        print(f"物品节点数: {num_items}")
        print(f"边数: {len(self.user_item_edges)}")
    
    def to_torch_sparse_coo(self):
        """转换为PyTorch稀疏COO格式的邻接矩阵"""
        # 构建完整的邻接矩阵 (用户和物品都在同一个图中)
        # 用户节点: [0, num_users)
        # 物品节点: [num_users, num_users + num_items)
        
        total_nodes = self.num_users + self.num_items
        
        edges_row = []
        edges_col = []
        
        # 用户->物品的边
        for user_idx, item_idx in self.user_item_edges:
            edges_row.append(user_idx)
            edges_col.append(self.num_users + item_idx)
        
        # 物品->用户的边 (无向图，需要双向边)
        for item_idx, user_idx in self.item_user_edges:
            edges_row.append(self.num_users + item_idx)
            edges_col.append(user_idx)
        
        # 转换为张量
        edge_index = torch.LongTensor([edges_row, edges_col])
        
        # 创建稀疏邻接矩阵
        adj_matrix = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1]),
            (total_nodes, total_nodes)
        )
        
        # 归一化 (对称归一化: D^(-1/2) * A * D^(-1/2))
        adj_matrix = self.normalize_adj_matrix(adj_matrix)
        
        return adj_matrix, edge_index
    
    def normalize_adj_matrix(self, adj_matrix):
        """对称归一化邻接矩阵"""
        # 计算度矩阵
        adj_dense = adj_matrix.to_dense()
        degree = torch.sum(adj_dense, dim=1)
        
        # D^(-1/2)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.
        
        # D^(-1/2) * A * D^(-1/2)
        degree_matrix = torch.diag(degree_inv_sqrt)
        norm_adj = torch.mm(torch.mm(degree_matrix, adj_dense), degree_matrix)
        
        # 转回稀疏矩阵
        norm_adj_sparse = norm_adj.to_sparse()
        
        return norm_adj_sparse


# ============================================================================
# 使用示例
# ============================================================================

def main():
    """数据加载和图构建示例"""
    
    # 1. 加载Amazon-Beauty数据集
    print("=" * 60)
    print("Step 1: 加载Amazon-Beauty数据集")
    print("=" * 60)
    
    loader = AmazonDataLoader(category='Beauty', data_dir='./data')
    data = loader.load_and_preprocess()
    
    # 2. 构建二部图
    print("\n" + "=" * 60)
    print("Step 2: 构建用户-物品二部图")
    print("=" * 60)
    
    graph = BipartiteGraph(
        num_users=data['num_users'],
        num_items=data['num_items'],
        train_interactions=data['train_interactions']
    )
    
    # 3. 转换为稀疏邻接矩阵
    adj_matrix, edge_index = graph.to_torch_sparse_coo()
    
    print(f"\n邻接矩阵形状: {adj_matrix.shape}")
    print(f"边索引形状: {edge_index.shape}")
    print(f"稀疏度: {edge_index.shape[1] / (adj_matrix.shape[0] * adj_matrix.shape[1]):.6f}")
    
    # 4. 保存图数据
    graph_data = {
        'adj_matrix': adj_matrix,
        'edge_index': edge_index,
        'num_users': data['num_users'],
        'num_items': data['num_items']
    }
    
    graph_path = os.path.join(loader.data_dir, 'Beauty_graph.pkl')
    torch.save(graph_data, graph_path)
    print(f"\n图数据已保存到: {graph_path}")
    
    # 5. 数据集统计信息
    print("\n" + "=" * 60)
    print("数据集统计信息")
    print("=" * 60)
    print(f"用户数量: {data['num_users']}")
    print(f"物品数量: {data['num_items']}")
    print(f"训练交互数: {len(data['train_df'])}")
    print(f"验证交互数: {len(data['val_df'])}")
    print(f"测试交互数: {len(data['test_df'])}")
    
    # 计算稀疏度
    total_possible = data['num_users'] * data['num_items']
    actual_interactions = len(data['train_df'])
    sparsity = 1 - (actual_interactions / total_possible)
    print(f"数据稀疏度: {sparsity:.6f}")
    
    # 计算平均交互数
    avg_interactions_per_user = len(data['train_df']) / data['num_users']
    avg_interactions_per_item = len(data['train_df']) / data['num_items']
    print(f"平均每用户交互数: {avg_interactions_per_user:.2f}")
    print(f"平均每物品交互数: {avg_interactions_per_item:.2f}")
    
    return data, graph_data


if __name__ == "__main__":
    data, graph_data = main()