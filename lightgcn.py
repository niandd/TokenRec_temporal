import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os

# ============================================================================
# LightGCN模型实现
# ============================================================================

class LightGCN(nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
    Paper: https://arxiv.org/abs/2002.02126
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        """
        Args:
            num_users: 用户数量
            num_items: 物品数量
            embedding_dim: embedding维度
            num_layers: GNN层数
        """
        super(LightGCN, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # 用户和物品的初始embedding
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Xavier初始化
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        print(f"LightGCN初始化:")
        print(f"  用户数: {num_users}")
        print(f"  物品数: {num_items}")
        print(f"  Embedding维度: {embedding_dim}")
        print(f"  GNN层数: {num_layers}")
    
    def forward(self, adj_matrix):
        """
        前向传播
        Args:
            adj_matrix: 归一化的邻接矩阵 (稀疏张量)
        Returns:
            user_embeddings: (num_users, embedding_dim)
            item_embeddings: (num_items, embedding_dim)
        """
        # 获取初始embedding
        all_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)  # (num_users + num_items, embedding_dim)
        
        # 存储每一层的embedding
        embeddings_list = [all_embeddings]
        
        # 多层图卷积
        for layer in range(self.num_layers):
            # 图卷积: E^(k+1) = (D^(-1/2) * A * D^(-1/2)) * E^(k)
            all_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        # Layer Combination: 平均所有层的embedding
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=0), dim=0)
        
        # 分离用户和物品的embedding
        user_embeddings = final_embeddings[:self.num_users]
        item_embeddings = final_embeddings[self.num_users:]
        
        return user_embeddings, item_embeddings
    
    def get_embedding(self, users, pos_items, neg_items):
        """
        获取特定用户和物品的embedding (用于训练)
        """
        users = users.long()
        pos_items = pos_items.long()
        neg_items = neg_items.long()
        
        user_emb = self.user_embedding(users)
        pos_item_emb = self.item_embedding(pos_items)
        neg_item_emb = self.item_embedding(neg_items)
        
        return user_emb, pos_item_emb, neg_item_emb
    
    def bpr_loss(self, users, pos_items, neg_items, user_embeddings, item_embeddings):
        """
        BPR (Bayesian Personalized Ranking) 损失
        """
        # 获取对应的embedding
        user_emb = user_embeddings[users]
        pos_item_emb = item_embeddings[pos_items]
        neg_item_emb = item_embeddings[neg_items]
        
        # 计算分数
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        
        # BPR损失: -log(sigmoid(pos_score - neg_score))
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        return bpr_loss
    
    def reg_loss(self, users, pos_items, neg_items):
        """
        L2正则化损失
        """
        user_emb, pos_item_emb, neg_item_emb = self.get_embedding(
            users, pos_items, neg_items
        )
        
        reg_loss = (1/2) * (
            torch.norm(user_emb) ** 2 +
            torch.norm(pos_item_emb) ** 2 +
            torch.norm(neg_item_emb) ** 2
        ) / len(users)
        
        return reg_loss


# ============================================================================
# 训练数据集
# ============================================================================

class BPRDataset(Dataset):
    """BPR训练数据集"""
    
    def __init__(self, train_interactions, num_items, num_negatives=1):
        """
        Args:
            train_interactions: dict {user_idx: [item_idx1, item_idx2, ...]}
            num_items: 物品总数
            num_negatives: 每个正样本的负样本数量
        """
        self.train_interactions = train_interactions
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # 创建训练样本
        self.samples = []
        for user_idx, items in train_interactions.items():
            for item_idx in items:
                self.samples.append((user_idx, item_idx))
        
        # 为每个用户创建物品集合（加速负采样）
        self.user_item_set = {
            user: set(items) for user, items in train_interactions.items()
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        user_idx, pos_item_idx = self.samples[idx]
        
        # 负采样
        neg_items = []
        user_items = self.user_item_set[user_idx]
        
        while len(neg_items) < self.num_negatives:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in user_items:
                neg_items.append(neg_item)
        
        return {
            'user': user_idx,
            'pos_item': pos_item_idx,
            'neg_item': neg_items[0]
        }


# ============================================================================
# LightGCN训练器
# ============================================================================

class LightGCNTrainer:
    """LightGCN训练器"""
    
    def __init__(self, model, adj_matrix, device='cuda'):
        self.model = model.to(device)
        self.adj_matrix = adj_matrix.to(device)
        self.device = device
    
    def train_epoch(self, train_loader, optimizer, reg_weight=1e-4):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_bpr_loss = 0
        total_reg_loss = 0
        
        # 前向传播获取所有embedding（需参与梯度，保证BPR损失对参数可微）
        user_embeddings, item_embeddings = self.model(self.adj_matrix)
        
        for batch in train_loader:
            users = batch['user'].to(self.device)
            pos_items = batch['pos_item'].to(self.device)
            neg_items = batch['neg_item'].to(self.device)
            
            # 计算BPR损失
            bpr_loss = self.model.bpr_loss(
                users, pos_items, neg_items,
                user_embeddings, item_embeddings
            )
            
            # 计算正则化损失
            reg_loss = self.model.reg_loss(users, pos_items, neg_items)
            
            # 总损失
            loss = bpr_loss + reg_weight * reg_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新embedding（使用最新参数重新前向，保持后续batch计算一致）
            user_embeddings, item_embeddings = self.model(self.adj_matrix)
            
            total_loss += loss.item()
            total_bpr_loss += bpr_loss.item()
            total_reg_loss += reg_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_bpr_loss = total_bpr_loss / len(train_loader)
        avg_reg_loss = total_reg_loss / len(train_loader)
        
        return avg_loss, avg_bpr_loss, avg_reg_loss
    
    def evaluate(self, test_interactions, k_list=[10, 20, 30]):
        """评估模型"""
        self.model.eval()
        
        with torch.no_grad():
            user_embeddings, item_embeddings = self.model(self.adj_matrix)
        
        metrics = {f'Recall@{k}': [] for k in k_list}
        metrics.update({f'NDCG@{k}': [] for k in k_list})
        
        for user_idx, test_items in test_interactions.items():
            if user_idx >= self.model.num_users:
                continue
            
            # 获取用户embedding
            user_emb = user_embeddings[user_idx].unsqueeze(0)
            
            # 计算与所有物品的分数
            scores = torch.matmul(user_emb, item_embeddings.t()).squeeze()
            
            # 获取top-k
            _, top_k_items = torch.topk(scores, max(k_list))
            top_k_items = top_k_items.cpu().numpy()
            
            # 计算指标
            for k in k_list:
                pred_items = set(top_k_items[:k])
                true_items = set(test_items)
                
                # Recall@K
                hits = len(pred_items & true_items)
                recall = hits / len(true_items) if len(true_items) > 0 else 0
                metrics[f'Recall@{k}'].append(recall)
                
                # NDCG@K
                dcg = 0
                for i, item in enumerate(top_k_items[:k]):
                    if item in true_items:
                        dcg += 1 / np.log2(i + 2)
                
                idcg = sum([1 / np.log2(i + 2) for i in range(min(len(true_items), k))])
                ndcg = dcg / idcg if idcg > 0 else 0
                metrics[f'NDCG@{k}'].append(ndcg)
        
        # 计算平均值
        avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
        
        return avg_metrics
    
    def train(self, train_loader, test_interactions, 
              epochs=100, lr=1e-3, reg_weight=1e-4, 
              eval_every=5, early_stop_patience=10,
              best_model_path: str = 'best_lightgcn.pth'):
        """完整训练流程"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        best_recall = 0
        patience_counter = 0
        
        print("\n开始训练LightGCN...")
        print("=" * 60)
        
        for epoch in range(epochs):
            # 训练
            avg_loss, avg_bpr_loss, avg_reg_loss = self.train_epoch(
                train_loader, optimizer, reg_weight
            )
            
            # 打印损失
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss={avg_loss:.4f}, "
                  f"BPR={avg_bpr_loss:.4f}, "
                  f"Reg={avg_reg_loss:.4f}")
            
            # 定期评估
            if (epoch + 1) % eval_every == 0:
                metrics = self.evaluate(test_interactions)
                
                print(f"  Evaluation:")
                for metric_name, value in metrics.items():
                    print(f"    {metric_name}: {value:.4f}")
                
                # Early stopping
                current_recall = metrics['Recall@20']
                if current_recall > best_recall:
                    best_recall = current_recall
                    patience_counter = 0
                    
                    # 保存最佳模型
                    self.save_model(best_model_path)
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print(f"\n训练完成! 最佳 Recall@20: {best_recall:.4f}")
        
        # 加载最佳模型（使用与保存相同的路径）
        self.load_model(best_model_path)
        
        return best_recall
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_users': self.model.num_users,
            'num_items': self.model.num_items,
            'embedding_dim': self.model.embedding_dim,
            'num_layers': self.model.num_layers
        }, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已从 {path} 加载")
    
    def extract_embeddings(self):
        """提取训练好的embeddings (用于TokenRec)"""
        self.model.eval()
        
        with torch.no_grad():
            user_embeddings, item_embeddings = self.model(self.adj_matrix)
        
        return user_embeddings.cpu(), item_embeddings.cpu()


# ============================================================================
# 完整训练流程
# ============================================================================

def train_lightgcn_on_amazon(data_path='./tokenrec_project/data/Beauty_processed.pkl',
                             graph_path='./tokenrec_project/data/Beauty_graph.pkl',
                             embedding_dim=64,
                             num_layers=3,
                             epochs=100,
                             batch_size=2048,
                             lr=1e-3,
                             reg_weight=1e-4):
    """在Amazon数据集上训练LightGCN"""
    
    # 1. 加载数据
    print("=" * 60)
    print("加载数据...")
    print("=" * 60)
    
    data = torch.load(data_path)
    graph_data = torch.load(graph_path)
    
    num_users = data['num_users']
    num_items = data['num_items']
    train_interactions = data['train_interactions']
    test_interactions = data['test_interactions']
    adj_matrix = graph_data['adj_matrix']
    
    print(f"用户数: {num_users}")
    print(f"物品数: {num_items}")
    print(f"训练交互数: {len(data['train_df'])}")
    
    # 2. 创建数据集和数据加载器
    print("\n创建数据加载器...")
    train_dataset = BPRDataset(train_interactions, num_items)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # 3. 初始化模型
    print("\n初始化LightGCN模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        num_layers=num_layers
    )
    
    # 4. 训练模型
    trainer = LightGCNTrainer(model, adj_matrix, device=device)
    
    best_recall = trainer.train(
        train_loader=train_loader,
        test_interactions=test_interactions,
        epochs=epochs,
        lr=lr,
        reg_weight=reg_weight,
        eval_every=5,
        early_stop_patience=10
    )
    
    # 5. 提取embeddings
    print("\n提取训练好的embeddings...")
    user_embeddings, item_embeddings = trainer.extract_embeddings()
    
    # 6. 保存embeddings
    embeddings = {
        'user_embeddings': user_embeddings,
        'item_embeddings': item_embeddings,
        'num_users': num_users,
        'num_items': num_items,
        'embedding_dim': embedding_dim
    }
    
    embeddings_path = './tokenrec_project/data/Beauty_lightgcn_embeddings.pkl'
    torch.save(embeddings, embeddings_path)
    print(f"Embeddings已保存到: {embeddings_path}")
    
    return model, user_embeddings, item_embeddings


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 训练LightGCN
    model, user_embeddings, item_embeddings = train_lightgcn_on_amazon(
        data_path='./tokenrec_project/data/Beauty_processed.pkl',
        graph_path='./tokenrec_project/data/Beauty_graph.pkl',
        embedding_dim=64,
        num_layers=3,
        epochs=100,
        batch_size=2048,
        lr=1e-3,
        reg_weight=1e-4
    )
    
    print("\n" + "=" * 60)
    print("LightGCN训练完成!")
    print("=" * 60)
    print(f"用户embeddings形状: {user_embeddings.shape}")
    print(f"物品embeddings形状: {item_embeddings.shape}")