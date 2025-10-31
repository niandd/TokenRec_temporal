"""
TokenRec完整训练流程
包含：数据加载 -> 图构建 -> GNN预训练 -> TokenRec训练
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse

# 假设你已经有了前面的模块
from amazon_loader import AmazonDataLoader, BipartiteGraph
from lightgcn import LightGCN, LightGCNTrainer, BPRDataset
from tokenrec_core import MQTokenizer, TokenRec


# ============================================================================
# TokenRec训练数据集
# ============================================================================

class TokenRecDataset:
    """TokenRec训练数据集"""
    
    def __init__(self, train_interactions, user_embeddings, item_embeddings,
                 num_negatives=1, max_seq_len=100,
                 interactions_with_time=None, use_time: bool = False):
        self.train_interactions = train_interactions
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.num_negatives = num_negatives
        self.max_seq_len = max_seq_len
        self.use_time = use_time
        self.interactions_with_time = interactions_with_time
        
        self.user_ids = list(train_interactions.keys())
        self.num_items = len(item_embeddings)
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_idx = self.user_ids[idx]
        items = self.train_interactions[user_idx]
        times = None
        if self.use_time and self.interactions_with_time is not None:
            user_time_struct = self.interactions_with_time.get(user_idx)
            if user_time_struct is not None:
                times = user_time_struct['timestamps']
        
        # 留一个作为目标
        if len(items) > 1:
            history_items = items[:-1]
            target_item = items[-1]
            if times is not None and len(times) == len(items):
                history_times = times[:-1]
            else:
                history_times = None
        else:
            history_items = []
            target_item = items[0]
            history_times = None
        
        # 限制历史长度
        if len(history_items) > self.max_seq_len:
            history_items = history_items[-self.max_seq_len:]
        
        # 负采样
        neg_item = torch.randint(0, self.num_items, (1,)).item()
        while neg_item in items:
            neg_item = torch.randint(0, self.num_items, (1,)).item()
        
        # 获取embeddings
        user_emb = self.user_embeddings[user_idx]
        pos_item_emb = self.item_embeddings[target_item]
        neg_item_emb = self.item_embeddings[neg_item]
        
        # 历史物品embeddings
        if len(history_items) > 0:
            history_embs = self.item_embeddings[history_items]
        else:
            history_embs = torch.zeros(1, self.item_embeddings.shape[1])
        
        sample = {
            'user_idx': user_idx,
            'user_emb': user_emb,
            'target_item': target_item,
            'pos_item_emb': pos_item_emb,
            'neg_item_emb': neg_item_emb,
            'history_embs': history_embs,
            'history_len': len(history_items)
        }
        if self.use_time:
            if history_times is None or len(history_times) == 0:
                sample['history_timestamps'] = torch.zeros(history_embs.shape[0])
            else:
                sample['history_timestamps'] = torch.tensor(history_times, dtype=torch.float32)
        return sample


def collate_fn(batch):
    """处理变长序列"""
    user_indices = torch.tensor([item['user_idx'] for item in batch])
    user_embs = torch.stack([item['user_emb'] for item in batch])
    target_items = torch.tensor([item['target_item'] for item in batch])
    pos_item_embs = torch.stack([item['pos_item_emb'] for item in batch])
    neg_item_embs = torch.stack([item['neg_item_emb'] for item in batch])
    
    # Padding历史序列
    max_len = max([item['history_embs'].shape[0] for item in batch])
    emb_dim = batch[0]['history_embs'].shape[1]
    
    padded_history = torch.zeros(len(batch), max_len, emb_dim)
    history_lens = torch.tensor([item['history_len'] for item in batch])
    
    for i, item in enumerate(batch):
        hist_len = item['history_embs'].shape[0]
        padded_history[i, :hist_len, :] = item['history_embs']
    
    out = {
        'user_idx': user_indices,
        'user_emb': user_embs,
        'target_item': target_items,
        'pos_item_emb': pos_item_embs,
        'neg_item_emb': neg_item_embs,
        'history_embs': padded_history,
        'history_len': history_lens
    }
    # 可选时间戳对齐
    if 'history_timestamps' in batch[0]:
        max_len_ts = max([len(item['history_timestamps']) for item in batch])
        padded_ts = torch.zeros(len(batch), max_len_ts)
        for i, item in enumerate(batch):
            ts = item['history_timestamps']
            if ts.ndim == 0:
                ts = ts.unsqueeze(0)
            padded_ts[i, :len(ts)] = ts[:max_len_ts]
        out['history_timestamps'] = padded_ts
    return out


# ============================================================================
# 完整Pipeline
# ============================================================================

class TokenRecPipeline:
    """TokenRec完整训练流程"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        os.makedirs(config['output_dir'], exist_ok=True)
    
    def step1_load_data(self):
        """步骤1: 加载和预处理数据"""
        print("\n" + "=" * 60)
        print("步骤 1/5: 数据加载和预处理")
        print("=" * 60)
        
        from amazon_loader import AmazonDataLoader
        
        loader = AmazonDataLoader(
            category=self.config['dataset'],
            data_dir=self.config['data_dir']
        )
        
        data = loader.load_and_preprocess()
        
        self.data = data
        return data
    
    def step2_build_graph(self):
        """步骤2: 构建用户-物品二部图"""
        print("\n" + "=" * 60)
        print("步骤 2/5: 构建二部图")
        print("=" * 60)
        
        from amazon_loader import BipartiteGraph
        
        graph = BipartiteGraph(
            num_users=self.data['num_users'],
            num_items=self.data['num_items'],
            train_interactions=self.data['train_interactions']
        )
        
        adj_matrix, edge_index = graph.to_torch_sparse_coo()
        
        self.adj_matrix = adj_matrix
        self.edge_index = edge_index
        
        return adj_matrix, edge_index
    
    def step3_train_gnn(self):
        """步骤3: 训练LightGCN获取embeddings"""
        print("\n" + "=" * 60)
        print("步骤 3/5: 训练LightGCN")
        print("=" * 60)
        
        from lightgcn import LightGCN, LightGCNTrainer, BPRDataset
        
        # 创建模型
        model = LightGCN(
            num_users=self.data['num_users'],
            num_items=self.data['num_items'],
            embedding_dim=self.config['gnn_emb_dim'],
            num_layers=self.config['gnn_layers']
        )
        
        # 创建训练数据
        train_dataset = BPRDataset(
            self.data['train_interactions'],
            self.data['num_items']
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['gnn_batch_size'],
            shuffle=True,
            num_workers=4
        )
        
        # 训练
        trainer = LightGCNTrainer(model, self.adj_matrix, device=self.device)

        # 组装 LightGCN ckpt 路径
        ckpt_dir = self.config.get('ckpt_dir_lightgcn', None)
        if ckpt_dir is None:
            ckpt_dir = os.path.join(self.config['output_dir'], 'ckpt_lightgcn')
        os.makedirs(ckpt_dir, exist_ok=True)
        best_ckpt_path = os.path.join(ckpt_dir, 'best_lightgcn.pth')

        trainer.train(
            train_loader=train_loader,
            test_interactions=self.data['test_interactions'],
            epochs=self.config['gnn_epochs'],
            lr=self.config['gnn_lr'],
            reg_weight=self.config['gnn_reg'],
            eval_every=5,
            early_stop_patience=10,
            best_model_path=best_ckpt_path
        )
        
        # 提取embeddings
        user_embeddings, item_embeddings = trainer.extract_embeddings()
        
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        
        # 保存
        torch.save({
            'user_embeddings': user_embeddings,
            'item_embeddings': item_embeddings
        }, os.path.join(self.config['output_dir'], 'gnn_embeddings.pkl'))
        
        return user_embeddings, item_embeddings
    
    def step4_train_tokenizers(self):
        """步骤4: 训练MQ-Tokenizers"""
        print("\n" + "=" * 60)
        print("步骤 4/5: 训练MQ-Tokenizers")
        print("=" * 60)
        
        from tokenrec_core import MQTokenizer
        
        # 初始化tokenizers
        user_tokenizer = MQTokenizer(
            input_dim=self.config['gnn_emb_dim'],
            K=self.config['K'],
            L=self.config['L'],
            d_c=self.config['d_c'],
            mask_ratio=self.config['mask_ratio']
        ).to(self.device)
        
        item_tokenizer = MQTokenizer(
            input_dim=self.config['gnn_emb_dim'],
            K=self.config['K'],
            L=self.config['L'],
            d_c=self.config['d_c'],
            mask_ratio=self.config['mask_ratio']
        ).to(self.device)
        
        # 训练user tokenizer
        print("\n训练User MQ-Tokenizer...")
        # 使用 K-Means 初始化码本
        print("  使用 K-Means 初始化 User 码本...")
        user_tokenizer.initialize_codebooks_with_kmeans(self.user_embeddings.to(self.device))
        self._train_single_tokenizer(
            user_tokenizer,
            self.user_embeddings,
            epochs=self.config['tokenizer_epochs'],
            batch_size=self.config['tokenizer_batch_size'],
            alpha=0.1,
            beta=0.1,
            lr=1e-4,
            max_grad_norm=1.0
        )
        
        # 训练item tokenizer
        print("\n训练Item MQ-Tokenizer...")
        # 使用 K-Means 初始化码本
        print("  使用 K-Means 初始化 Item 码本...")
        item_tokenizer.initialize_codebooks_with_kmeans(self.item_embeddings.to(self.device))
        self._train_single_tokenizer(
            item_tokenizer,
            self.item_embeddings,
            epochs=self.config['tokenizer_epochs'],
            batch_size=self.config['tokenizer_batch_size'],
            alpha=0.1,
            beta=0.1,
            lr=1e-4,
            max_grad_norm=1.0
        )
        
        self.user_tokenizer = user_tokenizer
        self.item_tokenizer = item_tokenizer
        
        # 保存tokenizers到 ckpt/tokenrec 下
        ckpt_dir_tokenrec = self.config.get('ckpt_dir_tokenrec', None)
        if ckpt_dir_tokenrec is None:
            ckpt_dir_tokenrec = os.path.join(self.config['output_dir'], 'ckpt_tokenrec')
        os.makedirs(ckpt_dir_tokenrec, exist_ok=True)
        torch.save({
            'user_tokenizer': user_tokenizer.state_dict(),
            'item_tokenizer': item_tokenizer.state_dict()
        }, os.path.join(ckpt_dir_tokenrec, 'tokenizers.pkl'))
        
        # 可选：导出所有用户/物品的 token ID
        if self.config.get('export_tokens', False):
            print("\n导出用户与物品的 Token IDs...")
            with torch.no_grad():
                u_tokens, _, _ = user_tokenizer(self.user_embeddings.to(self.device))
                i_tokens, _, _ = item_tokenizer(self.item_embeddings.to(self.device))
            torch.save({'user_tokens': u_tokens.cpu(), 'item_tokens': i_tokens.cpu()},
                       os.path.join(self.config['output_dir'], 'tokens_all.pt'))
            print("Token IDs 已保存: tokens_all.pt")

        return user_tokenizer, item_tokenizer
    
    def _train_single_tokenizer(self, tokenizer, embeddings, epochs, batch_size, alpha, beta, lr=1e-4, max_grad_norm=1.0):
        """训练单个tokenizer，支持权重与梯度裁剪"""
        optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=lr)
        embeddings = embeddings.to(self.device)
        
        num_samples = len(embeddings)
        
        for epoch in range(epochs):
            tokenizer.train()
            
            indices = torch.randperm(num_samples)
            total_loss = 0
            total_recon = 0
            total_codebook = 0
            total_commit = 0
            num_batches = 0
            
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                batch_emb = embeddings[batch_indices]
                # 归一化以稳定量化损失的尺度
                batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
                
                tokens, reconstructed, loss_dict = tokenizer(batch_emb)
                
                recon_loss = loss_dict['recon_loss']
                codebook_loss = loss_dict['codebook_loss']
                commitment_loss = loss_dict['commitment_loss']
                
                loss = recon_loss + alpha * codebook_loss + beta * commitment_loss
                
                optimizer.zero_grad()
                loss.backward()
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_grad_norm)
                optimizer.step()
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_codebook += codebook_loss.item()
                total_commit += commitment_loss.item()
                num_batches += 1
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                avg_recon = total_recon / num_batches
                avg_codebook = total_codebook / num_batches
                avg_commit = total_commit / num_batches
                print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f} | recon={avg_recon:.4f}, codebook={avg_codebook:.4f} (x{alpha}), commit={avg_commit:.4f} (x{beta})")
    
    def step5_train_tokenrec(self):
        """步骤5: 训练TokenRec"""
        print("\n" + "=" * 60)
        print("步骤 5/5: 训练TokenRec")
        print("=" * 60)
        
        from tokenrec_core import TokenRec
        
        # 初始化模型
        model = TokenRec(
            self.user_tokenizer,
            self.item_tokenizer,
            llm_model_name=self.config['llm_model'],
            item_emb_dim=self.config['gnn_emb_dim']
        ).to(self.device)
        
        # 冻结tokenizers
        for param in model.user_tokenizer.parameters():
            param.requires_grad = False
        for param in model.item_tokenizer.parameters():
            param.requires_grad = False
        
        # 创建数据集
        interactions_with_time = None
        if self.config.get('use_time', False):
            interactions_with_time = self.data.get('train_interactions_with_time')

        train_dataset = TokenRecDataset(
            self.data['train_interactions'],
            self.user_embeddings,
            self.item_embeddings,
            max_seq_len=self.config['max_seq_len'],
            interactions_with_time=interactions_with_time,
            use_time=self.config.get('use_time', False)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['tokenrec_batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        # 优化器
        optimizer = torch.optim.AdamW([
            {'params': model.llm.parameters(), 'lr': self.config['tokenrec_lr']},
            {'params': model.projection.parameters(), 'lr': self.config['tokenrec_lr'] * 10}
        ])
        
        # 训练
        item_emb_database = self.item_embeddings.to(self.device)
        
        best_recall = 0
        
        for epoch in range(self.config['tokenrec_epochs']):
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                user_emb = batch['user_emb'].to(self.device)
                pos_item_emb = batch['pos_item_emb'].to(self.device)
                neg_item_emb = batch['neg_item_emb'].to(self.device)
                history_embs = batch['history_embs'].to(self.device)
                history_len = batch['history_len']
                history_timestamps = batch.get('history_timestamps', None)
                
                # 如果没有历史
                if history_len.sum() == 0:
                    history_embs = None
                
                # 前向传播
                z = model(user_emb, history_embs, history_timestamps)
                
                # 计算损失
                loss = model.compute_ranking_loss(
                    z, pos_item_emb, neg_item_emb,
                    margin=self.config['margin']
                )
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{self.config['tokenrec_epochs']}: Loss = {avg_loss:.4f}")
            
            # 定期评估
            if (epoch + 1) % 5 == 0:
                metrics = self._evaluate_tokenrec(model, item_emb_database)
                
                print(f"  Evaluation:")
                for metric_name, value in metrics.items():
                    print(f"    {metric_name}: {value:.4f}")
                
                # 保存最佳模型
                if metrics['Recall@20'] > best_recall:
                    best_recall = metrics['Recall@20']
            ckpt_dir_tokenrec = self.config.get('ckpt_dir_tokenrec', None)
            if ckpt_dir_tokenrec is None:
                ckpt_dir_tokenrec = os.path.join(self.config['output_dir'], 'ckpt_tokenrec')
            os.makedirs(ckpt_dir_tokenrec, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir_tokenrec, 'best_tokenrec.pth'))
        
        print(f"\n训练完成! 最佳 Recall@20: {best_recall:.4f}")
        
        return model
    
    def _evaluate_tokenrec(self, model, item_emb_database):
        """评估TokenRec"""
        model.eval()
        
        metrics = {'Recall@10': [], 'Recall@20': [], 'Recall@30': [],
                  'NDCG@10': [], 'NDCG@20': [], 'NDCG@30': []}
        
        # 创建测试数据集
        interactions_with_time = None
        if self.config.get('use_time', False):
            interactions_with_time = self.data.get('test_interactions_with_time')

        test_dataset = TokenRecDataset(
            self.data['test_interactions'],
            self.user_embeddings,
            self.item_embeddings,
            interactions_with_time=interactions_with_time,
            use_time=self.config.get('use_time', False)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            collate_fn=collate_fn
        )
        
        with torch.no_grad():
            for batch in test_loader:
                user_emb = batch['user_emb'].to(self.device)
                target_items = batch['target_item'].numpy()
                
                # 生成表示
                z = model(user_emb, None, None)
                
                # 检索top-30
                top_k_indices, _ = model.retrieve_top_k(z, item_emb_database, k=30)
                
                # 计算指标
                for i, target in enumerate(target_items):
                    preds = top_k_indices[i].cpu().numpy()
                    
                    for k in [10, 20, 30]:
                        recall = 1.0 if target in preds[:k] else 0.0
                        metrics[f'Recall@{k}'].append(recall)
                        
                        # NDCG
                        if target in preds[:k]:
                            rank = list(preds[:k]).index(target)
                            ndcg = 1.0 / torch.log2(torch.tensor(rank + 2.0)).item()
                        else:
                            ndcg = 0.0
                        metrics[f'NDCG@{k}'].append(ndcg)
        
        return {key: torch.tensor(values).mean().item() 
                for key, values in metrics.items()}
    
    def run_full_pipeline(self):
        """运行完整流程"""
        print("\n" + "=" * 60)
        print("TokenRec 完整训练流程")
        print("=" * 60)
        
        # 步骤1: 加载数据
        self.step1_load_data()
        
        # 步骤2: 构建图
        self.step2_build_graph()
        
        # 步骤3: 训练GNN
        self.step3_train_gnn()
        
        # 步骤4: 训练Tokenizers
        self.step4_train_tokenizers()
        
        # 步骤5: 训练TokenRec
        model = self.step5_train_tokenrec()
        
        print("\n" + "=" * 60)
        print("训练流程完成!")
        print("=" * 60)
        
        return model


# ============================================================================
# 主函数
# ============================================================================

def main():
    # 配置参数
    config = {
        # 数据参数
        'dataset': 'Beauty',
        'data_dir': './data',
        'output_dir': './outputs',
        'ckpt_dir_lightgcn': None,
        'export_tokens': True,
        # 特征开关
        'use_time': False,
        'use_text': False,
        
        # GNN参数
        'gnn_emb_dim': 64,
        'gnn_layers': 3,
        'gnn_epochs': 100,
        'gnn_batch_size': 2048,
        'gnn_lr': 1e-3,
        'gnn_reg': 1e-4,
        
        # Tokenizer参数
        'K': 3,
        'L': 512,
        'd_c': 64,
        'mask_ratio': 0.2,
        'beta_user': 0.25,
        'beta_item': 0.25,
        'tokenizer_epochs': 100,
        'tokenizer_batch_size': 256,
        
        # TokenRec参数
        'llm_model': 't5-small',
        'max_seq_len': 100,
        'tokenrec_epochs': 50,
        'tokenrec_batch_size': 32,
        'tokenrec_lr': 1e-4,
        'margin': 0.1
    }
    
    # CLI 覆盖
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_time', action='store_true')
    parser.add_argument('--use_text', action='store_true')
    parser.add_argument('--output_dir', type=str, default=config['output_dir'])
    parser.add_argument('--data_dir', type=str, default=config['data_dir'])
    parser.add_argument('--ckpt_dir_lightgcn', type=str, default='')
    parser.add_argument('--export_tokens', action='store_true')
    args, _ = parser.parse_known_args()
    config['use_time'] = bool(args.use_time)
    config['use_text'] = bool(args.use_text)
    config['output_dir'] = args.output_dir
    config['data_dir'] = args.data_dir
    if args.ckpt_dir_lightgcn:
        config['ckpt_dir_lightgcn'] = args.ckpt_dir_lightgcn
    if args.export_tokens:
        config['export_tokens'] = True

    # 运行完整流程
    pipeline = TokenRecPipeline(config)
    model = pipeline.run_full_pipeline()
    
    return model


if __name__ == "__main__":
    model = main()