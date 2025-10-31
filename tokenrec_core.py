import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np

class MQTokenizer(nn.Module):
    """Masked Vector-Quantized Tokenizer for Users/Items"""
    def __init__(self, input_dim, K=3, L=512, d_c=64, mask_ratio=0.2):
        """
        Args:
            input_dim: 输入表示的维度 (来自GNN的embedding维度)
            K: 子编码器/子码本的数量
            L: 每个子码本中的token数量
            d_c: 码本embedding的维度
            mask_ratio: 掩码比例
        """
        super().__init__()
        self.K = K
        self.L = L
        self.d_c = d_c
        self.mask_ratio = mask_ratio
        
        # K-way编码器
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, d_c)
            ) for _ in range(K)
        ])
        
        # K-way码本 (可学习的)
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(L, d_c)) for _ in range(K)
        ])
        
        # K-to-1解码器
        self.decoder = nn.Sequential(
            nn.Linear(d_c, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        # 输出归一化，稳定编码尺度
        self.output_norm = nn.LayerNorm(d_c)

    @torch.no_grad()
    def initialize_codebooks_with_kmeans(self, embeddings: torch.Tensor, max_samples: int = 100000, max_iters: int = 20, tol: float = 1e-4, seed: int = 42):
        """
        使用 K-Means 对每个子编码器输出做聚类，初始化每个子码本（L, d_c）。
        Args:
            embeddings: (num_samples, input_dim) 的 GNN 表示（应在相同 device 上）
            max_samples: 参与初始化的最大样本数（子采样以加速）
            max_iters: K-Means 最大迭代次数
            tol: 收敛阈值（均值移动量的范数）
            seed: 随机种子
        """
        device = embeddings.device
        torch.manual_seed(seed)

        num_samples = embeddings.shape[0]
        if num_samples > max_samples:
            indices = torch.randperm(num_samples, device=device)[:max_samples]
            emb_sample = embeddings[indices]
        else:
            emb_sample = embeddings

        # 逐个子编码器进行编码并聚类
        for k in range(self.K):
            # 编码并做与训练一致的归一化
            encoded = self.encoders[k](emb_sample)  # (n, d_c)
            encoded = self.output_norm(encoded)
            # L2 归一化以稳定尺度
            encoded_n = F.normalize(encoded, p=2, dim=1)

            # K-Means++ 简化初始化：随机选取一个中心，其后按距离概率选取
            n, d = encoded_n.shape
            L = self.L
            # 选第一个中心
            centers = torch.empty(L, d, device=device)
            first_idx = torch.randint(0, n, (1,), device=device).item()
            centers[0] = encoded_n[first_idx]

            # 选其余中心
            closest_dist = torch.cdist(encoded_n, centers[0:1], p=2).squeeze(1)  # (n,)
            for c in range(1, L):
                probs = closest_dist.clamp(min=1e-12)
                probs = probs / probs.sum()
                next_idx = torch.multinomial(probs, 1).item()
                centers[c] = encoded_n[next_idx]
                new_dist = torch.cdist(encoded_n, centers[c:c+1], p=2).squeeze(1)
                closest_dist = torch.minimum(closest_dist, new_dist)

            # 迭代更新
            for _ in range(max_iters):
                # 分配
                dists = 1 - torch.matmul(encoded_n, centers.t())  # (n, L) 余弦距离等价
                assignments = torch.argmin(dists, dim=1)  # (n,)

                # 更新中心
                new_centers = torch.zeros_like(centers)
                counts = torch.zeros(L, device=device)
                for c in range(L):
                    mask = (assignments == c)
                    if mask.any():
                        cluster_points = encoded_n[mask]
                        new_centers[c] = cluster_points.mean(dim=0)
                        counts[c] = mask.sum()
                    else:
                        # 若空簇，随机重采样一个样本点作为中心，避免塌陷
                        ridx = torch.randint(0, n, (1,), device=device).item()
                        new_centers[c] = encoded_n[ridx]
                        counts[c] = 1

                # 归一化中心
                new_centers = F.normalize(new_centers, p=2, dim=1)

                # 收敛检测
                shift = torch.norm(centers - new_centers, p=2, dim=1).mean()
                centers = new_centers
                if shift.item() < tol:
                    break

            # 写入码本（使用单位范数中心，与量化阶段保持一致）
            self.codebooks[k].data.copy_(centers)
    
    def masking(self, x):
        """随机掩码操作"""
        if self.training:
            mask = torch.bernoulli(torch.full_like(x, 1 - self.mask_ratio))
            return x * mask
        return x
    
    def quantize(self, encoded, codebook):
        """向量量化：找到最近的码本向量"""
        # encoded: (batch_size, d_c)
        # codebook: (L, d_c)
        
        # 先做L2归一化，使用等价的余弦距离来稳定尺度
        encoded_n = F.normalize(encoded, p=2, dim=1)
        codebook_n = F.normalize(codebook, p=2, dim=1)
        # 余弦相似度越大越近 => 使用(1 - cos)当作距离
        cos_sim = torch.matmul(encoded_n, codebook_n.t())  # (batch, L)
        distances = 1 - cos_sim
        
        # 找到最近的索引
        indices = torch.argmin(distances, dim=1)
        
        # 获取量化后的向量（用归一化后的码本，保持稳定）
        quantized = codebook_n[indices]
        
        return quantized, indices
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim) - 来自GNN的表示
        Returns:
            tokens: (batch_size, K) - K个离散token索引
            reconstructed: (batch_size, input_dim) - 重建的表示
            loss_dict: 包含各种损失的字典
        """
        # 1. 掩码操作
        x_masked = self.masking(x)
        
        # 2. K-way编码和量化
        tokens = []
        quantized_list = []
        commitment_loss = 0
        codebook_loss = 0
        
        for k in range(self.K):
            # 编码
            encoded = self.encoders[k](x_masked)  # (batch_size, d_c)
            encoded = self.output_norm(encoded)
            
            # 量化
            quantized, indices = self.quantize(encoded, self.codebooks[k])
            
            # 使用straight-through estimator
            quantized = encoded + (quantized - encoded).detach()
            
            tokens.append(indices)
            quantized_list.append(quantized)
            
            # 计算损失
            codebook_loss += F.mse_loss(encoded.detach(), quantized)
            commitment_loss += F.mse_loss(encoded, quantized.detach())
        
        # 3. K-to-1解码
        # 平均池化所有量化后的向量
        avg_quantized = torch.stack(quantized_list, dim=1).mean(dim=1)
        reconstructed = self.decoder(avg_quantized)
        
        # 4. 计算重建损失
        recon_loss = F.mse_loss(reconstructed, x)
        
        tokens = torch.stack(tokens, dim=1)  # (batch_size, K)
        
        loss_dict = {
            'recon_loss': recon_loss,
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss
        }
        
        return tokens, reconstructed, loss_dict


class TokenRec(nn.Module):
    """TokenRec主模型"""
    def __init__(self, user_mq_tokenizer, item_mq_tokenizer, 
                 llm_model_name='t5-small', item_emb_dim=64):
        super().__init__()
        
        self.user_tokenizer = user_mq_tokenizer
        self.item_tokenizer = item_mq_tokenizer
        
        # LLM骨干网络 (T5-small)
        self.llm = T5EncoderModel.from_pretrained(llm_model_name)
        self.tokenizer_llm = T5Tokenizer.from_pretrained(llm_model_name)
        
        # 为OOV tokens添加新的embeddings
        K = user_mq_tokenizer.K
        L = user_mq_tokenizer.L
        vocab_size = self.llm.config.vocab_size
        
        # 扩展词汇表
        num_new_tokens = 2 * K * L  # user tokens + item tokens
        self.llm.resize_token_embeddings(vocab_size + num_new_tokens)
        
        # 投影层：将LLM输出映射到物品表示空间
        hidden_size = self.llm.config.d_model
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, item_emb_dim)
        )
        
        self.item_emb_dim = item_emb_dim
    
    def create_prompt(self, user_tokens, item_tokens_list=None, item_time_list=None):
        """
        创建提示词
        Args:
            user_tokens: (batch_size, K)
            item_tokens_list: list of (batch_size, K) or None
            item_time_list: list of (batch_size,) 对应每个历史交互的时间戳(可选)
        """
        batch_size = int(user_tokens.shape[0])
        prompts = []

        # 统一保障：将时间列表长度与item列表对齐（若提供）
        if item_tokens_list is not None and item_time_list is not None:
            if len(item_time_list) != len(item_tokens_list):
                # 对齐到较短长度，避免索引越界
                min_len = min(len(item_time_list), len(item_tokens_list))
                item_time_list = item_time_list[:min_len]
                item_tokens_list = item_tokens_list[:min_len]

        for i in range(batch_size):
            # 用户token转换为文本（严格按第二维K循环）
            K_user = int(user_tokens.shape[1])
            user_token_str = " ".join(
                [f"<u{k}-{int(user_tokens[i, k].item())}>" for k in range(K_user)]
            )

            if item_tokens_list is None:
                # Prompt 1: 只有用户ID
                prompt = f"Given user {user_token_str}, recommend items that the user may like."
            else:
                # Prompt 2: 用户ID + 交互历史（当前样本 i 的历史）
                tokens_i = item_tokens_list[i]
                # 保障形状为 (seq_len, K)
                if tokens_i.dim() == 1:
                    tokens_i = tokens_i.unsqueeze(0)
                seq_len_i = int(tokens_i.shape[0])
                K_item = int(tokens_i.shape[1])

                item_str_list = []
                for t in range(seq_len_i):
                    item_token_str = " ".join(
                        [f"<i{k}-{int(tokens_i[t, k].item())}>" for k in range(K_item)]
                    )
                    item_str_list.append(item_token_str)
                items_str = ", ".join(item_str_list)

                # 附加时间信息（若提供）
                if item_time_list is not None:
                    times_i = item_time_list[i]
                    # 将可能的tensor/ndarray/列表统一为Python列表字符串
                    try:
                        if torch.is_tensor(times_i):
                            times_seq = [int(x.item()) for x in times_i.view(-1)]
                        elif isinstance(times_i, np.ndarray):
                            times_seq = [int(x) for x in times_i.reshape(-1).tolist()]
                        else:
                            times_seq = [int(x) for x in list(times_i)]
                        # 对齐长度，防止时间长度与序列长度不一致
                        if len(times_seq) != seq_len_i:
                            times_seq = times_seq[:seq_len_i]
                        time_values = ", ".join([str(x) for x in times_seq])
                        prompt = (
                            f"Given user {user_token_str} and the user has interacted with items {items_str} "
                            f"at timestamps [{time_values}], recommend items that the user may like."
                        )
                    except Exception:
                        prompt = (
                            f"Given user {user_token_str} and the user has interacted with items {items_str}, "
                            f"recommend items that the user may like."
                        )
                else:
                    prompt = (
                        f"Given user {user_token_str} and the user has interacted with items {items_str}, "
                        f"recommend items that the user may like."
                    )

            prompts.append(prompt)

        return prompts
    
    def forward(self, user_emb, item_history_emb=None, item_history_timestamps=None):
        """
        Args:
            user_emb: (batch_size, user_emb_dim) - 用户的GNN表示
            item_history_emb: (batch_size, seq_len, item_emb_dim) - 用户交互历史的GNN表示
            item_history_timestamps: (batch_size, seq_len) - 历史交互对应时间戳（可选）
        Returns:
            z: (batch_size, item_emb_dim) - 生成的物品表示
        """
        # 1. 获取用户tokens
        user_tokens, _, _ = self.user_tokenizer(user_emb)
        
        # 2. 获取物品tokens (如果有历史交互)
        item_tokens_list = None
        item_time_list = None
        if item_history_emb is not None:
            batch_size, seq_len, _ = item_history_emb.shape
            item_history_flat = item_history_emb.view(-1, item_history_emb.shape[-1])
            item_tokens_flat, _, _ = self.item_tokenizer(item_history_flat)
            item_tokens_list = [item_tokens_flat[i*seq_len:(i+1)*seq_len] 
                               for i in range(batch_size)]
            if item_history_timestamps is not None:
                item_time_list = [item_history_timestamps[i, :seq_len] for i in range(batch_size)]
        
        # 3. 创建提示词
        prompts = self.create_prompt(user_tokens, item_tokens_list, item_time_list)
        
        # 4. Tokenize prompts
        inputs = self.tokenizer_llm(prompts, return_tensors='pt', 
                                    padding=True, truncation=True)
        inputs = {k: v.to(user_emb.device) for k, v in inputs.items()}
        
        # 5. 通过LLM编码
        outputs = self.llm(**inputs)
        hidden_states = outputs.last_hidden_state
        
        # 6. 取[CLS] token或平均池化
        h = hidden_states[:, 0, :]  # 使用第一个token的表示
        
        # 7. 投影到物品表示空间
        z = self.projection(h)
        
        return z
    
    def compute_ranking_loss(self, z, pos_item_emb, neg_item_emb, margin=0.1):
        """
        计算排序损失
        Args:
            z: (batch_size, item_emb_dim) - 用户的生成表示
            pos_item_emb: (batch_size, item_emb_dim) - 正样本物品表示
            neg_item_emb: (batch_size, item_emb_dim) - 负样本物品表示
        """
        # Cosine相似度
        pos_sim = F.cosine_similarity(z, pos_item_emb, dim=1)
        neg_sim = F.cosine_similarity(z, neg_item_emb, dim=1)
        
        # Pairwise ranking loss
        pos_loss = 1 - pos_sim
        neg_loss = torch.clamp(neg_sim - margin, min=0)
        
        loss = (pos_loss + neg_loss).mean()
        return loss
    
    def retrieve_top_k(self, z, item_emb_database, k=20):
        """
        从物品库中检索top-k物品
        Args:
            z: (batch_size, item_emb_dim)
            item_emb_database: (num_items, item_emb_dim)
            k: 检索的物品数量
        Returns:
            top_k_indices: (batch_size, k)
            top_k_scores: (batch_size, k)
        """
        # 计算余弦相似度
        z_norm = F.normalize(z, p=2, dim=1)
        item_norm = F.normalize(item_emb_database, p=2, dim=1)
        
        scores = torch.matmul(z_norm, item_norm.t())  # (batch_size, num_items)
        
        # 获取top-k
        top_k_scores, top_k_indices = torch.topk(scores, k, dim=1)
        
        return top_k_indices, top_k_scores


# 训练示例代码
def train_mq_tokenizer(mq_tokenizer, gnn_embeddings, epochs=100, beta=0.25):
    """训练MQ-Tokenizer"""
    optimizer = torch.optim.AdamW(mq_tokenizer.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        mq_tokenizer.train()
        
        tokens, reconstructed, loss_dict = mq_tokenizer(gnn_embeddings)
        
        # 总损失
        total_loss = loss_dict['recon_loss'] + \
                    loss_dict['codebook_loss'] + \
                    beta * loss_dict['commitment_loss']
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {total_loss.item():.4f}")


def train_tokenrec(model, train_loader, item_emb_database, epochs=50):
    """训练TokenRec模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            user_emb = batch['user_emb']
            pos_item_emb = batch['pos_item_emb']
            neg_item_emb = batch['neg_item_emb']
            item_history_emb = batch.get('item_history_emb', None)
            
            # 前向传播
            z = model(user_emb, item_history_emb)
            
            # 计算损失
            loss = model.compute_ranking_loss(z, pos_item_emb, neg_item_emb)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Avg Loss = {total_loss/len(train_loader):.4f}")


# 使用示例
if __name__ == "__main__":
    # 假设从LightGCN获得的embedding维度是64
    gnn_emb_dim = 64
    
    # 1. 初始化MQ-Tokenizers
    user_tokenizer = MQTokenizer(input_dim=gnn_emb_dim, K=3, L=512, d_c=64)
    item_tokenizer = MQTokenizer(input_dim=gnn_emb_dim, K=3, L=512, d_c=64)
    
    # 2. 初始化TokenRec
    model = TokenRec(user_tokenizer, item_tokenizer, item_emb_dim=gnn_emb_dim)
    
    # 3. 训练示例
    # 首先训练tokenizers (使用GNN embeddings)
    # train_mq_tokenizer(user_tokenizer, user_gnn_embeddings)
    # train_mq_tokenizer(item_tokenizer, item_gnn_embeddings)
    
    # 然后训练TokenRec
    # train_tokenrec(model, train_loader, item_emb_database)
    
    print("TokenRec模型初始化完成!")