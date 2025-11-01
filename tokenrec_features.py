"""
TokenRec 特征使用详解和改进方案
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

# ============================================================================
# 当前TokenRec使用的特征
# ============================================================================

"""
当前方案原始TokenRec论文):

1. USER特征:
   ✓ User ID (通过GNN学习的embedding)
   ✓ 交互历史 (哪些items)
   ✗ 没有用:年龄、性别、地理位置等
   ✗ 没有用:交互时间信息

2. ITEM特征:
   ✓ Item ID (通过GNN学习的embedding)
   ✗ 没有用:标题、描述、类别、价格等
   ✗ 没有用:物品的创建时间、流行度等

3. 交互特征:
   ✓ 用户点击/购买了哪些items (仅使用交互关系)
   ✗ 没有用:交互时间戳
   ✗ 没有用:评分信息
   ✗ 没有用:交互顺序的时序信息

为什么这样设计？
- 论文专注于"协同过滤"，即利用用户-物品的交互模式
- GNN (LightGCN) 自动从交互图中学习user/item的表示
- 这些表示隐式地包含了协同信息("相似用户喜欢相似物品")
"""

# ============================================================================
# 改进方案1: 添加时间特征编码
# ============================================================================

class TimeEncoder(nn.Module):
    """时间特征编码器"""
    
    def __init__(self, emb_dim=64):
        super().__init__()
        self.emb_dim = emb_dim
        
        # 方法1: Sinusoidal Position Encoding (类似Transformer)
        # 将时间戳映射到周期性的sin/cos函数
        
        # 方法2: 可学习的时间embedding
        # 将时间离散化后用embedding层
        
    def timestamp_to_sinusoidal(self, timestamps):
        """
        将时间戳转换为sin/cos编码
        
        Args:
            timestamps: (batch_size, seq_len) - Unix时间戳
        Returns:
            time_emb: (batch_size, seq_len, emb_dim)
        """
        # 归一化时间戳到 [0, 1]
        min_time = timestamps.min()
        max_time = timestamps.max()
        normalized_time = (timestamps - min_time) / (max_time - min_time + 1e-8)
        
        # 生成不同频率的sin/cos
        position = normalized_time.unsqueeze(-1)  # (batch, seq, 1)
        
        div_term = torch.exp(
            torch.arange(0, self.emb_dim, 2).float() * 
            (-np.log(10000.0) / self.emb_dim)
        )
        
        time_emb = torch.zeros(*timestamps.shape, self.emb_dim)
        time_emb[:, :, 0::2] = torch.sin(position * div_term)
        time_emb[:, :, 1::2] = torch.cos(position * div_term)
        
        return time_emb
    
    def timestamp_to_discrete(self, timestamps, num_bins=100):
        """
        将时间戳离散化为bins
        
        Args:
            timestamps: (batch_size, seq_len)
            num_bins: 时间区间数量
        Returns:
            time_ids: (batch_size, seq_len) - 时间区间ID
        """
        min_time = timestamps.min()
        max_time = timestamps.max()
        
        # 归一化到 [0, num_bins-1]
        normalized = (timestamps - min_time) / (max_time - min_time + 1e-8)
        time_ids = (normalized * (num_bins - 1)).long()
        
        return time_ids
    
    def extract_time_features(self, timestamps):
        """
        提取多种时间特征
        
        Args:
            timestamps: (batch_size, seq_len) - Unix时间戳
        Returns:
            features: dict of time features
        """
        # 转换为datetime
        # 注意:这里假设timestamps是numpy数组
        if isinstance(timestamps, torch.Tensor):
            timestamps = timestamps.cpu().numpy()
        
        features = {}
        
        # 提取日期时间组件
        datetimes = [datetime.fromtimestamp(ts) for ts in timestamps.flatten()]
        
        # 小时 (0-23)
        hours = torch.tensor([dt.hour for dt in datetimes]).reshape(timestamps.shape)
        features['hour'] = hours
        
        # 星期几 (0-6)
        weekdays = torch.tensor([dt.weekday() for dt in datetimes]).reshape(timestamps.shape)
        features['weekday'] = weekdays
        
        # 月份 (1-12)
        months = torch.tensor([dt.month for dt in datetimes]).reshape(timestamps.shape)
        features['month'] = months
        
        # 季度 (0-3)
        quarters = torch.tensor([(dt.month - 1) // 3 for dt in datetimes]).reshape(timestamps.shape)
        features['quarter'] = quarters
        
        return features


class EnhancedMQTokenizer(nn.Module):
    """
    增强版MQ-Tokenizer,支持时间特征
    """
    
    def __init__(self, input_dim, K=3, L=512, d_c=64, 
                 use_time=False, time_emb_dim=16, mask_ratio=0.2):
        super().__init__()
        self.use_time = use_time
        
        # 如果使用时间，需要融合时间embedding
        if use_time:
            self.time_encoder = TimeEncoder(emb_dim=time_emb_dim)
            # 融合层:将item embedding和time embedding结合
            self.fusion = nn.Linear(input_dim + time_emb_dim, input_dim)
        
        # 原始的MQ-Tokenizer组件
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
        
        # K-way码本
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
    
    def forward(self, x, timestamps=None):
        """
        Args:
            x: (batch_size, input_dim) - item/user embeddings
            timestamps: (batch_size,) - 可选的时间戳
        """
        # 如果提供了时间信息
        if self.use_time and timestamps is not None:
            # 编码时间
            time_emb = self.time_encoder.timestamp_to_sinusoidal(timestamps)
            # 确保time_emb是2D的
            if len(time_emb.shape) == 3:
                time_emb = time_emb.squeeze(1)
            
            # 融合item embedding和time embedding
            x_with_time = torch.cat([x, time_emb], dim=-1)
            x = self.fusion(x_with_time)
        
        # 后续处理与原始MQ-Tokenizer相同
        # ... (省略，与原代码相同)
        
        return x  # 返回处理后的结果


# ============================================================================
# 改进方案2: 添加物品的文本特征
# ============================================================================

class ItemTextEncoder(nn.Module):
    """物品文本特征编码器"""
    
    def __init__(self, text_model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.model = AutoModel.from_pretrained(text_model_name)
        
        # 冻结预训练模型
        for param in self.model.parameters():
            param.requires_grad = False
    
    def encode_text(self, texts):
        """
        编码物品的标题/描述
        
        Args:
            texts: List[str] - 物品的文本描述
        Returns:
            text_embeddings: (batch_size, hidden_dim)
        """
        # Tokenize
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=128,
            return_tensors='pt'
        )
        
        # Encode
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS] token或mean pooling
            text_embeddings = outputs.last_hidden_state[:, 0, :]
        
        return text_embeddings


class MultiModalItemEncoder(nn.Module):
    """
    多模态物品编码器
    融合:ID embedding + 文本embedding + 时间特征
    """
    
    def __init__(self, id_emb_dim=64, text_emb_dim=384, time_emb_dim=16, 
                 output_dim=64):
        super().__init__()
        
        # 各个模态的投影层
        self.id_proj = nn.Linear(id_emb_dim, output_dim)
        self.text_proj = nn.Linear(text_emb_dim, output_dim)
        self.time_proj = nn.Linear(time_emb_dim, output_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
    
    def forward(self, id_emb, text_emb, time_emb):
        """
        Args:
            id_emb: (batch, id_emb_dim) - 从GNN学到的ID embedding
            text_emb: (batch, text_emb_dim) - 文本embedding
            time_emb: (batch, time_emb_dim) - 时间embedding
        Returns:
            fused_emb: (batch, output_dim) - 融合后的表示
        """
        # 投影到相同维度
        id_proj = self.id_proj(id_emb)
        text_proj = self.text_proj(text_emb)
        time_proj = self.time_proj(time_emb)
        
        # 拼接
        concat = torch.cat([id_proj, text_proj, time_proj], dim=-1)
        
        # 融合
        fused_emb = self.fusion(concat)
        
        return fused_emb


# ============================================================================
# 完整的数据加载示例(包含时间信息)
# ============================================================================

class AmazonDatasetWithTime:
    """
    增强版Amazon数据集加载器
    包含时间戳、文本等额外信息
    """
    
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
    
    def load_data(self, path):
        """
        加载Amazon数据集并保留所有信息
        
        返回的数据结构:
        {
            'user_id': [...],
            'item_id': [...],
            'timestamp': [...],  # 新增！
            'rating': [...],
            'item_title': [...],  # 新增！
            'item_description': [...],  # 新增！
        }
        """
        import json
        import gzip
        
        data = []
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                review = json.loads(line)
                data.append({
                    'user_id': review['reviewerID'],
                    'item_id': review['asin'],
                    'timestamp': review['unixReviewTime'],  # 保留时间戳！
                    'rating': review['overall'],
                    # 如果有的话:
                    # 'item_title': review.get('title', ''),
                    # 'item_description': review.get('description', ''),
                })
        
        return data
    
    def create_sequences_with_time(self, user_interactions):
        """
        创建包含时间信息的交互序列
        
        Args:
            user_interactions: List[(item_id, timestamp)]
        Returns:
            item_seq: [item1, item2, ...]
            time_seq: [t1, t2, ...]
            time_intervals: [t2-t1, t3-t2, ...]  # 时间间隔
        """
        # 按时间排序
        sorted_interactions = sorted(user_interactions, key=lambda x: x[1])
        
        item_seq = [item for item, _ in sorted_interactions]
        time_seq = [time for _, time in sorted_interactions]
        
        # 计算时间间隔(相对时间)
        time_intervals = []
        for i in range(1, len(time_seq)):
            interval = time_seq[i] - time_seq[i-1]
            time_intervals.append(interval)
        
        return item_seq, time_seq, time_intervals


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """展示如何使用时间特征"""
    
    print("=" * 60)
    print("示例:添加时间特征到TokenRec")
    print("=" * 60)
    
    # 1. 创建时间编码器
    time_encoder = TimeEncoder(emb_dim=16)
    
    # 2. 模拟时间戳数据
    # 假设是2023年的某些时间点
    timestamps = torch.tensor([
        1672531200,  # 2023-01-01
        1675209600,  # 2023-02-01
        1677628800,  # 2023-03-01
    ]).float()
    
    print("\n时间戳:", timestamps)
    
    # 3. 编码时间
    time_emb = time_encoder.timestamp_to_sinusoidal(timestamps.unsqueeze(0))
    print(f"时间embedding形状: {time_emb.shape}")
    
    # 4. 提取时间特征
    time_features = time_encoder.extract_time_features(timestamps)
    print("\n提取的时间特征:")
    for key, value in time_features.items():
        print(f"  {key}: {value}")
    
    # 5. 创建增强版tokenizer
    print("\n创建增强版MQ-Tokenizer...")
    enhanced_tokenizer = EnhancedMQTokenizer(
        input_dim=64,
        use_time=True,
        time_emb_dim=16
    )
    
    # 6. 测试
    item_emb = torch.randn(3, 64)
    output = enhanced_tokenizer(item_emb, timestamps.unsqueeze(0))
    print(f"输出形状: {output.shape}")
    
    print("\n✓ 时间特征集成成功！")


if __name__ == "__main__":
    example_usage()
    
    print("\n" + "=" * 60)
    print("总结:TokenRec可以这样扩展")
    print("=" * 60)
    print("""
    1. 时间特征:
       - Sinusoidal编码(类似Transformer位置编码)
       - 离散化为时间bins
       - 提取小时/星期/月份等
    
    2. 物品文本特征:
       - 使用Sentence-BERT编码标题/描述
       - 与ID embedding融合
    
    3. 用户特征:
       - 年龄、性别等人口统计学特征
       - 历史行为的统计特征(活跃度、多样性等)
    
    4. 交互特征:
       - 时间间隔(两次交互的间隔)
       - 评分信息
       - 交互类型(点击/购买/评论等)
    
    这些都可以作为额外的输入, 与GNN学到的embedding融合!
    """)