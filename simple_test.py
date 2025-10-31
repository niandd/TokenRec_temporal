"""
TokenRec测试脚本 - 逐步验证每个模块
运行此脚本来检查代码是否可以正常工作
"""

import torch
import numpy as np
import sys
import os
from datetime import datetime

def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_dependencies():
    """检查依赖包"""
    print_section("Step 0: 检查依赖包")
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'transformers': 'Hugging Face Transformers',
        'tqdm': 'tqdm'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name:30s} - 已安装")
        except ImportError:
            print(f"✗ {name:30s} - 未安装")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"\n✓ CUDA 可用")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n⚠️  CUDA 不可用，将使用CPU (训练会很慢)")
    
    return True

def test_mq_tokenizer():
    """测试MQ-Tokenizer"""
    print_section("Step 1: 测试MQ-Tokenizer")
    
    try:
        from tokenrec_core import MQTokenizer
        
        # 创建测试数据
        batch_size = 16
        emb_dim = 64
        test_embeddings = torch.randn(batch_size, emb_dim)
        
        print(f"输入形状: {test_embeddings.shape}")
        
        # 初始化tokenizer
        tokenizer = MQTokenizer(
            input_dim=emb_dim,
            K=3,
            L=128,  # 减小以加快测试
            d_c=32,
            mask_ratio=0.2
        )
        
        print(f"✓ MQTokenizer初始化成功")
        print(f"  参数量: {sum(p.numel() for p in tokenizer.parameters()):,}")
        
        # 前向传播
        tokens, reconstructed, loss_dict = tokenizer(test_embeddings)
        
        print(f"✓ 前向传播成功")
        print(f"  Tokens形状: {tokens.shape}")
        print(f"  重建形状: {reconstructed.shape}")
        print(f"  重建损失: {loss_dict['recon_loss'].item():.4f}")
        print(f"  码本损失: {loss_dict['codebook_loss'].item():.4f}")
        print(f"  承诺损失: {loss_dict['commitment_loss'].item():.4f}")
        
        # 测试训练步骤
        optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)
        loss = loss_dict['recon_loss'] + loss_dict['codebook_loss'] + 0.25 * loss_dict['commitment_loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✓ 反向传播成功")
        
        return True
        
    except Exception as e:
        print(f"✗ MQTokenizer测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_tokenrec_model():
    """测试TokenRec模型"""
    print_section("Step 2: 测试TokenRec模型")
    
    try:
        from tokenrec_core import MQTokenizer, TokenRec
        
        # 模拟参数
        num_users = 100
        num_items = 50
        emb_dim = 64
        
        # 创建tokenizers
        user_tokenizer = MQTokenizer(emb_dim, K=2, L=64, d_c=32)
        item_tokenizer = MQTokenizer(emb_dim, K=2, L=64, d_c=32)
        
        print(f"✓ Tokenizers创建成功")
        
        # 创建TokenRec (使用小模型测试)
        print("正在加载T5模型 (可能需要几分钟)...")
        model = TokenRec(
            user_tokenizer,
            item_tokenizer,
            llm_model_name='t5-small',
            item_emb_dim=emb_dim
        )
        
        print(f"✓ TokenRec初始化成功")
        print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        batch_size = 4
        user_emb = torch.randn(batch_size, emb_dim)
        
        # 不使用历史
        print("\n测试1: 不使用交互历史")
        z = model(user_emb, None)
        print(f"✓ 输出形状: {z.shape}")
        
        # 使用历史
        print("\n测试2: 使用交互历史")
        seq_len = 5
        item_history = torch.randn(batch_size, seq_len, emb_dim)
        z = model(user_emb, item_history)
        print(f"✓ 输出形状: {z.shape}")
        
        # 测试检索
        print("\n测试3: Top-K检索")
        item_database = torch.randn(num_items, emb_dim)
        top_k_indices, top_k_scores = model.retrieve_top_k(z, item_database, k=10)
        print(f"✓ Top-K形状: {top_k_indices.shape}")
        print(f"  检索的物品ID: {top_k_indices[0].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"✗ TokenRec测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_lightgcn():
    """测试LightGCN"""
    print_section("Step 3: 测试LightGCN")
    
    try:
        from lightgcn import LightGCN
        
        num_users = 100
        num_items = 50
        emb_dim = 32
        
        # 创建模型
        model = LightGCN(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=emb_dim,
            num_layers=2
        )
        
        print(f"✓ LightGCN初始化成功")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 创建测试邻接矩阵
        total_nodes = num_users + num_items
        num_edges = 200
        
        # 随机生成边
        row = torch.randint(0, total_nodes, (num_edges,))
        col = torch.randint(0, total_nodes, (num_edges,))
        values = torch.ones(num_edges)
        
        adj_matrix = torch.sparse_coo_tensor(
            torch.stack([row, col]),
            values,
            (total_nodes, total_nodes)
        )
        
        print(f"✓ 邻接矩阵创建成功: {adj_matrix.shape}")
        
        # 前向传播
        user_emb, item_emb = model(adj_matrix)
        
        print(f"✓ 前向传播成功")
        print(f"  用户embeddings: {user_emb.shape}")
        print(f"  物品embeddings: {item_emb.shape}")
        
        # 测试BPR损失
        users = torch.randint(0, num_users, (16,))
        pos_items = torch.randint(0, num_items, (16,))
        neg_items = torch.randint(0, num_items, (16,))
        
        bpr_loss = model.bpr_loss(users, pos_items, neg_items, user_emb, item_emb)
        reg_loss = model.reg_loss(users, pos_items, neg_items)
        
        print(f"✓ 损失计算成功")
        print(f"  BPR Loss: {bpr_loss.item():.4f}")
        print(f"  Reg Loss: {reg_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ LightGCN测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loader():
    """测试数据加载器"""
    print_section("Step 4: 测试数据加载")
    
    try:
        from amazon_loader import AmazonDataLoader, BipartiteGraph
        
        print("注意: 此步骤需要下载Amazon数据集")
        print("如果数据集不存在，将跳过此测试\n")
        
        loader = AmazonDataLoader(category='Beauty', data_dir='./tokenrec_project/data')
        
        # 检查数据文件是否存在
        if not os.path.exists(loader.raw_data_path):
            print(f"⚠️  数据文件不存在: {loader.raw_data_path}")
            print("请手动下载数据集或运行下载函数")
            return None
        
        # 加载数据
        data = loader.load_and_preprocess()
        
        print(f"✓ 数据加载成功")
        print(f"  用户数: {data['num_users']}")
        print(f"  物品数: {data['num_items']}")
        print(f"  训练交互: {len(data['train_df'])}")
        
        # 构建图
        graph = BipartiteGraph(
            num_users=data['num_users'],
            num_items=data['num_items'],
            train_interactions=data['train_interactions']
        )
        
        adj_matrix, edge_index = graph.to_torch_sparse_coo()
        
        print(f"✓ 图构建成功")
        print(f"  邻接矩阵: {adj_matrix.shape}")
        print(f"  边数量: {edge_index.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  数据加载测试跳过: {str(e)}")
        return None

def test_mini_pipeline():
    """测试小规模端到端流程"""
    print_section("Step 5: 端到端小规模测试")
    
    try:
        from tokenrec_core import MQTokenizer, TokenRec
        
        print("创建模拟数据...")
        
        # 小规模数据
        num_users = 50
        num_items = 30
        emb_dim = 32
        
        # 模拟GNN embeddings
        user_embeddings = torch.randn(num_users, emb_dim)
        item_embeddings = torch.randn(num_items, emb_dim)
        
        print(f"✓ 数据创建成功")
        
        # 步骤1: 训练User Tokenizer
        print("\n1. 训练User Tokenizer (10 epochs)...")
        user_tokenizer = MQTokenizer(emb_dim, K=2, L=32, d_c=16, mask_ratio=0.2)
        optimizer = torch.optim.Adam(user_tokenizer.parameters(), lr=1e-3)
        
        for epoch in range(10):
            tokens, recon, losses = user_tokenizer(user_embeddings)
            loss = losses['recon_loss'] + losses['codebook_loss'] + 0.25 * losses['commitment_loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")
        
        print("✓ User Tokenizer训练完成")
        
        # 步骤2: 训练Item Tokenizer
        print("\n2. 训练Item Tokenizer (10 epochs)...")
        item_tokenizer = MQTokenizer(emb_dim, K=2, L=32, d_c=16, mask_ratio=0.2)
        optimizer = torch.optim.Adam(item_tokenizer.parameters(), lr=1e-3)
        
        for epoch in range(10):
            tokens, recon, losses = item_tokenizer(item_embeddings)
            loss = losses['recon_loss'] + losses['codebook_loss'] + 0.25 * losses['commitment_loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")
        
        print("✓ Item Tokenizer训练完成")
        
        # 步骤3: 创建TokenRec
        print("\n3. 创建TokenRec模型...")
        model = TokenRec(user_tokenizer, item_tokenizer, 
                        llm_model_name='t5-small', item_emb_dim=emb_dim)
        
        # 冻结tokenizers
        for param in model.user_tokenizer.parameters():
            param.requires_grad = False
        for param in model.item_tokenizer.parameters():
            param.requires_grad = False
        
        print("✓ TokenRec创建成功")
        
        # 步骤4: 测试训练步骤
        print("\n4. 测试训练步骤 (5 steps)...")
        optimizer = torch.optim.Adam([
            {'params': model.llm.parameters(), 'lr': 1e-4},
            {'params': model.projection.parameters(), 'lr': 1e-3}
        ])
        
        batch_size = 8
        for step in range(5):
            # 随机采样
            user_idx = torch.randint(0, num_users, (batch_size,))
            pos_idx = torch.randint(0, num_items, (batch_size,))
            neg_idx = torch.randint(0, num_items, (batch_size,))
            
            user_emb = user_embeddings[user_idx]
            pos_emb = item_embeddings[pos_idx]
            neg_emb = item_embeddings[neg_idx]
            
            # 前向传播
            z = model(user_emb, None)
            
            # 计算损失
            loss = model.compute_ranking_loss(z, pos_emb, neg_emb, margin=0.1)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"  Step {step+1}: Loss = {loss.item():.4f}")
        
        print("✓ 训练步骤测试完成")
        
        # 步骤5: 测试推理
        print("\n5. 测试推理...")
        model.eval()
        with torch.no_grad():
            test_user_emb = user_embeddings[:5]
            z = model(test_user_emb, None)
            top_k_indices, top_k_scores = model.retrieve_top_k(z, item_embeddings, k=10)
        
        print(f"✓ 推理成功")
        print(f"  为{len(test_user_emb)}个用户推荐了top-10物品")
        print(f"  示例推荐: {top_k_indices[0].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"✗ 端到端测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("  TokenRec 代码测试脚本")
    print("  开始时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    results = {}
    
    # 0. 检查依赖
    results['dependencies'] = check_dependencies()
    if not results['dependencies']:
        print("\n❌ 依赖检查失败，请先安装缺失的包")
        return
    
    # 1. 测试MQ-Tokenizer
    results['mq_tokenizer'] = test_mq_tokenizer()
    
    # 2. 测试TokenRec
    results['tokenrec'] = test_tokenrec_model()
    
    # 3. 测试LightGCN
    results['lightgcn'] = test_lightgcn()
    
    # 4. 测试数据加载 (可选)
    results['data_loader'] = test_data_loader()
    
    # 5. 端到端测试
    results['end_to_end'] = test_mini_pipeline()
    
    # 总结
    print_section("测试总结")
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ 通过"
        elif result is False:
            status = "✗ 失败"
        else:
            status = "⊘ 跳过"
        
        print(f"{test_name:20s}: {status}")
    
    passed = sum(1 for r in results.values() if r is True)
    total = len([r for r in results.values() if r is not None])
    
    print(f"\n通过率: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 所有测试通过！代码可以正常运行。")
        print("   现在可以运行完整的训练流程。")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息并修复。")
    
    print("\n结束时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()