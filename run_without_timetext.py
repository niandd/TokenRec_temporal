import os
import sys
import torch
from tokenrec_training import TokenRecPipeline


def _setup_logging(project_root):
    """将 stdout/stderr 重定向到项目根目录下的日志文件(覆盖写入)。"""
    log_path = os.path.join(project_root, 'logs_without_timetext.log')
    # 覆盖写入，确保每次运行都是新的日志
    log_file = open(log_path, 'w')
    sys.stdout = log_file
    sys.stderr = log_file


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    _setup_logging(project_root)
    data_dir = os.path.join(project_root, 'data', 'without_timetext')
    ckpt_dir = os.path.join(project_root, 'ckpt', 'lightgcn', 'without_timetext')
    output_dir = data_dir  # embeddings/tokens 放数据目录；tokenizers/TokenRec ckpt 放 ckpt_dir_tokenrec
    ckpt_dir_tokenrec = os.path.join(project_root, 'ckpt', 'tokenrec', 'without_timetext')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print('=' * 60)
    print('配置与目录')
    print('=' * 60)
    print(f'data_dir: {data_dir}')
    print(f'ckpt_dir_lightgcn: {ckpt_dir}')
    print(f'output_dir: {output_dir}')

    # 配置
    config = {
        'dataset': 'Beauty',
        'data_dir': data_dir,
        'output_dir': output_dir,
        'ckpt_dir_lightgcn': ckpt_dir,
        'ckpt_dir_tokenrec': ckpt_dir_tokenrec,
        'use_time': False,
        'use_text': False,
        'export_tokens': True,
        # GNN
        'gnn_emb_dim': 64,
        'gnn_layers': 3,
        'gnn_epochs': 20,
        'gnn_batch_size': 2048,
        'gnn_lr': 1e-3,
        'gnn_reg': 1e-4,
        # MQ-Tokenizer
        'K': 3,
        'L': 512,
        'd_c': 64,
        'mask_ratio': 0.2,
        'beta_user': 0.25,
        'beta_item': 0.25,
        'tokenizer_epochs': 50,
        'tokenizer_batch_size': 256,
        # TokenRec
        'llm_model': 't5-small',
        'max_seq_len': 100,
        'tokenrec_epochs': 10,
        'tokenrec_batch_size': 32,
        'tokenrec_lr': 1e-4,
        'margin': 0.1,
    }

    pipeline = TokenRecPipeline(config)

    # 1) 数据
    data = pipeline.step1_load_data()
    print('\n数据集下载与预处理完成，指标:')
    print(f"  用户数: {data['num_users']}")
    print(f"  物品数: {data['num_items']}")
    print(f"  训练交互数: {len(data['train_df'])}")
    print(f"  验证交互数: {len(data['val_df'])}")
    print(f"  测试交互数: {len(data['test_df'])}")

    # 2) 图
    pipeline.step2_build_graph()

    # 3) 训练 LightGCN 并保存 best ckpt 与 embeddings
    pipeline.step3_train_gnn()
    print('\nLightGCN 训练完成并已保存:')
    print(f"  best ckpt: {os.path.join(ckpt_dir, 'best_lightgcn.pth')}")
    print(f"  embeddings: {os.path.join(output_dir, 'gnn_embeddings.pkl')}")

    # 4) 训练 MQ-Tokenizers（并导出 token IDs）
    pipeline.step4_train_tokenizers()
    print('\nMQ-Tokenizer 训练完成:')
    print(f"  tokenizers: {os.path.join(output_dir, 'tokenizers.pkl')}")
    print(f"  tokens_all: {os.path.join(output_dir, 'tokens_all.pt')}")

    # 5) 训练 TokenRec（期间会定期打印 Recall/NDCG 指标，衡量是否收敛）
    model = pipeline.step5_train_tokenrec()
    print('\nTokenRec 训练完成。')


if __name__ == '__main__':
    main()



