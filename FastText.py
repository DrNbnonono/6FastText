import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
import json
import os
import numpy as np
from tqdm import tqdm
import time
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FastTextModel(nn.Module):
    """简化的FastText模型，专注于GPU加速"""
    def __init__(self, vocab_size, embedding_dim=100):
        super(FastTextModel, self).__init__()
        self.embedding_dim = embedding_dim
        
        # 主词嵌入
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 初始化权重
        initrange = 0.5 / self.embedding_dim
        self.in_embeddings.weight.data.uniform_(-initrange, initrange)
        self.out_embeddings.weight.data.uniform_(-0, 0)
    
    def forward(self, input_ids, context_ids, negative_ids=None):
        # 获取词嵌入
        input_embeds = self.in_embeddings(input_ids)
        
        # 上下文处理
        if context_ids.dim() == 2:
            # 使用掩码处理填充
            mask = (context_ids != 0).float().unsqueeze(-1)
            context_embeds = self.in_embeddings(context_ids) * mask
            context_embeds = context_embeds.sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        else:
            context_embeds = self.in_embeddings(context_ids).mean(0, keepdim=True)
        
        # 正样本得分
        pos_output = self.out_embeddings(input_ids)
        pos_score = torch.sum(context_embeds * pos_output, dim=1)
        pos_score = F.logsigmoid(pos_score)
        
        # 负样本得分
        neg_score = 0
        if negative_ids is not None:
            neg_output = self.out_embeddings(negative_ids)
            neg_score = torch.bmm(neg_output, context_embeds.unsqueeze(2)).squeeze()
            neg_score = F.logsigmoid(-neg_score).sum(1)
        
        return -(pos_score + neg_score).mean()

class IPv6WordDataset(Dataset):
    """简化的数据集类"""
    def __init__(self, file_path, vocab_path, window_size=5, max_samples=None):
        self.samples = []
        
        # 加载词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)
        
        # 加载地址词序列
        with open(file_path, encoding='utf-8') as f:
            lines = f.readlines()
            if max_samples:
                lines = lines[:max_samples]
                
            for line in tqdm(lines, desc="生成训练样本", ncols=100):
                words = line.strip().split()
                for i in range(len(words)):
                    input_word = words[i]
                    context = words[max(0,i-window_size):i] + words[i+1:min(len(words),i+window_size+1)]
                    if context:
                        self.samples.append((input_word, context))
        
        logging.info(f"生成了 {len(self.samples)} 个训练样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_word, context = self.samples[idx]
        input_id = self.word2id.get(input_word, self.word2id["[UNK]"])
        context_ids = [self.word2id.get(word, self.word2id["[UNK]"]) for word in context]
        
        return {
            "input_id": input_id,
            "context_ids": context_ids
        }

def collate_fn(batch):
    """自定义批处理函数"""
    input_ids = torch.tensor([item["input_id"] for item in batch])
    
    # 获取每个样本的上下文ID列表
    context_ids_list = [item["context_ids"] for item in batch]
    
    # 计算最大上下文长度
    max_context_len = max(len(context) for context in context_ids_list)
    
    # 填充上下文序列
    padded_context_ids = []
    for context in context_ids_list:
        if len(context) < max_context_len:
            padded = context + [0] * (max_context_len - len(context))
        else:
            padded = context
        padded_context_ids.append(padded)
    
    context_ids = torch.tensor(padded_context_ids)
    
    return {
        "input_ids": input_ids,
        "context_ids": context_ids
    }

def visualize_embeddings(word_embeddings, output_path, n_components=2, perplexity=30, max_words=5000):
    """可视化词嵌入"""
    logging.info("开始可视化词嵌入...")
    
    # 提取词向量和对应的词
    words = []
    vectors = []
    
    # 限制可视化的词数量，避免过度拥挤
    count = 0
    for word, vector in word_embeddings.items():
        if word not in ["[PAD]", "[UNK]"]:
            words.append(word)
            vectors.append(vector)
            count += 1
            if count >= max_words:
                break
    
    vectors = np.array(vectors)
    
    # 使用t-SNE降维
    logging.info(f"使用t-SNE将词向量降至{n_components}维...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=1000, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'x': reduced_vectors[:, 0],
        'y': reduced_vectors[:, 1],
        'word': words,
        'nybble': [word[0] for word in words],  # 提取nybble部分
        'position': [word[1:] if len(word) > 1 else "" for word in words]  # 提取位置部分
    })
    
    # 绘制散点图
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='x', y='y', hue='nybble', data=df, palette='tab20', s=50, alpha=0.7)
    plt.title('IPv6地址词向量t-SNE可视化', fontsize=16)
    plt.xlabel('t-SNE维度1', fontsize=12)
    plt.ylabel('t-SNE维度2', fontsize=12)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"可视化结果已保存到 {output_path}")
    
    # 保存降维后的数据
    df.to_csv(os.path.join(os.path.dirname(output_path), "tsne_data.csv"), index=False)
    logging.info(f"降维数据已保存到 {os.path.join(os.path.dirname(output_path), 'tsne_data.csv')}")
    
    return df

def train(config):
    """训练FastText模型，专注于GPU加速"""
    # 设置随机种子
    torch.manual_seed(42)
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        # 启用CUDA优化
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        logging.warning("未检测到GPU，将使用CPU训练")
    
    # 加载词汇表
    with open(config["vocab_path"], "r", encoding="utf-8") as f:
        word2id = json.load(f)
    id2word = {v: k for k, v in word2id.items()}
    vocab_size = len(word2id)
    logging.info(f"词汇表大小: {vocab_size}")
    
    # 创建数据集
    dataset = IPv6WordDataset(
        file_path=config["train_data_path"],
        vocab_path=config["vocab_path"],
        window_size=config["window_size"],
        max_samples=config.get("max_samples", None)
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),  # 启用pin_memory加速数据传输
        drop_last=True  # 丢弃最后一个不完整的批次
    )
    
    # 创建模型并移动到GPU
    model = FastTextModel(
        vocab_size=vocab_size,
        embedding_dim=config["embedding_dim"]
    ).to(device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # 混合精度训练 - 修复弃用警告
    use_amp = config.get("use_amp", True) and torch.cuda.is_available()
    # 使用新的API格式
    scaler = torch.amp.GradScaler() if use_amp else None
    logging.info(f"混合精度训练: {'启用' if use_amp else '禁用'}")
    
    # 负采样器
    def get_negative_samples(batch_size, num_samples, vocab_size, exclude_ids=None):
        """生成负样本ID"""
        neg_ids = torch.randint(2, vocab_size, (batch_size, num_samples), device=device)
        if exclude_ids is not None:
            # 向量化操作：创建掩码标识需要替换的位置
            mask = (neg_ids == exclude_ids.unsqueeze(1))
            # 对需要替换的位置重新采样
            if mask.any():
                replacements = torch.randint(2, vocab_size, (mask.sum(),), device=device)
                neg_ids = neg_ids.masked_scatter(mask, replacements)
        return neg_ids
    
    # 训练循环
    logging.info("开始训练...")
    start_time = time.time()
    
    # 创建进度条
    progress_bar = tqdm(total=len(dataloader) * config["epochs"], desc="训练进度", ncols=100)
    
    # 记录训练损失
    train_losses = []
    
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # 将数据移至GPU
            input_ids = batch["input_ids"].to(device)
            context_ids = batch["context_ids"].to(device)
            
            # 生成负样本
            negative_ids = get_negative_samples(
                input_ids.size(0), 
                config["num_negative"], 
                vocab_size,
                input_ids
            )
            
            # 前向传播和反向传播
            if use_amp:
                # 使用混合精度训练
                with torch.amp.autocast(device_type='cuda'):
                    loss = model(input_ids, context_ids, negative_ids)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(input_ids, context_ids, negative_ids)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 更新进度条
            total_loss += loss.item()
            
            # 获取GPU内存使用情况
            if torch.cuda.is_available() and (batch_idx % 50 == 0 or batch_idx == len(dataloader) - 1):
                gpu_mem_alloc = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
                
                progress_bar.set_postfix({
                    "epoch": f"{epoch+1}/{config['epochs']}",
                    "loss": f"{loss.item():.4f}", 
                    "GPU": f"{gpu_mem_alloc:.1f}GB/{gpu_mem_reserved:.1f}GB"
                })
            
            progress_bar.update(1)
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        logging.info(f"Epoch {epoch+1}/{config['epochs']}, 平均损失: {avg_loss:.4f}")
        
        # 每个epoch结束后清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # 关闭进度条
    progress_bar.close()
    
    # 训练完成
    elapsed_time = time.time() - start_time
    logging.info(f"训练完成，耗时: {elapsed_time:.2f}秒")
    
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.title('FastText模型训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(config["model_save_path"]), "training_loss.png"))
    logging.info(f"训练损失曲线已保存到 {os.path.join(os.path.dirname(config['model_save_path']), 'training_loss.png')}")
    
    # 保存模型
    torch.save(model.state_dict(), config["model_save_path"])
    logging.info(f"模型已保存到 {config['model_save_path']}")
    
    # 提取词嵌入
    logging.info("开始提取词嵌入...")
    word_embeddings = {}
    model.eval()
    
    # 批量处理词嵌入提取，提高效率
    batch_size = 1024
    all_words = list(word2id.items())
    
    with torch.no_grad():
        for i in range(0, len(all_words), batch_size):
            batch_words = all_words[i:i+batch_size]
            word_list = [word for word, _ in batch_words]
            id_list = [idx for _, idx in batch_words]
            
            # 将ID转换为张量并移至GPU
            id_tensor = torch.tensor(id_list, device=device)
            
            # 获取词嵌入
            embeddings = model.in_embeddings(id_tensor).cpu().numpy()
            
            # 保存结果
            for j, word in enumerate(word_list):
                word_embeddings[word] = embeddings[j]
            
            if i % 10000 == 0 or i + batch_size >= len(all_words):
                logging.info(f"已处理 {min(i+batch_size, len(all_words))}/{len(all_words)} 个词嵌入")
    
    # 保存词嵌入
    np.save(config["embeddings_save_path"], word_embeddings)
    logging.info(f"词嵌入已保存到 {config['embeddings_save_path']}")
    
    return model, word_embeddings, word2id, id2word

def main():
    """主函数"""
    # 创建FastText专用文件夹
    fasttext_dir = "d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/models/FastText"
    os.makedirs(fasttext_dir, exist_ok=True)
    
    config = {
        "train_data_path": "d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/data/processed/word_sequences.txt",
        "vocab_path": "d:/bigchuang/ipv6地址论文/10-6VecLM/6VecLM2/data/processed/vocabulary.json",
        "model_save_path": f"{fasttext_dir}/ipv6_fasttext.pt",
        "embeddings_save_path": f"{fasttext_dir}/ipv6_embeddings.npy",
        "visualization_path": f"{fasttext_dir}/embeddings_tsne.png",
        "batch_size": 8192,  # 大批次以充分利用GPU
        "embedding_dim": 100,
        "window_size": 5,
        "num_negative": 5,
        "learning_rate": 0.01,
        "epochs": 5,
        "num_workers": 4,  # 根据CPU核心数调整
        "use_amp": True    # 启用混合精度训练
    }
    
    # 训练模型
    model, word_embeddings, word2id, id2word = train(config)
    
    # 可视化词嵌入
    try:
        visualize_embeddings(
            word_embeddings, 
            config["visualization_path"]
        )
    except Exception as e:
        logging.error(f"可视化过程出错: {e}")
        logging.warning("跳过可视化")

if __name__ == "__main__":
    # 设置CUDA环境变量以优化性能
    if torch.cuda.is_available():
        # 打印GPU信息
        logging.info(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA设备数量: {torch.cuda.device_count()}")
        
        # 启用CUDA优化
        torch.backends.cudnn.benchmark = True
    
    # 启动训练
    main()