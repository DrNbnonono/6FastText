import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math
import sys
import argparse
from torch.autograd import Variable
import ipaddress

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """基于Transformer的IPv6地址生成模型"""
    def __init__(self, vocab_size, d_model, nhead, dim_feedforward, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer模型
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.generator = nn.Linear(d_model, vocab_size)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, tgt_mask=None):
        """前向传播"""
        # 编码
        memory = self.encode(src, None, src_padding_mask)
        
        # 解码
        output = self.decode(memory, tgt, tgt_mask, None, tgt_padding_mask, src_padding_mask)
        
        # 生成
        return self.generator(output)
    
    def encode(self, src, src_mask=None, src_padding_mask=None):
        """编码器部分"""
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        return self.transformer.encoder(src, mask=src_mask, src_key_padding_mask=src_padding_mask)
    
    def decode(self, memory, tgt, tgt_mask=None, memory_mask=None, 
               tgt_padding_mask=None, memory_padding_mask=None):
        """解码器部分"""
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        return self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask, 
                                       memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_padding_mask,
                                       memory_key_padding_mask=memory_padding_mask)

    def get_attention_weights(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        获取注意力权重
        
        Args:
            src: 源序列
            tgt: 目标序列
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        
        Returns:
            attention_weights: 注意力权重列表，每个元素是一个层的注意力权重
        """
        # 嵌入和位置编码
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        # 存储注意力权重
        attention_weights = []
        
        # 由于PyTorch的Transformer不直接暴露注意力权重，我们需要修改forward方法来获取它们
        # 这里我们使用一个钩子来获取注意力权重
        
        # 创建一个钩子函数来捕获注意力权重
        def get_attention_hook(module, input, output):
            # 注意力权重通常是输出的第1个元素
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights.append(output[1])
        
        # 为每个编码器层的自注意力模块注册钩子
        hooks = []
        for i, layer in enumerate(self.transformer.encoder.layers):
            hook = layer.self_attn.register_forward_hook(get_attention_hook)
            hooks.append(hook)
        
        # 为每个解码器层的自注意力和交叉注意力模块注册钩子
        for i, layer in enumerate(self.transformer.decoder.layers):
            hook = layer.self_attn.register_forward_hook(get_attention_hook)
            hooks.append(hook)
            hook = layer.multihead_attn.register_forward_hook(get_attention_hook)
            hooks.append(hook)
        
        # 前向传播以获取注意力权重
        memory = self.transformer.encoder(src, mask=src_mask)
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return attention_weights

class FastTextModelLoader:
    """FastText模型加载器，用于加载预训练的FastText模型和词嵌入"""
    def __init__(self, model_path=None, embeddings_path=None, vocab_path=None):
        self.model = None
        self.embeddings = None
        self.word2id = None
        self.id2word = None
        self.state_dict = None
        
        # 加载词嵌入
        if embeddings_path and os.path.exists(embeddings_path):
            self.load_embeddings(embeddings_path)
            logging.info(f"已加载词嵌入: {embeddings_path}")
        
        # 加载词汇表
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocabulary(vocab_path)
            logging.info(f"已加载词汇表: {vocab_path}")
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logging.info(f"已加载模型: {model_path}")
    
    def load_embeddings(self, embeddings_path):
        """加载预训练的词嵌入"""
        try:
            self.embeddings = np.load(embeddings_path, allow_pickle=True).item()
            logging.info(f"成功加载词嵌入，包含 {len(self.embeddings)} 个词向量")
        except Exception as e:
            logging.error(f"加载词嵌入失败: {e}")
            self.embeddings = {}
    
    def load_vocabulary(self, vocab_path):
        """加载词汇表"""
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.word2id = json.load(f)
            
            # 创建id到word的映射
            self.id2word = {v: k for k, v in self.word2id.items()}
            logging.info(f"成功加载词汇表，包含 {len(self.word2id)} 个词")
        except Exception as e:
            logging.error(f"加载词汇表失败: {e}")
            self.word2id = {}
            self.id2word = {}
    
    def load_model(self, model_path):
        """加载预训练的模型"""
        try:
            # 加载模型状态字典
            self.state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            logging.info(f"已加载模型状态字典: {model_path}")
        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            self.state_dict = None
    
    def get_word_embedding(self, word):
        """获取词的嵌入向量"""
        if self.embeddings and word in self.embeddings:
            return self.embeddings[word]
        return None
    
    def get_word_id(self, word):
        """获取词的ID"""
        if self.word2id:
            return self.word2id.get(word, self.word2id.get("[UNK]", 0))
        return 0
    
    def get_id_word(self, idx):
        """根据ID获取词"""
        if self.id2word:
            return self.id2word.get(idx, "[UNK]")
        return "[UNK]"

def next_word_sampling(model, loader, src, src_padding_mask, current_tokens, temperature=0.1, device='cpu'):
    """
    基于模型输出采样下一个词
    
    Args:
        model: Transformer模型
        loader: FastTextModelLoader实例
        src: 源序列
        src_padding_mask: 源序列填充掩码
        current_tokens: 当前已生成的序列
        temperature: 温度参数，控制采样随机性
        device: 设备
    
    Returns:
        下一个词的ID
    """
    model.eval()
    
    with torch.no_grad():
        # 将当前序列转换为张量
        tgt = torch.LongTensor(current_tokens).unsqueeze(0).to(device)
        
        # 创建目标掩码 - 注意这里使用的是方形掩码
        seq_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(device)
        
        # 前向传播
        output = model(src, tgt, src_padding_mask, None, tgt_mask)
        
        # 获取最后一个时间步的输出
        logits = output[:, -1, :]
        
        # 应用温度
        probs = F.softmax(logits / temperature, dim=-1)
        
        # 采样
        next_token = torch.multinomial(probs, 1).item()
        
        return next_token

def greedy_decode(model, loader, src, src_padding_mask, max_len, start_symbol, temperature=0.1, device='cpu'):
    """
    贪婪解码生成序列
    
    Args:
        model: Transformer模型
        loader: FastTextModelLoader实例
        src: 源序列
        src_padding_mask: 源序列填充掩码
        max_len: 最大生成长度
        start_symbol: 起始符号
        temperature: 温度参数
        device: 设备
    
    Returns:
        生成的序列
    """
    model.eval()
    
    with torch.no_grad():
        # 初始化目标序列
        current_tokens = [start_symbol]
        
        for i in range(max_len-1):
            # 采样下一个词
            next_token = next_word_sampling(
                model, loader, src, src_padding_mask, current_tokens, temperature, device
            )
            
            # 将新生成的词添加到序列中
            current_tokens.append(next_token)
            
            # 如果生成了结束符，提前结束
            if next_token == loader.get_word_id("[EOS]"):
                break
    
    return current_tokens

def beam_search_decode(model, loader, src, src_padding_mask, max_len, start_symbol, beam_size=5, temperature=0.1, device='cpu'):
    """
    束搜索解码生成序列
    
    Args:
        model: Transformer模型
        loader: FastTextModelLoader实例
        src: 源序列
        src_padding_mask: 源序列填充掩码
        max_len: 最大生成长度
        start_symbol: 起始符号
        beam_size: 束大小
        temperature: 温度参数
        device: 设备
    
    Returns:
        生成的序列列表
    """
    model.eval()
    
    with torch.no_grad():
        # 初始化候选序列
        sequences = [([start_symbol], 0.0)]
        
        for _ in range(max_len-1):
            all_candidates = []
            
            # 扩展每个当前候选
            for seq, score in sequences:
                if len(seq) == max_len or seq[-1] == loader.get_word_id("[EOS]"):
                    all_candidates.append((seq, score))
                    continue
                
                # 将当前序列转换为张量
                tgt = torch.LongTensor(seq).unsqueeze(0).to(device)
                
                # 创建目标掩码 - 使用方形掩码
                seq_len = tgt.size(1)
                tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(device)
                
                # 前向传播
                output = model(src, tgt, src_padding_mask, None, tgt_mask)
                
                # 获取最后一个时间步的输出
                logits = output[:, -1, :]
                
                # 应用温度
                probs = F.softmax(logits / temperature, dim=-1)
                
                # 获取top-k个候选
                topk_probs, topk_indices = torch.topk(probs, beam_size)
                
                for i in range(beam_size):
                    next_token = topk_indices[0, i].item()
                    next_prob = topk_probs[0, i].item()
                    
                    # 创建新序列
                    new_seq = seq + [next_token]
                    
                    # 更新分数
                    new_score = score + math.log(next_prob)
                    
                    all_candidates.append((new_seq, new_score))
            
            # 选择top-k个候选
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
    
    return sequences

def generate_addresses(model, loader, seed_data, num_samples=1000, max_len=32, temperature=0.1, beam_size=None, device='cpu'):
    """
    生成IPv6地址
    
    Args:
        model: Transformer模型
        loader: FastTextModelLoader实例
        seed_data: 种子数据
        num_samples: 生成样本数量
        max_len: 最大生成长度
        temperature: 温度参数
        beam_size: 束搜索大小，None表示使用贪婪搜索
        device: 设备
    
    Returns:
        生成的IPv6地址列表
    """
    generated_addresses = []
    
    # 确定编码器输入长度
    encoder_input_length = 16  # 默认值，可以根据需要调整
    
    for i in tqdm(range(min(num_samples, len(seed_data))), desc="生成地址"):
        # 准备输入数据
        src_words = seed_data[i][:encoder_input_length]
        src_ids = [loader.get_word_id(w) for w in src_words]
        src = torch.LongTensor([src_ids]).to(device)
        
        # 创建源序列掩码 - 修复掩码格式
        src_padding_mask = (src == loader.get_word_id("[PAD]"))
        
        # 获取起始符号
        start_symbol = loader.get_word_id(seed_data[i][encoder_input_length]) if len(seed_data[i]) > encoder_input_length else loader.get_word_id("[PAD]")
        
        # 生成序列
        if beam_size:
            sequences = beam_search_decode(
                model, loader, src, src_padding_mask, max_len - encoder_input_length, 
                start_symbol, beam_size, temperature, device
            )
            # 取分数最高的序列
            predict = sequences[0][0]
        else:
            predict = greedy_decode(
                model, loader, src, src_padding_mask, max_len - encoder_input_length, 
                start_symbol, temperature, device
            )
        
        # 合并源序列和生成序列
        full_sequence = src_ids + predict[1:]  # 去掉起始符号
        
        # 将ID转换为词
        predict_words = [loader.get_id_word(idx) for idx in full_sequence]
        
        # 构建IPv6地址字符串
        address_parts = []
        current_part = ""
        
        for word in predict_words:
            if word == "[PAD]" or word == "[UNK]" or word == "[EOS]":
                continue
            current_part += word
            if len(current_part) == 4:
                address_parts.append(current_part)
                current_part = ""
        
        # 处理最后一部分（如果有）
        if current_part:
            address_parts.append(current_part)
        
        # 构建完整地址
        ipv6_address = ":".join(address_parts)
        
        # 验证地址格式
        try:
            ipaddress.IPv6Address(ipv6_address)
            generated_addresses.append(ipv6_address)
        except:
            # 跳过无效地址
            continue
    
    # 去重
    generated_addresses = list(set(generated_addresses))
    
    return generated_addresses

def write_data(addresses, output_path):
    """将生成的地址写入文件"""
    with open(output_path, "w") as f:
        for address in addresses:
            f.write(address + "\n")
    logging.info(f"已将 {len(addresses)} 个地址写入: {output_path}")

def load_seed_data(data_path, max_samples=100000):
    """加载种子数据"""
    seed_data = []
    with open(data_path, "r") as f:
        for line in f:
            words = line.strip().split()
            if words:
                seed_data.append(words)
            if len(seed_data) >= max_samples:
                break
    return seed_data

def main():
    parser = argparse.ArgumentParser(description="生成IPv6地址")
    parser.add_argument("--model_path", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/FastText/FastTextmodels/transformer/ipv6_transformer.pt", help="模型路径")
    parser.add_argument("--vocab_path", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/FastText/data/processed/vocabulary.json", help="词汇表路径")
    parser.add_argument("--embeddings_path", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/FastText/FastTextmodels/embedding/ipv6_embeddings.npy", help="词嵌入路径")
    parser.add_argument("--data_path", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/FastText/data/processed/word_sequences.txt", help="种子数据路径")
    parser.add_argument("--output_dir", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/FastText/data/generated", help="输出目录")
    parser.add_argument("--num_samples", type=int, default=10000, help="生成样本数量")
    parser.add_argument("--temperatures", type=str, default="0.1,0.2,0.5,1.0", help="温度参数列表，用逗号分隔")
    parser.add_argument("--beam_size", type=int, default=None, help="束搜索大小，None表示使用贪婪搜索")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型和数据
    loader = FastTextModelLoader(
        model_path=args.model_path,
        embeddings_path=args.embeddings_path,
        vocab_path=args.vocab_path
    )
    
    # 加载种子数据
    seed_data = load_seed_data(args.data_path, args.num_samples)
    logging.info(f"已加载 {len(seed_data)} 个种子数据")
    
    # 解析温度参数
    temperatures = [float(t) for t in args.temperatures.split(",")]
    
    # 加载模型配置 - 确保与训练时使用的配置一致
    config = {
        "d_model": 100,  # 与FastText词嵌入维度一致
        "nhead": 10,     # 注意力头数量
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,  # 这里必须与训练时一致
        "dropout": 0.1,
    }
    
    # 创建模型
    vocab_size = len(loader.word2id)
    model = TransformerModel(
        vocab_size, 
        config["d_model"], 
        config["nhead"], 
        config["dim_feedforward"],
        config["num_encoder_layers"], 
        config["num_decoder_layers"], 
        config["dropout"]
    )
    
    # 加载预训练权重
    try:
        model.load_state_dict(loader.state_dict)
        logging.info("成功加载模型权重")
    except Exception as e:
        logging.error(f"加载模型权重失败: {e}")
        return
    
    model.to(args.device)
    model.eval()
    
    # 对每个温度参数生成地址
    for temp in temperatures:
        logging.info(f"使用温度 {temp} 生成地址")
        
        # 生成地址
        addresses = generate_addresses(
            model, 
            loader, 
            seed_data, 
            args.num_samples, 
            temperature=temp,
            beam_size=args.beam_size,
            device=args.device
        )
        
        # 写入文件
        output_path = os.path.join(args.output_dir, f"generated_addresses_t{temp:.3f}.txt")
        write_data(addresses, output_path)

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        logging.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("未检测到GPU，将使用CPU生成")
    
    main()