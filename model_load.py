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

# 添加项目根目录到路径，以便导入Transformertrain模块
sys.path.append("d:/bigchuang/ipv6地址论文/10-6VecLM/FastText")
from Transformertrain import TransformerModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            # 检查加载的是模型对象还是状态字典
            if isinstance(state_dict, dict) and not hasattr(state_dict, 'eval'):
                # 如果是状态字典，先不设置模型，等到main函数中创建模型后再加载
                self.state_dict = state_dict
                self.model = None
                logging.info(f"已加载模型状态字典: {model_path}")
            else:
                # 如果是完整模型对象
                self.model = state_dict
                self.model.eval()
                logging.info(f"已加载完整模型: {model_path}")
        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            self.model = None
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

def next_generation(embeddings, vector, temperature=0.1, position=0):
    """
    基于词嵌入相似度生成下一个词
    
    Args:
        embeddings: 词嵌入字典
        vector: 当前输出向量
        temperature: 温度参数，控制采样随机性
        position: 当前生成的位置
    
    Returns:
        下一个词的ID
    """
    # 将所有词嵌入转换为矩阵形式
    word_list = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[word] for word in word_list])
    
    # 计算余弦相似度
    vector_norm = vector / np.linalg.norm(vector)
    embedding_matrix_norm = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    similarity = np.dot(embedding_matrix_norm, vector_norm)
    
    # 应用温度
    similarity = np.exp(similarity / temperature)
    similarity = similarity / np.sum(similarity)
    
    # 采样
    next_word_idx = np.random.choice(len(word_list), p=similarity)
    next_word = word_list[next_word_idx]
    
    return next_word

def greedy_decode(model, loader, src, src_mask, max_len, start_symbol, temperature=0.1, device='cpu'):
    """
    贪婪解码生成序列 (已修改，直接使用模型输出 logits)
    
    Args:
        model: Transformer模型
        loader: FastTextModelLoader实例
        src: 源序列
        src_mask: 源序列掩码
        max_len: 最大生成长度
        start_symbol: 起始符号
        temperature: 温度参数
        device: 设备
    
    Returns:
        生成的序列
    """
    model.eval()
    
    with torch.no_grad():
        # 编码源序列
        src = src.to(device)
        src_mask = src_mask.to(device)
        
        # 初始化目标序列
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).long()
        
        for i in range(max_len-1):
            # 创建目标掩码 - 使用None让模型自动处理掩码
            tgt_mask = None
            
            # 前向传播
            out = model(src, ys, src_mask, tgt_mask) # out shape: (1, current_seq_len, vocab_size)
            
            # 获取最后一个时间步的输出 logits
            logits = out[:, -1, :] # logits shape: (1, vocab_size)
            
            # 应用温度
            logits = logits / temperature
            
            # 计算概率分布
            probs = F.softmax(logits, dim=-1) # probs shape: (1, vocab_size)
            
            # 从概率分布中采样下一个词的 ID
            # multinomial 需要 2D 输入 (batch_size, num_classes)
            next_word_id = torch.multinomial(probs, num_samples=1).item() # 获取标量 ID
            
            # 将新生成的词添加到序列中
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word_id).long()], dim=1)
    
    return ys

def beam_search_decode(model, loader, src, src_mask, max_len, start_symbol, beam_size=5, temperature=0.1, device='cpu'):
    """
    束搜索解码生成序列
    
    Args:
        model: Transformer模型
        loader: FastTextModelLoader实例
        src: 源序列
        src_mask: 源序列掩码
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
        # 编码源序列
        src = src.to(device)
        src_mask = src_mask.to(device)
        
        # 初始化候选序列
        sequences = [(torch.ones(1, 1).fill_(start_symbol).type_as(src.data).long(), 0.0)]
        
        for _ in range(max_len-1):
            all_candidates = []
            
            # 扩展每个当前候选
            for seq, score in sequences:
                if seq.size(1) == max_len:
                    all_candidates.append((seq, score))
                    continue
                
                # 创建目标掩码 - 使用None让模型自动处理掩码
                tgt_mask = None
                
                # 前向传播
                out = model(src, seq, src_mask, tgt_mask)
                
                # 获取最后一个时间步的输出 logits
                logits = out[:, -1, :]
                
                # 应用温度
                logits = logits / temperature
                
                # 计算概率分布
                probs = F.softmax(logits, dim=-1)
                
                # 获取top-k个候选
                topk = torch.topk(probs, min(beam_size, probs.size(-1)), dim=-1)
                top_probs = topk.values[0].cpu().numpy()
                top_indices = topk.indices[0].cpu().numpy()
                
                for idx, prob in zip(top_indices, top_probs):
                    next_word_id = int(idx)
                    
                    # 创建新序列
                    new_seq = torch.cat([seq, torch.ones(1, 1).type_as(src.data).fill_(next_word_id).long()], dim=1)
                    
                    # 更新分数
                    new_score = score + np.log(prob)
                    
                    all_candidates.append((new_seq, new_score))
            
            # 选择top-k个候选
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
    
    return sequences

def generate_addresses(model, loader, seed_data, num_samples=1000, max_len=32, temperature=0.1, beam_size=None, device='cpu'):
    """
    生成IPv6地址 - 基于前64位预测后64位
    
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
    
    # 固定编码器输入长度为16（对应IPv6地址的前64位）
    encoder_input_length = 16
    
    for i in tqdm(range(min(num_samples, len(seed_data))), desc="生成地址"):
        # 准备输入数据 - 只使用前16个词（前64位）
        if len(seed_data[i]) < encoder_input_length:
            # 跳过长度不足的数据
            continue
            
        src_words = seed_data[i][:encoder_input_length]
        src_ids = [loader.get_word_id(w) for w in src_words]
        src = torch.LongTensor([src_ids]).to(device)
        
        # 创建源序列掩码
        src_padding_mask = (src == loader.get_word_id("[PAD]"))
        
        # 获取起始符号 - 使用第17个词作为起始符号
        if len(seed_data[i]) > encoder_input_length:
            start_symbol = loader.get_word_id(seed_data[i][encoder_input_length])
        else:
            # 如果没有第17个词，使用PAD作为起始符号
            start_symbol = loader.get_word_id("[PAD]")
        
        # 生成序列 - 生成后16个词（后64位）
        if beam_size:
            sequences = beam_search_decode(model, loader, src, src_padding_mask, 16, 
                                          start_symbol, beam_size, temperature, device)
            # 取分数最高的序列
            predict = sequences[0][0].cpu().numpy()[0]
        else:
            predict = greedy_decode(model, loader, src, src_padding_mask, 16, 
                                   start_symbol, temperature, device).cpu().numpy()[0]
        
        # 合并源序列和生成序列 - 前16个词 + 后16个词
        full_sequence = np.concatenate([src_ids, predict[1:]])  # 去掉起始符号
        
        # 将ID转换为词
        predict_words = [loader.get_id_word(idx) for idx in full_sequence]
        
        # 构建IPv6地址字符串
        address_parts = []
        current_part = ""
        
        for word in predict_words:
            if word == "[PAD]" or word == "[UNK]":
                continue
            current_part += word[0] if len(word) > 0 else ""  # 只取第一个字符
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
    parser.add_argument("--num_samples", type=int, default=1000, help="生成样本数量")
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
        "dim_feedforward": 2048,  # 这里必须与训练时一致，错误是由于这个值不匹配导致的
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
    if loader.model is not None:
        # 如果已经加载了完整模型，直接使用
        model = loader.model
    elif hasattr(loader, 'state_dict') and loader.state_dict is not None:
        # 如果加载了状态字典，应用到模型
        model.load_state_dict(loader.state_dict)
    else:
        # 尝试直接从文件加载状态字典
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device)))
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