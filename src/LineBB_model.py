import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
import os


max_seq_len = 150
os.environ["TORCH_USE_CUDA_DSA"] = "1"


class Model(nn.Module):
    """
    顶层模型：对应论文里的 LineDef 主体。
    - 输入：一个 batch 的文件 -> code_tensor（文件-行-token 的 3D 结构）+ 词级/行级的邻接矩阵
    - 内部：调用 SentenceAttention 模块，先做词级 GCN+注意力，再做行级 GCN+注意力
    - 输出：文件级缺陷概率 + 词级注意力 + 行级注意力 + 行向量
    """
    def __init__(self, vocab_size, embed_dim,
                 word_gcn_hidden_dim,        # 词级 GCN 输出维度
                 sent_gcn_hidden_dim,        # 行级 GCN 输出维度（也是文件向量维度）
                 word_att_dim,               # 词级注意力中间维度
                 sent_att_dim,               # 行级注意力中间维度
                 line_hidden_dim,
                 use_layer_norm, dropout,
                 device):
        super(Model, self).__init__()
        self.device = device

        self.sent_attention = SentenceAttention(
            vocab_size, embed_dim,
            word_gcn_hidden_dim, sent_gcn_hidden_dim,
            word_att_dim, sent_att_dim,
            use_layer_norm, dropout, device)

        self.fc = nn.Linear(sent_gcn_hidden_dim, 1)
        self.sig = nn.Sigmoid()

        self.use_layer_norm = use_layer_norm
        self.dropout = dropout

        # 行级注意力机制
        self.line_attention = LineAttention(
            hidden_size=line_hidden_dim,
            line_in_dim=embed_dim,
            block_in_dim=sent_gcn_hidden_dim,
            use_layer_norm=self.use_layer_norm,
            dropout=self.dropout,
            device=self.device
        )

    def forward(self, code_tensor, code_line_tensor, word_edge, line_edge, basic_block_size, block_neighbors):
        code_lengths = []
        sent_lengths = []

        for file in code_tensor:
            code_line = []                     # 当前文件内每行长度列表
            code_lengths.append(len(file))     # 该文件的总行数
            for line in file:
                code_line.append(len(line))    # 当前行的 token 数
            sent_lengths.append(code_line)     # 加入到所有文件的行长列表中

        code_tensor = code_tensor.type(torch.LongTensor).to(self.device)
        code_lengths = torch.tensor(code_lengths).type(torch.LongTensor).to(self.device)
        sent_lengths = torch.tensor(sent_lengths).type(torch.LongTensor).to(self.device)

        code_embeds, word_att_weights, block_att_weights, blocks_before_gcn = self.sent_attention(
            code_tensor, code_lengths, word_edge, line_edge)

        scores = self.fc(code_embeds)
        final_scrs = self.sig(scores)

        '''
        行级
        '''
        with torch.random.fork_rng(): # 用独立 RNG 包裹行支路
            line_scores = self.line_attention(code_line_tensor, blocks_before_gcn, basic_block_size, block_neighbors)

        return final_scrs, line_scores, block_att_weights, blocks_before_gcn


class SentenceAttention(nn.Module):

    def __init__(self, vocab_size, embed_dim,
                 word_gcn_hidden_dim,
                 sent_gcn_hidden_dim,
                 word_att_dim, sent_att_dim,
                 use_layer_norm, dropout,
                 device):
        super(SentenceAttention, self).__init__()
        self.device = device
        self.emb_dim = embed_dim
        self.dim = sent_gcn_hidden_dim

        # 词级注意力 + 词级 GCN 模块
        self.word_attention = WordAttention(
            vocab_size, embed_dim,
            word_gcn_hidden_dim, word_att_dim,
            use_layer_norm, dropout, device)

        self.gcn = GCNConv(word_gcn_hidden_dim, sent_gcn_hidden_dim)
        self.gcn1 = GCNConv(sent_gcn_hidden_dim, sent_gcn_hidden_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(sent_gcn_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        self.sent_attention = nn.Linear(sent_gcn_hidden_dim, sent_att_dim)

        self.sentence_context_vector = nn.Linear(sent_att_dim, 1, bias=False)

    def forward(self, code_tensor, code_lengths, word_adj_tensor, line_adj_tensor):
        packed_sents = code_tensor.reshape(sum(code_lengths), self.emb_dim)

        sents, word_att_weights = self.word_attention(
            packed_sents, code_lengths, word_adj_tensor)

        sents = self.dropout(sents)
        sents = sents.reshape(
            (len(code_lengths),
             int(len(sents) / len(code_lengths)),
             sents.shape[1])
        )  # -> (num_files, num_lines_per_file, word_gcn_hidden_dim)

        line_data_list = []  # 用于存放每个文件的行级图 Data

        # 遍历每个文件的行向量 node 与对应的行级边 edge
        for node, edge in zip(sents, line_adj_tensor):
            # edge = torch.tensor(edge, dtype=torch.long).to(self.device)  # 行级邻接矩阵中的边索引 (2, num_edges)
            if isinstance(edge, torch.Tensor):
                edge = edge.to(dtype=torch.long, device=self.device)
            else:
                edge = torch.as_tensor(edge, dtype=torch.long, device=self.device)

            line_data_list.append(Data(node, edge))

        batch = Batch.from_data_list(line_data_list)

        line_out = self.gcn(batch.x, batch.edge_index)
        if self.use_layer_norm:
            normed_sents = self.layer_norm(line_out)
        else:
            normed_sents = line_out

        # '''
        # 两层GCN
        # '''
        # line_out = F.relu(self.gcn(batch.x, batch.edge_index))
        # line_out = self.layer_norm(line_out) if self.use_layer_norm else line_out
        # line_out = F.dropout(line_out)
        # normed_sents = F.relu(self.gcn1(line_out, batch.edge_index))


        # ----- 行级注意力 -----
        att = torch.tanh(self.sent_attention(normed_sents))  # (total_num_lines, sent_att_dim)
        att = self.sentence_context_vector(att).squeeze(1)   # (total_num_lines,)

        val = att.max()
        att = torch.exp(att - val)

        att = att.reshape(len(code_lengths), int(len(att) / len(code_lengths)))

        sent_att_weights = att / torch.sum(att, dim=1, keepdim=True)  # shape: (num_files, num_lines_per_file)

        # ----- 用注意力加权求和得到文件向量 -----
        code_tensor = line_out.reshape(
            (len(code_lengths),
             int(len(line_out) / len(code_lengths)),
             line_out.shape[1])
        )

        code_tensor = code_tensor * sent_att_weights.unsqueeze(2)

        code_tensor = code_tensor.sum(dim=1)

        word_att_weights = word_att_weights.reshape(
            (len(code_lengths),
             int(len(word_att_weights) / len(code_lengths)),
             word_att_weights.shape[1])
        )

        return code_tensor, word_att_weights, sent_att_weights, sents


class WordAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim,
                 gcn_hidden_dim,
                 att_dim,
                 use_layer_norm, dropout,
                 device):
        super(WordAttention, self).__init__()
        self.device = device
        self.emb_dim = embed_dim

        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        self.gcn = GCNConv(embed_dim, gcn_hidden_dim)
        self.gcn1 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(gcn_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        self.attention = nn.Linear(gcn_hidden_dim, att_dim)
        self.context_vector = nn.Linear(att_dim, 1, bias=False)

    def init_embeddings(self, embeddings):
        self.embeddings.weight = nn.Parameter(embeddings)

    def freeze_embeddings(self, freeze=False):
        self.embeddings.weight.requires_grad = freeze

    def forward(self, sents, code_lenth, adj_tensor):
        sents = self.embeddings(sents)
        adj_tensor = [item for sublist in adj_tensor for item in sublist]

        word_data_list = []

        for node, edge_index in zip(sents, adj_tensor):
            if np.any(edge_index == -1):
                edge_index = np.array([])
            edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
            word_data_list.append(Data(node, edge_index))

        batch = Batch.from_data_list(word_data_list)
        word_out = self.gcn(batch.x, batch.edge_index)
        if self.use_layer_norm:
            normed_words = self.layer_norm(word_out)
        else:
            normed_words = word_out

        att = torch.tanh(self.attention(normed_words))
        att = self.context_vector(att).squeeze(1)
        val = att.max()
        att = torch.exp(att - val)
        att = att.reshape(sum(code_lenth), max_seq_len)
        att_weights = att / torch.sum(att, dim=1, keepdim=True)  # (总行数, max_seq_len)

        word_out = word_out.reshape(
            (sum(code_lenth),
             max_seq_len,
             word_out.shape[1])
        )

        sents = word_out * att_weights.unsqueeze(2)
        sents = sents.sum(dim=1)

        return sents, att_weights


class ResidualBlock(nn.Module):  # 定义残差块：y = x + F(x)
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    def forward(self, x):  # 前向传播：生成逐行logit；训练时计算focal loss
        return x + self.layers(x)  # Skip connection


class LineAttention(nn.Module):
    def __init__(self, hidden_size, line_in_dim, block_in_dim, use_layer_norm, dropout, device):
        super(LineAttention, self).__init__()
        self.hidden_size = hidden_size
        # self.embed_dim = self.args.embed_dim
        self.use_layer_norm = use_layer_norm
        self.dropout = dropout
        self.device = device

        self.q_proj = nn.Linear(line_in_dim, hidden_size, bias=False)
        self.kv_proj = nn.Linear(block_in_dim, hidden_size, bias=False)
        self.q_norm = nn.LayerNorm(hidden_size)
        self.kv_norm = nn.LayerNorm(hidden_size)

        self.line_attention = nn.MultiheadAttention(
            self.hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=self.dropout
        )

        self.line_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            ResidualBlock(  # Optional residual block
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.LayerNorm(self.hidden_size),
                    nn.GELU(),
                    nn.Dropout(self.dropout)
                )
            )
        )
        self.line_head = nn.Linear(self.hidden_size, 1)

    def forward(self, code_line_tensor, blocks_before_gcn, basic_block_size, block_neighbors):
        if isinstance(code_line_tensor, list):
            code_line_tensor = [t.to(self.device, dtype=torch.float32, non_blocking=True) for t in code_line_tensor]
        else:
            code_line_tensor = code_line_tensor.to(self.device, dtype=torch.float32, non_blocking=True)

        line_probs = []

        for i in range(len(code_line_tensor)):
            global_lines = code_line_tensor[i]
            block_matrix = blocks_before_gcn[i]
            block_size = basic_block_size[i]
            neigh_info = block_neighbors[i]

            if not torch.is_tensor(block_size):
                block_size = torch.as_tensor(block_size, device=self.device, dtype=torch.long)
            else:
                block_size = block_size.to(self.device, dtype=torch.long)

            L_i = int(global_lines.size(0))
            device = global_lines.device

            block_matrix = block_matrix.to(device=device, dtype=torch.float32, non_blocking=True)
            if block_matrix.dim() != 2:
                block_matrix = block_matrix.reshape(-1, block_matrix.size(-1))
            B_i = int(block_size.numel())

            if L_i == 0 or B_i == 0:
                line_probs.append(torch.zeros(L_i, device=device));
                continue

            if int(block_matrix.size(0)) < B_i:
                pad_rows = B_i - int(block_matrix.size(0))
                pad = torch.zeros(pad_rows, block_matrix.size(1), device=device, dtype=block_matrix.dtype)
                block_reprs = torch.cat([block_matrix, pad], dim=0)
            elif int(block_matrix.size(0)) > B_i:
                block_reprs = block_matrix[:B_i]
            else:
                block_reprs = block_matrix  # [B_i, block_in_dim]

            # ===== 1) 行 → 所属块索引 blk_idx =====
            repeat_indices = []
            for blk_idx, cnt in enumerate(block_size.tolist()):
                repeat_indices.extend([blk_idx] * int(cnt))
            if len(repeat_indices) < L_i:
                last_blk = repeat_indices[-1] if repeat_indices else 0
                repeat_indices.extend([last_blk] * (L_i - len(repeat_indices)))
            blk_idx = torch.as_tensor(repeat_indices[:L_i], device=device, dtype=torch.long)  # [L_i]

            # ===== 2) 为每个块构造“动态邻域” N(b) = pred(b) + [b] + succ(b) =====
            # neigh_info 期望为长度 B_i 的 list，每个元素是 {'pred': [...], 'succ': [...]}
            # 做一次预处理把所有块的邻域收好，并记录每个邻域的长度（用于 padding）
            neighbors_per_block = []
            max_m = 1
            for b in range(B_i):
                preds = sorted(set(int(x) for x in (neigh_info[b].get('pred', []) if b < len(neigh_info) else []) if
                                   0 <= int(x) < B_i))
                succs = sorted(set(int(x) for x in (neigh_info[b].get('succ', []) if b < len(neigh_info) else []) if
                                   0 <= int(x) < B_i))
                nb = preds + [b] + succs  # 至少包含自身
                if len(nb) == 0:
                    nb = [b]
                neighbors_per_block.append(nb)
                if len(nb) > max_m:
                    max_m = len(nb)

            # ===== 3) 为每一行按其所属块 b 取 N(b)，做统一 padding → (L_i, max_m) =====
            # K/V 索引矩阵（每行一条邻域列表，右侧 pad 为 0）；同时构造 key_padding_mask=True 表示“pad位”
            K_indices = torch.zeros((L_i, max_m), dtype=torch.long, device=device)
            key_padding_mask = torch.ones((L_i, max_m), dtype=torch.bool, device=device)  # 先全 True
            for j in range(L_i):
                nb = neighbors_per_block[int(blk_idx[j])]
                m = len(nb)
                K_indices[j, :m] = torch.as_tensor(nb, dtype=torch.long, device=device)
                key_padding_mask[j, :m] = False  # 真正的邻居不 mask

            # ===== 4) 计算 Q/K/V =====
            # Q: 每一行的向量 → [L_i, H]
            q = self.q_norm(self.q_proj(global_lines))  # [L_i, H]

            # K/V: 先按 K_indices 取块向量，再投影到 H，并做 LayerNorm
            kv_raw = block_reprs.index_select(0, K_indices.reshape(-1))  # [L_i*max_m, block_in_dim]
            kv_raw = kv_raw.view(L_i, max_m, -1)  # [L_i, max_m, block_in_dim]
            kv = self.kv_norm(self.kv_proj(kv_raw))  # [L_i, max_m, H]

            # ===== 5) 多头注意力：行 ↔（所属块 ∪ 全部前驱 ∪ 全部后继）=====
            attn_out, _ = self.line_attention(
                q.unsqueeze(1),  # [L_i, 1, H]
                kv,  # [L_i, max_m, H]
                kv,  # [L_i, max_m, H]
                key_padding_mask=key_padding_mask  # [L_i, max_m]，pad 位置为 True
            )  # → [L_i, 1, H]
            attn_out = attn_out.squeeze(1)  # [L_i, H]

            line_final = torch.cat([q, attn_out], dim=-1)  # [L_i, 2H]
            line_feat = self.line_proj(line_final)  # [L_i, H]
            logits = self.line_head(line_feat).squeeze(-1)  # [L_i]
            line_probs.append(torch.sigmoid(logits))

        return line_probs
