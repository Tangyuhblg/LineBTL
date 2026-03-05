import re
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os


os.environ["TORCH_USE_CUDA_DSA"] = "1"

max_seq_len = 150


all_train_releases = {'activemq': 'activemq-5.0.0',
                      'camel': 'camel-1.4.0',
                      'derby': 'derby-10.2.1.6',
                      'groovy': 'groovy-1_5_7',
                      'hbase': 'hbase-0.94.0',
                      'hive': 'hive-0.9.0',
                      'jruby': 'jruby-1.1',
                      'lucene': 'lucene-2.3.0',
                      'wicket': 'wicket-1.3.0-incubating-beta-1'}

all_eval_releases = {'activemq': ['activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'],
                     'camel': ['camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'],
                     'derby': ['derby-10.3.1.4', 'derby-10.5.1.1'],
                     'groovy': ['groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'],
                     'hbase': ['hbase-0.95.0', 'hbase-0.95.2'],
                     'hive': ['hive-0.10.0', 'hive-0.12.0'],
                     'jruby': ['jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'],
                     'lucene': ['lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'],
                     'wicket': ['wicket-1.3.0-beta2', 'wicket-1.5.3']}

all_releases = {'activemq': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'],
                'camel': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'],
                'derby': ['derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1'],
                'groovy': ['groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'],
                'hbase': ['hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2'],
                'hive': ['hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0'],
                'jruby': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'],
                'lucene': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'],
                'wicket': ['wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3']}


all_projs = list(all_train_releases.keys())

file_lvl_gt = '../datasets/basic_block_data/'

word2vec_dir = '../output/Word2Vec_model/'


# ====================================================================
# 1. mini-batch 生成器：打乱文件顺序，按 batch_size 切分
#    对应论文中“使用 mini-batch 训练深度模型”的实现细节。
# ====================================================================
def batch_generator(code, code_line, label, word_edge, line_edge, basic_block_size, block_neighbors, batch_size, random_seed=0):
    """
    code:      代码 3D/2D 结构（一般为 [num_files, num_lines, max_seq_len] 的 token index）
    label:     文件级标签（缺陷文件=1 / 非缺陷文件=0），对应论文的 file-level ground truth
    word_edge: 每个文件中，每行对应的词级边列表（token-level graph）
    line_edge: 每个文件的行级边（line-level graph）
    batch_size: mini-batch 大小
    random_seed: 随机种子，用于可复现的打乱顺序（shuffle）
    """
    total_samples = len(code)          # 文件总数 = 样本数
    num_batches = total_samples // batch_size  # 能组成的完整 batch 数（余数丢弃）

    if random_seed is not None:
        np.random.seed(random_seed)    # 设定 numpy 随机种子，保证划分可复现

    indices = np.arange(total_samples) # 生成 [0, 1, ..., N-1] 的索引数组
    np.random.shuffle(indices)         # 打乱文件顺序（对应随机采样 mini-batch）

    # 逐 batch 生成数据
    for i in range(num_batches):
        start_idx = i * batch_size     # 当前 batch 起始下标
        end_idx = min((i + 1) * batch_size, total_samples)  # 当前 batch 结束下标

        batch_indices = indices[start_idx:end_idx]  # 当前 batch 包含的文件索引
        batch_code = code[batch_indices]            # 挑出对应文件的 token index
        batch_label = label[batch_indices]          # 挑出对应文件的标签
        batch_code_line = code_line[batch_indices]

        # word_edge / line_edge 是 list 结构，只能逐个取
        batch_word_edge = [word_edge[idx] for idx in batch_indices]
        batch_line_edge = [line_edge[idx] for idx in batch_indices]

        batch_basic_block_size = [basic_block_size[idx] for idx in batch_indices]
        batch_basic_block_size = [torch.as_tensor(x, dtype=torch.long) for x in batch_basic_block_size]

        batch_block_neighbors = [block_neighbors[idx] for idx in batch_indices]

        # 使用 yield 实现生成器，每次返回一个 batch 的n元组
        yield batch_code, batch_code_line, batch_label, batch_word_edge, batch_line_edge, batch_basic_block_size, batch_block_neighbors


# ====================================================================
# 2. 词级边的 padding：保证每个文件都有同样行数的“行内 token 图”结构
#    对应论文中“对齐行数以便 batch 训练 GCN”。
# ====================================================================
def pad_word_edge_index(line_egde_index_list, max_sent_len, limit_sent_len=True):
    """
    line_egde_index_list: 对于一个文件，包含该文件每行的 token-level edge_index 列表。
                          结构：list[ list[np.ndarray(2, E_line)] ]
    max_sent_len: 当前 batch 内，所有文件中最大的行数，用来对齐行数。
    limit_sent_len: 是否把行数限制在 max_sent_len（此处 True/False 逻辑相同）。
    返回：
        padding 后的 edge_index 列表：
        - 如果某文件行数 < max_sent_len，则补若干“空行二部图”（边为 -1，占位）。
    """
    padded_edge_list = []              # 存放 padding 后的每个文件的 edge_index
    for edge in line_egde_index_list:  # 遍历每个文件的“行边集合”
        num = max_sent_len - len(edge) # 需要补多少个“占位行”
        if max_sent_len - len(edge) > 0:
            for _ in range(num):
                # 每补一个“空行”，就追加一个 (2,1) 的 [-1] 数组，
                # 在后续构建 PyG 图时，遇到 -1 可以视为“无效边”并跳过。
                edge.append(np.full((2, 1), -1))

        # 是否截断在 max_sent_len（此处两个 if 分支等价）
        if limit_sent_len:
            padded_edge_list.append(edge)
        else:
            padded_edge_list.append(edge)

    return padded_edge_list


# ====================================================================
# 3. 行级 edge_index 的裁剪：删除指向“超过最大行数”的行边
#    对应论文 line-level graph 的构建后，针对 batch 对齐进行裁剪。
# ====================================================================
def pad_line_edge_index(line_egde_index_list, max_sent_len, limit_sent_len=True):
    """
    line_egde_index_list: 每个文件的行级 edge_index（np.array(2, E)）
    max_sent_len: 本 batch 中统一保留的最大行数（行 index >= max_sent_len 的视为不存在）
    limit_sent_len: 若 True，则删除连接到这些不存在行的边。
    """
    padded_line_edge = []
    if limit_sent_len:
        for edge_index in line_egde_index_list:
            # idx: 布尔向量，标记 edge_index 中“某一列是否存在 > max_sent_len-1 的节点编号”
            idx = np.any(edge_index > max_sent_len - 1, axis=0)
            # 找出这些“非法列”的下标
            column_numbers = np.where(idx)[0]
            # 删除对应列（即删掉指向不存在行的边）
            edge_index = np.delete(edge_index, column_numbers, axis=1)
            padded_line_edge.append(edge_index)
    return padded_line_edge


# ====================================================================
# 4. 读取某个版本的行级 CSV，做简单清洗
#    对应论文的数据预处理部分：过滤空行和测试文件。
# ====================================================================
def get_df(rel, is_baseline=False):
    """
    rel:        版本名，如 'camel-1.4.0'
    is_baseline: 若 True，则在路径前面多加一层 '../'，用于兼容 baseline 代码的目录结构。
    返回：
        df: DataFrame，包含该版本所有源码文件的非空、非测试行及其标签信息。
    """
    if is_baseline:
        df = pd.read_csv('../' + file_lvl_gt + rel + ".csv")
    else:
        df = pd.read_csv(file_lvl_gt + rel + ".csv")

    # 将缺失值填成空字符串，避免后续字符串操作时报错
    df = df.fillna('')

    # 删除标记为“空行”的记录，对应论文中不考虑空白行
    df = df[df['is_blank'] == False]
    # 删除测试文件的行（is_test_file == True），
    # 论文中通常只关注非测试代码的缺陷预测。
    df = df[df['is_test_file'] == False]

    return df


# ====================================================================
# 5. 构造行级图的 edge_index：对应论文中的 line-level graph（行之间的窗口共现图）
# ====================================================================
def prepare_line_adj(line_lenth, weighted_graph=False, block_neighbors=None):
    """
    line_lenth: 基本块数（节点个数）
    weighted_graph: 是否叠加权重（窗口共现次数）
    block_neighbors: None 或长度为 line_lenth 的列表，
        每个元素是 {'pred': [前驱下标...], 'succ': [后继下标...]}（下标均是 0..line_lenth-1）
    返回：edge_index: np.array(2, E)
    """
    windows = []

    if block_neighbors is None:
        # 兼容旧逻辑：固定滑动窗口（不建议在基本块模式下使用）
        window_size = 4
        idx = range(line_lenth)
        if line_lenth <= window_size:
            windows.append(idx)
        else:
            for j in range(line_lenth - window_size + 1):
                windows.append(idx[j: j + window_size])
    else:
        assert len(block_neighbors) == line_lenth, \
            f"block_neighbors length {len(block_neighbors)} != {line_lenth}"
        for i in range(line_lenth):
            preds = sorted(set(int(x) for x in block_neighbors[i].get('pred', []) if 0 <= int(x) < line_lenth))
            succs = sorted(set(int(x) for x in block_neighbors[i].get('succ', []) if 0 <= int(x) < line_lenth))
            # 按顺序拼出动态窗口：pred 升序 → i → succ 升序
            win = preds + [i] + succs
            # 只有自己一个节点则无需成边；否则将窗口加入做“完全子图”
            if len(win) > 1:
                windows.append(win)

    # 统计窗口内两两共现，构建双向边
    pair_count = {}
    for window in windows:
        window = list(window)
        for p in range(1, len(window)):
            for q in range(0, p):
                u, v = window[p], window[q]
                pair_count[(u, v)] = pair_count.get((u, v), 0.0) + 1.0
                pair_count[(v, u)] = pair_count.get((v, u), 0.0) + 1.0

    if not pair_count:
        return np.zeros((2, 0), dtype=int)

    row, col, weight = [], [], []
    for (u, v), w in pair_count.items():
        row.append(u); col.append(v)
        weight.append(w if weighted_graph else 1.0)

    adj = sp.csr_matrix((weight, (row, col)), shape=(line_lenth, line_lenth))
    edge_index = np.array(adj.nonzero())
    return edge_index



# ====================================================================
# 6. 构造行内 token 图 + token 序列（二维）+ 行级图
#    对应论文中的 token-level graph 构建 + 行级 graph 构建。
# ====================================================================
def prepare_code2d(code_list, line_lenth, to_lowercase=False, weighted_graph=False, block_neighbors=None):
    """
    code_list:   每个基本块拼接后的文本（已按块顺序）
    line_lenth:  基本块数
    block_neighbors: List[{'pred': [...], 'succ': [...]}] 或 None
    """
    code2d = []
    all_word_edge_index = []

    for c in code_list:
        windows = []
        c = re.sub('\\s+', ' ', c)
        if to_lowercase:
            c = c.lower()
        token_list = c.strip().split()
        total_tokens = len(token_list)
        if total_tokens > max_seq_len:
            token_list = token_list[:max_seq_len]
            total_tokens = max_seq_len
        if total_tokens < max_seq_len:
            token_list = token_list + ['<pad>'] * (max_seq_len - total_tokens)
        code2d.append(token_list)


        idx = range(0, total_tokens)
        window_size = 2
        if total_tokens <= window_size:
            windows.append(idx)
        else:
            for j in range(total_tokens - window_size + 1):
                windows.append(idx[j: j + window_size])
        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    wp, wq = window[p], window[q]
                    word_pair_count[(wp, wq)] = word_pair_count.get((wp, wq), 0.0) + 1.0
                    word_pair_count[(wq, wp)] = word_pair_count.get((wq, wp), 0.0) + 1.0
        row, col, weight = [], [], []
        for (p, q), w in word_pair_count.items():
            row.append(p); col.append(q); weight.append(w if weighted_graph else 1.0)
        adj = sp.csr_matrix((weight, (row, col)), shape=(max_seq_len, max_seq_len))
        all_word_edge_index.append(np.array(adj.nonzero()))

    # ——关键：把 block_neighbors 传给新的 prepare_line_adj——
    all_line_edge_index = prepare_line_adj(
        line_lenth=line_lenth,
        weighted_graph=weighted_graph,
        block_neighbors=block_neighbors
    )
    return code2d, all_word_edge_index, all_line_edge_index



# ====================================================================
# 7. 从 DataFrame 生成 3D 代码结构 + 文件标签 + 图结构
#    这是整个数据预处理 pipeline 的核心出口，对应论文中“文件级 & 行级输入构建”。
# ====================================================================
def get_code3d_and_label(df, to_lowercase=False, weighted_graph=False, window_size=None):
    """
    现在把第二维从“行”换成“基本块”，并使用 CSV 中的 pred_blocks / succ_blocks
    来构造 basic-block-level 图。
    """
    if window_size is None:
        window_size = 4

    code3d, all_file_label = [], []
    word_edge, basic_block_edge, basic_block_size, block_neighbors_list = [], [], [], []

    for filename, group_df in df.groupby('filename'):
        if 'new_line_number' in group_df.columns:
            group_df = group_df.sort_values('new_line_number')

        file_label = bool(group_df['file-label'].iloc[0])

        # ——1) 确定块的输出顺序 + 收集每块的文本、尺寸、raw pred/succ——
        block_texts, block_sizes, block_order = [], [], []
        raw_pred_map, raw_succ_map = {}, {}

        # 使用排序后的 block_id 顺序来构造二维矩阵的“行序”
        for bid, block_df in group_df.groupby('block_id', sort=True):
            if 'new_line_number' in block_df.columns:
                block_df = block_df.sort_values('new_line_number')

            block_texts.append(' '.join(str(s) for s in block_df['code_line'].tolist()))
            block_sizes.append(len(block_df))
            block_order.append(int(bid))

            # 取该块最常见（或第一条） pred/succ 字符串
            pred_str = block_df['pred_blocks'].dropna().astype(str)
            succ_str = block_df['succ_blocks'].dropna().astype(str)
            pred_str = pred_str.mode().iloc[0] if not pred_str.empty else ''
            succ_str = succ_str.mode().iloc[0] if not succ_str.empty else ''
            raw_pred_map[int(bid)] = pred_str
            raw_succ_map[int(bid)] = succ_str

        num_blocks = len(block_texts)
        # 原始 block_id → 在 code2d 中的位置（0..num_blocks-1）
        bid_to_pos = {bid: idx for idx, bid in enumerate(block_order)}

        # ——2) 解析 pred/succ 字符串，映射到 0..num_blocks-1 的邻居下标——
        def _parse_list(s):
            if s is None:
                return []
            s = str(s).strip()
            if s == '':
                return []
            out = []
            for t in s.split('|'):
                t = t.strip()
                if t == '':
                    continue
                # 旧 CSV 里有可能是整数，也可能带空格
                try:
                    out.append(int(t))
                except:
                    continue
            return out

        block_neighbors = []
        for bid in block_order:
            preds_raw = _parse_list(raw_pred_map.get(bid, ''))
            succs_raw = _parse_list(raw_succ_map.get(bid, ''))
            # 映射原始 block_id -> 位置下标；过滤越界/缺失
            preds = [bid_to_pos[x] for x in preds_raw if x in bid_to_pos]
            succs = [bid_to_pos[x] for x in succs_raw if x in bid_to_pos]
            block_neighbors.append({'pred': preds, 'succ': succs})

        # ——3) 构造二维 token（每块一行）+ 基本块图（CFG 动态窗口）——
        code2d, word_adj, basic_block_adj = prepare_code2d(
            code_list=block_texts,
            line_lenth=num_blocks,
            to_lowercase=to_lowercase,
            weighted_graph=weighted_graph,
            block_neighbors=block_neighbors
        )

        code3d.append(code2d)
        all_file_label.append(file_label)
        word_edge.append(word_adj)
        basic_block_edge.append(basic_block_adj)
        basic_block_size.append(block_sizes)
        block_neighbors_list.append(block_neighbors)

    return code3d, all_file_label, word_edge, basic_block_edge, basic_block_size, block_neighbors_list


'''
    行级注意力机制
'''
def prepare_code2d_line(code_list, to_lowercase=False):
    """
    将一份文件的“行文本列表” → “二维 token 列表”
    输入: code_list = ['public class A {', 'int x=1;', ...]
    输出: code2d = [['public','class','A','{'], ['int','x','=', '1',';'], ...]
    """
    code2d = []

    for c in code_list:
        c = re.sub('\\s+', ' ', c)  # 把多空白折叠为一个空格（等价 r'\s+'）

        if to_lowercase:  # 可选：小写化（注意：对代码大小写敏感语言可能有副作用）
            c = c.lower()

        token_list = c.strip().split()
        total_tokens = len(token_list)

        # 关键：保证每行 token 长度恒为 max_seq_len
        if total_tokens > max_seq_len:
            token_list = token_list[:max_seq_len]
        elif total_tokens < max_seq_len:
            token_list = token_list + ['<pad>'] * (max_seq_len - total_tokens)

        code2d.append(token_list) # [[代码行1], [代码行2], ...]

    return code2d


def get_code3d_line_and_label(df, to_lowercase=False, max_sent_length=None):
    """
    将 DataFrame 转成 (code3d, labels)：
    - code3d: List[File]；每个 File 是 List[LineTokens]；每个 LineTokens 是 List[str]
    - labels: List[int/bool]；文件级标签（是否缺陷文件）
    """
    code3d = []
    all_file_label = []
    all_line_label = []

    # 以文件为单位分组（确保同一文件的行聚在一起）
    for filename, group_df in df.groupby('filename'):
        group_df = group_df[group_df['code_line'].ne('')]

        # 该文件的文件级标签（所有行应一致）
        file_label = bool(group_df['file-label'].iloc[0])
        line_label = bool(group_df['line-label'].iloc[0])

        code = list(group_df['code_line'])  # 一个.java文件里的所有代码行

        # 使用 prepare_code2d_line 来确保正确的数据结构
        code2d = prepare_code2d_line(code, to_lowercase)

        # 如果指定了最大行数，进行截断
        if max_sent_length and len(code2d) > max_sent_length:
            code2d = code2d[:max_sent_length]

        code3d.append(code2d)
        all_file_label.append(file_label)
        all_line_label.append(line_label)

    return code3d, all_file_label, all_line_label


# ====================================================================
# 8. 返回 Word2Vec 模型路径
# ====================================================================
def get_w2v_path():
    """
    简单的路径封装，方便主程序在不同地方调用，
    与论文中“预训练词向量”部分对应。
    """
    return word2vec_dir


# ====================================================================
# 9. 从 gensim Word2Vec 模型拿出权重，用于初始化 PyTorch Embedding
#    对应论文中“利用预训练词向量初始化嵌入层”的实现。
# ====================================================================
import numpy as np
import torch


def get_w2v_weight_for_deep_learning_models(word2vec_model, embed_dim):
    """
    word2vec_model: gensim 训练好的 Word2Vec 模型（项目级或跨项目训练）
    embed_dim:      词向量维度 D，应与 Word2Vec 训练时设置一致，并与模型中的 embed_dim 相匹配
    返回：
        w2v_weight: torch.Tensor, 形状为 (V+1, D)：
            - 前 V 行为词表中每个 token 的词向量（与 gensim 模型一致）
            - 最后一行为全 0 向量，用作 OOV / PAD 的 embedding。
    """
    wv = word2vec_model.wv      # KeyedVectors 对象
    vecs = wv.vectors           # numpy 数组，shape=(V, D)，V=词表大小，D=embed_dim

    # 确保预训练词向量的维度与模型期望的 embed_dim 一致
    if vecs.shape[1] != embed_dim:
        raise ValueError(f"Embedding dim mismatch: w2v={vecs.shape[1]} vs expected={embed_dim}")

    # 将 numpy 向量转为 torch 张量，并放在 GPU（cuda）上
    base = torch.tensor(vecs, dtype=torch.float32, device='cuda')  # (V, D)

    # 在最后追加一行全 0 作为 OOV / PAD token 的向量
    oov = torch.zeros(1, embed_dim, dtype=torch.float32, device='cuda')  # (1, D)

    # 拼接得到最终的 embedding 矩阵 (V+1, D)
    return torch.cat([base, oov], dim=0)


# ====================================================================
# 10. 对 code3d 做“行数 padding”：保证每个文件的行数对齐
#     对应论文中为了 batch 训练，需要统一每个文件的最大行数。
# ====================================================================
def pad_code(code_list_3d, max_sent_len, limit_sent_len=True, mode='train'):
    """
    code_list_3d: 三维列表 [num_files, num_lines, max_seq_len] 的 token 字符串/索引
    max_sent_len: 每个文件最多保留/对齐到多少行（句子数）
    limit_sent_len: 若 True，超过 max_sent_len 的行会被截断丢弃
    mode:
        - 'train'：若行数不足 max_sent_len，则在末尾补若干“全0行”；
        - 其他（如 'test'）：视需求可以不强制补齐，仅截断。
    返回：
        paded: 经过行数对齐后的 3D 列表
    """
    paded = []

    # 遍历每个文件
    for file in code_list_3d:
        sent_list = []
        # 遍历该文件的每一行
        for line in file:
            new_line = line
            # 若该行长度 > max_seq_len，则截断到 max_seq_len（多余 token 丢弃）
            # 虽然在 prepare_code2d 中已经保证了 <= max_seq_len，这里是再保险。
            if len(line) > max_seq_len:
                new_line = line[:max_seq_len]
            sent_list.append(new_line)

        if mode == 'train':
            # 如果当前文件行数不足 max_sent_len，则在末尾补若干“全0行”。
            # 此时假定 line 已经是 integer index（0 通常对应 PAD/OOV）。
            if max_sent_len - len(file) > 0:
                for i in range(0, max_sent_len - len(file)):
                    sent_list.append([0] * max_seq_len)

        if limit_sent_len:
            # 只保留前 max_sent_len 行，超出的行丢弃
            paded.append(sent_list[:max_sent_len])
        else:
            paded.append(sent_list)

    return paded


# ====================================================================
# 11. 将 token 字符串转换为 Word2Vec 索引（+1 预留 OOV）
#     这一步对应论文中“将代码文本映射为离散 token id，再查词向量”的过程。
# ====================================================================
def get_x_vec(code_3d, word2vec):
    """
    code_3d: [num_files, num_lines, max_seq_len] 的 token 字符串列表
    word2vec: gensim Word2Vec 模型
    返回：
        与 code_3d 结构相同的 3D 列表，每个 token 从字符串替换为整数索引：
        - 若在 vocab 中，则使用 vocab[token] ∈ [0, V-1]
        - 若不在 vocab 中，则使用 OOV_IDX = V（对应 embedding 矩阵的最后一行全0向量）
    """
    kv = word2vec.wv
    vocab = kv.key_to_index      # 字典：token -> idx(0..V-1)
    OOV_IDX = len(kv)            # V，正好对应 embedding 矩阵中最后一行 OOV/PAD 向量

    # 三重列表推导：
    # texts: 一个文件中的所有行
    # text:  某一行的 token 字符串列表
    # tok:   某个 token 字符串
    return [[[(vocab[tok] if tok in vocab else OOV_IDX) for tok in text]
             for text in texts] for texts in code_3d]
