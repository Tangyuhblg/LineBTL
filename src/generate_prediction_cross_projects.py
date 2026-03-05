import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import argparse
from gensim.models import Word2Vec
from tqdm import tqdm

from LineBB_model import *
from my_util import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0) 

# all_eval_rels_cross_projects = {
#     'activemq': ['camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
#     'camel': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
#     'derby': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
#     'groovy': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'hbase-0.95.2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
#     'hbase': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
#     'hive': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
#     'jruby': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'hive-0.12.0', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
#     'lucene': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'wicket-1.5.3'],
#     'wicket': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1']
#     }


all_eval_rels_cross_projects = {
    'activemq': ['camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
    'camel': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
    'derby': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
    'groovy': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'hbase-0.95.2', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
    'hbase': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
    'hive': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
    'jruby': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'],
    'lucene': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'wicket-1.5.3'],
    'wicket': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1']
    }

arg = argparse.ArgumentParser()

arg.add_argument('-dataset',type=str, default='wicket', help='software project name (lowercase)')
arg.add_argument('-embed_dim', type=int, default=150, help='word embedding size')
arg.add_argument('-word_gcn_hidden_dim', type=int, default=128, help='word attention hidden size')
arg.add_argument('-sent_gcn_hidden_dim', type=int, default=128, help='sentence attention hidden size')
arg.add_argument('-exp_name',type=str,default='')
arg.add_argument('-target_epochs',type=str,default='5')
arg.add_argument('-dropout', type=float, default=0.2, help='dropout rate')
arg.add_argument('-weighted_graph', type=bool, default=False, help='Whether to use weighted graph')

args = arg.parse_args()

weight_dict = {}

# model setting
max_grad_norm = 5
embed_dim = args.embed_dim
word_gcn_hidden_dim = args.word_gcn_hidden_dim
sent_gcn_hidden_dim = args.sent_gcn_hidden_dim
word_att_dim = 64
sent_att_dim = 64
use_layer_norm = True
dropout = args.dropout
weighted_graph = args.weighted_graph
save_every_epochs = 5
exp_name = args.exp_name

line_hidden_dim = 128
line_att_dim = 64


save_model_dir = '../output/model/LineBB/'
intermediate_output_dir = '../output/intermediate_output/LineBB/cross-project/'
prediction_dir = '../output/prediction/LineBB/cross-project/'
file_lvl_gt = '../datasets/basic_block_data/'

if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)


def build_blocks(df):
    """
    根据 block_id 聚合为基本块文本，并返回：
      block_texts: List[str]    每个基本块的拼接文本（用于 prepare_code2d）
      block_order: np.ndarray   基本块 id 的出现顺序
      basic_sizes: List[int]    每个基本块包含的行数（用于行→块映射）
    """
    if 'block_id' not in df.columns:
        # 退化为“1个基本块=整文件”
        code_lines = df['code_line'].tolist() if 'code_line' in df.columns else df['code'].tolist()
        return [" ".join(str(x) for x in code_lines)], np.array([0]), [len(code_lines)]

    block_order = pd.unique(df['block_id'])
    block_texts, basic_sizes = [], []
    for bid in block_order:
        lines_in_blk = df.loc[df['block_id'] == bid, 'code_line'].tolist() \
            if 'code_line' in df.columns else df.loc[df['block_id'] == bid, 'code'].tolist()
        block_texts.append(" ".join(str(x) for x in lines_in_blk))
        basic_sizes.append(len(lines_in_blk))
    return block_texts, block_order, basic_sizes


def parse_id_list(s: str):
    """把形如 '3|4|7' 的字符串解析成 int 列表；空/异常返回 []。"""
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    out = []
    for t in s.split('|'):
        t = t.strip()
        if not t:
            continue
        try:
            out.append(int(t))
        except Exception:
            continue
    return out


def build_block_neighbors(df: pd.DataFrame, block_order: list[int]) -> list[dict]:
    """
    依据 CSV 中 pred_blocks / succ_blocks 构造“邻接表”：
        返回长度 = num_blocks 的 list，其中每个元素是 {'pred': [pos...], 'succ': [pos...]}
        这里 pos 是该块在当前文件“块顺序”中的 0..N-1 位置（而非原始 block_id）
    """
    # 为了稳妥，先按 new_line_number 排序，保证 mode()/first() 的一致性
    if 'new_line_number' in df.columns:
        df = df.sort_values('new_line_number')

    # 原始ID -> 位置索引（0..N-1）
    bid_to_pos = {orig_id: idx for idx, orig_id in enumerate(block_order)}

    # 每个块挑选一个代表行上的 pred/succ 字符串（用众数/第一条）
    raw_pred_map, raw_succ_map = {}, {}
    for bid, block_df in df.groupby('block_id', sort=True):
        pred_series = block_df['pred_blocks'].dropna().astype(str)
        succ_series = block_df['succ_blocks'].dropna().astype(str)
        pred_str = pred_series.mode().iloc[0] if not pred_series.empty else ''
        succ_str = succ_series.mode().iloc[0] if not succ_series.empty else ''
        raw_pred_map[int(bid)] = pred_str
        raw_succ_map[int(bid)] = succ_str

    neighbors = []
    for bid in block_order:
        preds_raw = parse_id_list(raw_pred_map.get(bid, ''))
        succs_raw = parse_id_list(raw_succ_map.get(bid, ''))
        preds = [bid_to_pos[x] for x in preds_raw if x in bid_to_pos]
        succs = [bid_to_pos[x] for x in succs_raw if x in bid_to_pos]
        neighbors.append({'pred': preds, 'succ': succs})
    return neighbors


def predict_defective_files_in_releases(dataset_name, target_epochs):

    actual_save_model_dir = save_model_dir+dataset_name+'/'
    actual_prediction_dir = prediction_dir+dataset_name+'/'

    if not os.path.exists(actual_prediction_dir):
        os.makedirs(actual_prediction_dir)

    train_rel = all_train_releases[dataset_name]
    test_rel = all_eval_rels_cross_projects[dataset_name] 

    w2v_dir = get_w2v_path()

    word2vec_file_dir = os.path.join(w2v_dir,dataset_name+'-'+str(embed_dim)+'dim.bin')

    word2vec = Word2Vec.load(word2vec_file_dir)
    print('load Word2Vec for',dataset_name,'finished')

    total_vocab = len(word2vec.wv)

    vocab_size = total_vocab + 1  # for unknown tokens
        
    model = Model(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        word_gcn_hidden_dim=word_gcn_hidden_dim,
        sent_gcn_hidden_dim=sent_gcn_hidden_dim,
        word_att_dim=word_att_dim,
        sent_att_dim=sent_att_dim,
        line_hidden_dim=line_hidden_dim,
        use_layer_norm=use_layer_norm,
        dropout=dropout,
        device=device)

    if exp_name == '':
        checkpoint = torch.load(actual_save_model_dir+'checkpoint_'+target_epochs+'epochs.pth', map_location=device)

    else:
        checkpoint = torch.load(actual_save_model_dir+exp_name+'/checkpoint_'+exp_name+'_'+target_epochs+'epochs.pth', map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    model.sent_attention.word_attention.freeze_embeddings(True)

    model = model.to(device)
    model.eval()

    for rel in test_rel:
        print('using model from {} to generate prediction of {}'.format(train_rel,rel))
        
        actual_intermediate_output_dir = intermediate_output_dir+dataset_name+'/'+train_rel+'-'+rel+'/'

        if not os.path.exists(actual_intermediate_output_dir):
            os.makedirs(actual_intermediate_output_dir)

        test_df = get_df(rel)
    
        row_list = []  # for creating dataframe later...

        for filename, df in tqdm(test_df.groupby('filename')):

            file_label = bool(df['file-label'].unique())
            line_label = df['line-label'].tolist()
            line_number = df['new_line_number'].tolist()
            is_comments = df['is_comment'].tolist()
            line_lenth = len(df)
            code_lines = df['code_line'].tolist()
            block_id = df['block_id'].tolist()

            # 基本块构造（文本/顺序/大小）
            block_texts, block_order, basic_sizes = build_blocks(df)

            # pred/succ 构造邻接
            block_neighbors = build_block_neighbors(df, block_order)

            # 待修改
            code2d, word_edge, line_edge = prepare_code2d(block_texts, len(block_texts), True, weighted_graph, block_neighbors)
            code2d_line = prepare_code2d_line(code_lines, True)

            basic_block_size = [torch.as_tensor(basic_sizes, dtype=torch.long, device=device)]
            code3d, code3d_line = [code2d], [code2d_line]

            codevec = get_x_vec(code3d, word2vec)
            codevec_line = get_x_vec(code3d_line, word2vec)
            codevec_padded = pad_code(codevec, len(block_texts))
            codevec_line_padded = pad_code(codevec_line, len(code_lines))
            codevec_padded_tensor = torch.tensor(codevec_padded)
            codevec_line_padded_tensor = torch.tensor(codevec_line_padded)

            def edge_to_tensor(edge):
                return torch.as_tensor(edge, dtype=torch.long, device=device) if isinstance(edge, np.ndarray) else edge

            word_edge_list = [word_edge]
            line_edge_list = [edge_to_tensor(line_edge)]
            block_neighbors_list = [block_neighbors]

            with torch.no_grad():
                codevec_padded_tensor = torch.tensor(codevec)
                output, line_att_weights,  block_att_weights, _ = model(codevec_padded_tensor, codevec_line_padded_tensor, word_edge_list,
                                                                     line_edge_list, basic_block_size, block_neighbors_list)
                file_prob = output.item()
                prediction = bool(round(output.item()))

                line_scores = line_att_weights[0].detach().cpu().numpy()
                block_scores = block_att_weights[0].detach().cpu().numpy()

                if 'block_id' in df.columns:
                    bid = df['block_id'].to_numpy().astype(int)
                    if bid.min() >= 1:
                        bid = bid - 1
                    bid = np.clip(bid, 0, len(block_scores) - 1)
                    block_scores = block_scores[bid]  # [num_lines]
                else:
                    block_scores = np.full(len(code_lines),
                                           float(block_scores.mean()) if len(block_scores) > 0 else 0.0,
                                           dtype=float)
                torch.cuda.empty_cache()

                for i in range(0,len(code_lines)):
                    cur_line = code_lines[i]
                    cur_line_label = line_label[i]
                    cur_line_number = line_number[i]
                    cur_is_comment = is_comments[i]
                    cur_line_attn = line_scores[i]
                    cur_block_attn = block_scores[i]
                    cur_block_id = block_id[i]

                    # token_list = cur_line.strip().split()
                    #
                    # max_len = min(len(token_list), 50) # limit max token each line

                    row_dict = {
                        'project': dataset_name,
                        'train': train_rel,
                        'test': rel,
                        'filename': filename,
                        'file-level-ground-truth': file_label,
                        'prediction-prob': file_prob,
                        'prediction-label': prediction,
                        'line-number': cur_line_number,
                        'block_id': cur_block_id,
                        'code-line': cur_line,
                        'is-comment-line': cur_is_comment,
                        # 'token': tok,
                        # 'token-attention-score': word_attn,
                        'line-level-ground-truth': cur_line_label,
                        'line-attention-score': cur_line_attn,
                        'block-attention-score': cur_block_attn
                        }

                    row_list.append(row_dict)

        df = pd.DataFrame(row_list)

        df.to_csv(actual_prediction_dir+train_rel+'-'+rel+'.csv', index=False)

        print('finished release', rel)


dataset_name = args.dataset
target_epochs = args.target_epochs

predict_defective_files_in_releases(dataset_name, target_epochs)