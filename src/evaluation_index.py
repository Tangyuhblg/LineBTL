import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def file_metrics(df):

    cols = ['train','test','filename','file-level-ground-truth','prediction-prob','prediction-label']
    file_df = df[cols].drop_duplicates()


    y_true = file_df['file-level-ground-truth'].astype(int).values  # True->1, False->0
    y_prob = file_df['prediction-prob'].astype(float).values
    y_pred = file_df['prediction-label'].astype(int).values

    auc = roc_auc_score(y_true, y_prob)

    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    # print(tn, fp, fn, tp)
    tpr = tp/(tp+fn) if (tp+fn)>0 else 0.0
    tnr = tn/(tn+fp) if (tn+fp)>0 else 0.0
    ba  = (tpr + tnr)/2

    return {"AUC": round(auc, 3), "MCC": round(mcc, 3), "BA": round(ba, 3)}

import numpy as np
import pandas as pd

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'is-comment-line' in df.columns:
        df['is-comment-line'] = df['is-comment-line'].astype(bool)
        mask = df['is-comment-line'] == True

        block_attn_cols = ['block-attention-score', 'block-attention', 'line-block-attention-score']
        for col in block_attn_cols:
            if col in df.columns:
                df.loc[mask, col] = 0.0

        line_attn_cols = ['line-attention-score', 'line-attention']
        for col in line_attn_cols:
            if col in df.columns:
                df.loc[mask, col] = 0.0

    if 'line-attention-score' in df.columns:
        df = df.rename(columns={'line-attention-score': 'line_attn'})
    elif 'line-attention' in df.columns:
        df = df.rename(columns={'line-attention': 'line_attn'})
    else:
        raise ValueError("缺少列：line-attention-score / line-attention")

    if 'block-attention-score' in df.columns:
        df = df.rename(columns={'block-attention-score': 'block_attn'})
    elif 'block-attention' in df.columns:
        df = df.rename(columns={'block-attention': 'block_attn'})
    else:
        raise ValueError(
            "缺少块注意力列：block-attention-score / block-attention / line-block-attention-score / attention-score")

    need = ['test', 'filename', 'line-level-ground-truth']
    for c in need:
        if c not in df.columns:
            raise ValueError(f"缺少列：{c}")
    return df


def _sort_and_rank_blockwise(df: pd.DataFrame, top_ratio: float | None = None) -> pd.DataFrame:
    df = df.copy()
    df['block_attn'] = pd.to_numeric(df['block_attn'], errors='coerce').fillna(-np.inf)
    df['line_attn']  = pd.to_numeric(df['line_attn'],  errors='coerce').fillna(-np.inf)

    if 'block_id' not in df.columns:
        df['block_id'] = df.groupby(['test','filename','block_attn']).ngroup()

    df = df.sort_values(['test','filename','block_attn','block_id','line_attn'],
                        ascending=[True, True, False, True, False]).copy()

    df['order'] = df.groupby(['test','filename']).cumcount() + 1
    df['n_lines'] = df.groupby(['test','filename'])['order'].transform('max')

    # 块内顺序与块大小
    df['block_local_order'] = df.groupby(['test','filename','block_id']).cumcount() + 1
    df['block_size'] = df.groupby(['test','filename','block_id'])['block_local_order'].transform('max')

    if top_ratio is not None:
        df['topk_per_block'] = df['block_local_order'] <= np.ceil(top_ratio * df['block_size'])
    return df


def line_metrics(df: pd.DataFrame, filter_mode='gt_only',
                 per_block_top_ratio: float | None = None,
                 require_pred_pos: bool = False):
    d = _normalize_columns(df)
    d = _sort_and_rank_blockwise(d, top_ratio=per_block_top_ratio)

    if 'file-level-ground-truth' in d.columns and filter_mode == 'gt_only':
        d = d[(d['file-level-ground-truth'] == True) & (d['prediction-prob'] >= 0.5)]
    keep = d.groupby(['test','filename'])['line-level-ground-truth'].sum()
    d = d.set_index(['test','filename']).loc[keep[keep > 0].index].reset_index()

    if per_block_top_ratio is not None and 'topk_per_block' in d.columns:
        d = d[d['topk_per_block']].copy()

    d = d.sort_values(['test','filename','block_attn','block_id','line_attn'],
                      ascending=[True, True, False, True, False]).copy()

    d['rank_in_subset'] = d.groupby(['test','filename']).cumcount() + 1
    d['subset_size']    = d.groupby(['test','filename'])['rank_in_subset'].transform('max')
    d['effort']         = d['rank_in_subset'] / d['subset_size']

    # ---- Recall@20%LOC（候选集的前20%行）
    totals = d.groupby(['test','filename'])['line-level-ground-truth'].sum().rename('total_true').reset_index()
    top20  = d[d['effort'] <= 0.2]
    correct = top20.groupby(['test','filename'])['line-level-ground-truth'].sum().rename('correct_pred').reset_index()
    rec20  = correct.merge(totals, on=['test','filename'], how='right').fillna(0.0)
    rec20['Recall20LOC'] = np.where(rec20['total_true'] > 0, rec20['correct_pred'] / rec20['total_true'], np.nan)

    # ---- Effort@20%Recall（命中20%真实缺陷行所需的候选集占比）
    d['cum_pos'] = d.groupby(['test','filename'])['line-level-ground-truth'].cumsum()
    thr = totals.copy()
    thr['thr'] = np.maximum(1, np.ceil(0.2 * thr['total_true']).astype(int))
    merged = d.merge(thr[['test','filename','thr','total_true']], on=['test','filename'], how='left')
    def first_effort(g):
        if g['total_true'].iloc[0] == 0: return np.nan
        t = g[g['cum_pos'] >= g['thr'].iloc[0]]
        return float(t['effort'].iloc[0]) if len(t) else 1.0
    eff20 = merged.groupby(['test','filename']).apply(first_effort).rename('Effort20Recall').reset_index()

    # ---- IFA（候选集合内首次命中之前看了多少行）
    pos_only = d[d['line-level-ground-truth'] == True]
    ifa_per_file = pos_only.groupby(['test','filename'])['rank_in_subset'].min().rename('first_pos_rank').reset_index()
    ifa_per_file['IFA'] = ifa_per_file['first_pos_rank'] - 1

    # # Precision@Top20%LOC
    # # 统计 Top20% 内命中数与Top20%大小（把Top20%中的所有行视作“预测为正”）
    # prec_g = top20.groupby(['test', 'filename'])['line-level-ground-truth'] \
    #     .agg(tp_top20='sum', top20_size='count').reset_index()
    # prec_g['Precision20LOC'] = np.where(prec_g['top20_size'] > 0,
    #                                     prec_g['tp_top20'] / prec_g['top20_size'], np.nan)
    #
    # # F1@Top20%LOC（按文件先算，再做宏平均）
    # pr = prec_g.merge(rec20[['test', 'filename', 'Recall20LOC']], on=['test', 'filename'], how='outer')
    # pr['F1_20LOC'] = (2 * pr['Precision20LOC'] * pr['Recall20LOC']) / \
    #                  np.where((pr['Precision20LOC'] + pr['Recall20LOC']) > 0,
    #                           (pr['Precision20LOC'] + pr['Recall20LOC']), np.nan)

    return {
        'IFA':          round(float(ifa_per_file['IFA'].mean(skipna=True)), 3) if len(ifa_per_file) else np.nan,
        'Recall20LOC':  round(float(rec20['Recall20LOC'].mean(skipna=True)), 3) if len(rec20) else np.nan,
        'Effort20Recall': round(float(eff20['Effort20Recall'].mean(skipna=True)), 3) if len(eff20) else np.nan
        # 'Precision20LOC': float(pr['Precision20LOC'].mean(skipna=True)) if len(pr) else np.nan,
        # 'F1_20LOC': float(pr['F1_20LOC'].mean(skipna=True)) if len(pr) else np.nan
    }


if __name__ == '__main__':

    path = '../results/groovy-1_6_BETA_2.csv'
    # path = '../results/cross/activemq/activemq-5.0.0-camel-2.10.0.csv'
    df = pd.read_csv(path)

    print(file_metrics(df))
    print(line_metrics(df, filter_mode='gt_only', per_block_top_ratio=1.0))
