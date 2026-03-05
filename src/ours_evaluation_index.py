import os
import argparse
from pathlib import Path
import math
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef, balanced_accuracy_score, confusion_matrix
from tqdm import tqdm
import ast
import re
import math


def file_metrics(df):

    cols = ['filename', 'file-level-ground-truth', 'file-prob']
    file_df = df[cols].drop_duplicates()

    y_true = file_df['file-level-ground-truth'].astype(int).values  # True->1, False->0
    file_df['file-prob'] = file_df['file-prob'].fillna(0)
    y_pred = file_df['file-prob'].astype(int).values

    # AUC：基于真实标签与概率
    auc = roc_auc_score(y_true, y_pred)

    # MCC：Matthews 相关系数，全面衡量二分类（-1 ~ 1）
    mcc = matthews_corrcoef(y_true, y_pred)

    # Balanced Accuracy（BA） = (TPR + TNR)/2
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    # print(tn, fp, fn, tp)
    tpr = tp/(tp+fn) if (tp+fn)>0 else 0.0
    tnr = tn/(tn+fp) if (tn+fp)>0 else 0.0
    ba  = (tpr + tnr)/2

    return {"AUC": auc, "MCC": mcc, "BA": ba}


def _to_int_set(x):
    """Robustly parse a CSV cell into a set[int]."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return set()

    # already container
    if isinstance(x, (list, tuple, set, np.ndarray)):
        out = set()
        for v in x:
            try:
                out.add(int(v))
            except Exception:
                pass
        return out

    # scalar number
    if isinstance(x, (int, np.integer)):
        return {int(x)}
    if isinstance(x, (float, np.floating)) and not np.isnan(x):
        return {int(x)} if float(x).is_integer() else {int(round(float(x)))}

    # string cases
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return set()

        # try parse like "[1,2,3]" / "(1,2)" / "{1,2}" / "1"
        try:
            obj = ast.literal_eval(s)
            return _to_int_set(obj)
        except Exception:
            # fallback: extract all integers from "1,2|3;4" etc.
            nums = re.findall(r"-?\d+", s)
            return {int(n) for n in nums}

    return set()


def line_metrics(df):
    mask = (df["file-level-ground-truth"] == 1) & (df["file-prob"] == 1)
    file_df = df.loc[mask, ["filename", "defective_line_numbers", "origin-line-number"]].copy()

    n_files = len(df[mask])
    # 检查n_files是否为零，避免除零错误
    if n_files == 0:
        return "No files match the criteria."

    hits = []
    for _, row in file_df.iterrows():
        prob = _to_int_set(row["defective_line_numbers"])
        truth = _to_int_set(row["origin-line-number"])
        hit_lines = sorted(prob & truth)
        hits.append({
            "filename": row["filename"],
            "n_hits": len(hit_lines)
        })
    per_file = pd.DataFrame(hits)
    # 总命中
    total_hits = int(per_file["n_hits"].sum()) if not per_file.empty else 0
    # 一共多少缺陷
    total_true_lines = file_df["origin-line-number"].apply(lambda x: len(_to_int_set(x))).sum()

    # 平均每个文件命中，符合的文件数量，总命中数量，总缺陷行数
    return total_hits / n_files, n_files, total_hits / total_true_lines, total_hits, total_true_lines # 平均每个文件命中多少行


if __name__ == "__main__":
    # all_releases = {'activemq': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'],
    #                 'camel': ['camel-2.10.0', 'camel-2.11.0'],
    #                 'derby': ['derby-10.5.1.1'],
    #                 'groovy': ['groovy-1_6_BETA_2'],
    #                 'hbase': ['hbase-0.95.2'],
    #                 # 'hive': ['hive-0.12.0'],
    #                 'jruby': ['jruby-1.5.0', 'jruby-1.7.0.preview1'],
    #                 'lucene': ['lucene-3.0.0', 'lucene-3.1'],
    #                 'wicket': ['wicket-1.5.3']}
    all_releases = {'camel': ['camel-2.10.0', 'camel-2.11.0']
                    }

    for proj in list(all_releases.keys()):
        for rel in all_releases[proj]:
            out_dir = os.path.join('../results/ours/', proj)
            df = pd.read_csv(os.path.join(out_dir, rel + '.csv'))
            file_summary = file_metrics(df)
            line_summary = line_metrics(df)
            print(file_summary)
            print(line_summary)




