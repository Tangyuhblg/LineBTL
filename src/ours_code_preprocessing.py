import os, sys

from torch.utils.flop_counter import suffixes

# 项目根：/root/SOUND-main
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import csv
import os.path
from pathlib import Path
import sys
from my_util import *
import os
import pandas as pd
import re
from typing import List, Dict, Any



# 去掉训练集和测试集。LLM只需要判断测试集
all_releases = {'activemq': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'],
                'camel': ['camel-2.10.0', 'camel-2.11.0'],
                'derby': ['derby-10.5.1.1'],
                'groovy': ['groovy-1_6_BETA_2'],
                'hbase': ['hbase-0.95.2'],
                # 'hive': ['hive-0.12.0'],
                'jruby': ['jruby-1.5.0', 'jruby-1.7.0.preview1'],
                'lucene': ['lucene-3.0.0', 'lucene-3.1'],
                'wicket': ['wicket-1.5.3']}

# all_releases = {'activemq': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'],
#                 'camel':['camel-1.4.0']}

# all_releases = {'derby': ['derby-10.5.1.1']
#                 }

def preprocessing_llm_data(proj_name, input_file, pre_file, out_file):
    """
    从 original_data_dir 读取 LineBB 预测结果，
    只保留 file-level-ground-truth == TRUE 且 prediction-label == TRUE 的文件，
    然后提取列：
        filename, code-line, line-number, line-attention-score, block-attention-score
    写到 out_file/rel.csv
    """
    proj_all_rel = all_releases[proj_name]

    for rel in proj_all_rel:
        csv_path = input_file + rel + '.csv'
        df = pd.read_csv(csv_path)
        print(f"成功读取文件: {csv_path}, 共 {len(df)} 行")

        pre_csv_path = pre_file + rel + '.csv'
        df_pre = pd.read_csv(pre_csv_path)
        df_pre = df_pre[df_pre['is_test_file'] == False]
        # df_pre = df_pre[df_pre['is_blank'] == False]
        cols_pre = [
            'filename',
            'line_number',
            'code_line',
            'is_comment',
            'is_blank'
        ]


        # # 1) 先根据文件级标签过滤
        # # 假定列名为 file-level-ground-truth 和 prediction-label（区分大小写）
        # if ('file-level-ground-truth' not in df.columns) or ('prediction-label' not in df.columns):
        #     raise KeyError(
        #         f"{csv_path} 中找不到 'file-level-ground-truth' 或 'prediction-label' 列，"
        #         f"实际列名：{list(df.columns)}"
        #     )
        #
        # gt_col = 'file-level-ground-truth'
        # pred_col = 'prediction-label'
        #
        # # 统一转字符串再判断 TRUE，兼容 True/true/1 等
        # gt_true = df[gt_col].astype(str).str.upper().isin(['TRUE', '1', 'YES'])
        # pred_true = df[pred_col].astype(str).str.upper().isin(['TRUE', '1', 'YES'])
        #
        # df_filtered = df[gt_true | pred_true].copy()
        # print(f"  文件级 GT=TRUE 且 预测=TRUE 的行数: {len(df_filtered)}")
        #
        # if df_filtered.empty:
        #     # 如果这一版本里没有任何 (TRUE, TRUE) 的文件，直接跳过这一版
        #     print(f"  [跳过] {rel} 中不存在 file-level-ground-truth 和 prediction-label 同为 TRUE 的文件。")
        #     continue

        # 2) 只保留生成 LLM 输入需要的列
        cols = [
            'filename',
            'origin-line-number',
            'block_id',
            'line-attention-score',
            'block-attention-score',
            'file-level-ground-truth',
            'line-level-ground-truth'
        ]
        for c in cols:
            if c not in df.columns:
                raise KeyError(
                    f"{csv_path} 中找不到必需列 '{c}'，实际列名：{list(df.columns)}"
                )

        df_merge = df_pre[cols_pre].merge(df[cols], left_on=['filename', 'line_number'], right_on=['filename', 'origin-line-number'],
                                          how='left', suffixes=('', '_pred'))

        # df_merge = df_merge[df_merge['is_blank'] == False].drop(columns=['is_blank'])
        llm_data_file = df_merge

        # 3) 写出到 ../datasets/deepseek_input_data/{rel}.csv
        output_path = os.path.join(out_file, rel + '.csv')
        os.makedirs(out_file, exist_ok=True)
        llm_data_file.to_csv(output_path, index=False, encoding='utf-8')
        print(f"已写出 LLM 输入文件: {output_path}, 行数: {len(llm_data_file)}")



def guess_column(name: str) -> str:
    """标准化列名比较：去空格、转小写、替换连字符为下划线。"""
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def pick_col(columns, candidates):
    """在列名中找到候选之一，返回真实列名；否则抛错。"""
    norm_map = {guess_column(c): c for c in columns}
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]
    raise KeyError(f"找不到必需列之一：{candidates}；CSV 实际列：{list(columns)}")


def sanitize_output_name(filename: str) -> str:
    """
    将原 filename（可能带路径）转为扁平文件名：
    1) / 或 \ → '-'
    2) 连续分隔符合并为一个 '-'
    3) 确保以 .java 结尾（若无则补上）
    """
    s = filename.strip()
    s = re.sub(r"[\\/]+", "+", s)
    # s = re.sub(r"-{2,}", "_", s).strip("-")
    if not s.lower().endswith(".java"):
        s += ".java"
    return s


def unique_path(p: Path) -> Path:
    """若目标路径已存在，自动在扩展名前追加 __2, __3, ... 去重。"""
    if not p.exists():
        return p
    stem, suffix = p.stem, p.suffix
    n = 2
    while True:
        q = p.with_name(f"{stem}__{n}{suffix}")
        if not q.exists():
            return q
        n += 1


def write_java_file(lines_map: Dict[int, str], out_path: Path, encoding: str = "utf-8"):
    """将 {行号: 文本} 写成文件，缺失行写空行。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not lines_map:
        out_path.write_text("", encoding=encoding, newline="\n")
        return
    max_ln = max(lines_map.keys())
    with out_path.open("w", encoding=encoding, newline="\n") as f:
        for ln in range(1, max_ln + 1):
            text = lines_map.get(ln, "")
            if text is None:
                text = ""
            f.write(str(text).rstrip("\n") + "\n")


# def write_java_file(lines_map: Dict[int, str], out_path: Path, encoding: str = "utf-8"):
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     if not lines_map:
#         out_path.write_text("", encoding=encoding, newline="\n")
#         return
#     max_ln = max(lines_map.keys())
#     with out_path.open("w", encoding=encoding, newline="\n") as f:
#         for ln in range(1, max_ln + 1):
#             text = lines_map.get(ln, "")
#             if text is None:
#                 text = ""
#             # 只去掉尾部换行，不做 strip()，避免误删空格/缩进/花括号等
#             f.write((text[:-1] if text.endswith("\n") else text) + "\n")
#


def is_comment_line(comment_value: Any) -> bool:
    """
    判断是否为注释行。
    支持多种格式：True/False、'TRUE'/'FALSE'、'1'/'0'、1/0
    """
    if comment_value is None:
        return False

    # 转换为字符串处理
    comment_str = str(comment_value).strip().upper()

    # 检查是否为True的各种表示
    if comment_str in ['TRUE', '1', 'YES', 'T', 'Y']:
        return True

    # 如果是布尔类型
    if isinstance(comment_value, bool):
        return comment_value

    return False


def process_csv(csv_path, out_dir, encoding: str = "utf-8", chunksize: int = 200_000,):
    """
    分块读取 CSV，聚合同一 filename 下的所有行并写出。
    支持传入 str 或 Path。
    """
    # 统一转 Path（修复 AttributeError: 'str' has no attribute 'mkdir'）
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)

    # 自动探测分隔符（逗号/制表符/分号等）
    read_kwargs = dict(
        sep=None,                 # 自动推断
        engine="python",
        dtype=str,                # 先按字符串读取
        keep_default_na=False,    # 空字符串不转 NaN
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="warn",
    )

    # 建立内存索引：{ filename: { new_line_number: code_line } }
    files: Dict[str, Dict[int, str]] = {}

    total_rows, bad_rows = 0, 0

    try:
        reader = pd.read_csv(csv_path, chunksize=chunksize, **read_kwargs)
    except Exception:
        # 非分块（小文件）回退
        reader = [pd.read_csv(csv_path, **read_kwargs)]

    def to_int_safe(x, default=None):
        try:
            s = str(x).strip()
            if s == "" or s.lower() == "nan":
                return default

            # 支持 "12.0" / "12.000"
            if re.fullmatch(r"[+-]?\d+(\.0+)?", s):
                return int(float(s))

            v = float(s)
            if v.is_integer():
                return int(v)
        except Exception:
            pass

        # 兜底：抽取第一个整数
        m = re.search(r"[+-]?\d+", str(x))
        if m:
            try:
                return int(m.group(0))
            except Exception:
                return default

        return default

    for chunk_idx, df in enumerate(reader, start=1):
        total_rows += len(df)

        # 列名归一化映射
        try:
            col_filename = pick_col(df.columns, {"filename"})
            col_code = pick_col(df.columns, {"code_line", "code-line"})
            col_lno_origin = pick_col(df.columns, {"origin-line-number", "origin_line_number"})
            col_lno = pick_col(df.columns, {"new_line_number", "line_number"})
            col_line_attn = pick_col(df.columns, {"line_attention_score", "line-attention-score"})
            col_block_attn = pick_col(df.columns, {"block_attention_score", "block-attention-score"})
            col_block_id = pick_col(df.columns, {"block_id", "block-id"})
            col_comment = pick_col(df.columns, {"is-comment-line", 'is_comment_line', 'is_comment'})
            col_blank = pick_col(df.columns, {"is-blank-line", "is_blank_line", "is_blank"})
        except KeyError as e:
            print(f"[错误] {e}", file=sys.stderr)
            sys.exit(2)

        # 仅保留必须列
        df = df[[col_filename, col_lno_origin, col_lno, col_code, col_line_attn, col_block_attn, col_block_id, col_comment, col_blank]].copy()

        # 清洗与类型转换
        df[col_filename] = df[col_filename].astype(str).str.strip()
        df[col_code] = df[col_code].astype(str).fillna("")
        df[col_line_attn] = df[col_line_attn].astype(str)
        df[col_block_attn] = df[col_block_attn].astype(str)
        df[col_comment] = df[col_comment].astype(str)
        df[col_blank] = df[col_blank].astype(str)
        df[col_block_id] = df[col_block_id].map(lambda x: to_int_safe(x, default=-1))
        df[col_lno] = df[col_lno].map(to_int_safe)
        df[col_lno_origin] = df[col_lno_origin].map(to_int_safe)

        # 丢弃非法行
        before = len(df)
        df = df.dropna(subset=[col_filename, col_lno])
        df = df[df[col_filename] != ""]
        after = len(df)
        bad_rows += (before - after)

        if col_blank:
            blanks = df[col_blank].astype(str)
        else:
            blanks = ["FALSE"] * len(df)

        for filename, lno_origin, lno, code, line_attn, block_attn, block_id, comment in zip(df[col_filename], df[col_lno_origin],
                                                                                   df[col_lno], df[col_code],
                                                                                   df[col_line_attn],
                                                                                   df[col_block_attn], df[col_block_id], df[col_comment]):
            if lno is None:
                continue
            code_str = "" if code is None else str(code)

            is_comment = is_comment_line(comment)
            is_blank = str(blanks).strip().upper() in ["TRUE"]

            if is_comment:
                augmented = code_str
            else:
                # 先收集所有行的信息，稍后统一排序
                augmented = code_str

            bucket = files.setdefault(filename, {})
            # 存储原始信息，稍后处理
            bucket[int(lno)] = {
                'code': code_str,
                'is_comment': is_comment,
                "is_blank": is_blank,
                "block_id": int(block_id) if block_id is not None else -1,
                'line_attn': float(line_attn) if line_attn and str(line_attn).strip() else 0.0,
                'block_attn': float(block_attn) if block_attn and str(block_attn).strip() else 0.0
            }

    # 在所有数据处理完成后，对每个文件进行排序并添加注释
    for filename, lines_map in files.items():
        # 提取非注释行进行排序
        non_comment_lines = [
            (ln, info) for ln, info in lines_map.items()
            if (not info["is_comment"]) and (not info.get("is_blank", False))
        ]

        # 1) 分 block_id
        by_block = {}
        for ln, info in non_comment_lines:
            bid = info.get("block_id", -1)
            if bid == -1: continue
            by_block.setdefault(bid, []).append((ln, info))

        # 2) 每个 block 内取 line_attn Top5
        selected = []
        TOPK_PER_BLOCK = 6
        for bid, items in by_block.items():
            items_sorted = sorted(
                items,
                key=lambda x: (-x[1]["line_attn"], -x[1]["block_attn"], x[0])  # tie-break：block_attn、行号
            )
            if len(items_sorted) <= TOPK_PER_BLOCK:
                selected.extend(items_sorted)
            else:
                selected.extend(items_sorted[:TOPK_PER_BLOCK])


        # 3) 合并后再按 block_attn / line_attn 排序，生成全局 rank
        selected_sorted = sorted(
            selected,
            key=lambda x: (-x[1]["block_attn"], -x[1]["line_attn"], x[0])
        )
        sorting_rank = {ln: rank + 1 for rank, (ln, _) in enumerate(selected_sorted)}

        # 4) 写出：只有在 sorting_rank 内的行才加 defect_sorting
        final_lines_map = {}
        max_ln = max(lines_map.keys()) if lines_map else 0
        for ln in range(1, max_ln + 1):
            if ln in lines_map:
                info = lines_map[ln]
                if info["is_comment"]:
                    final_lines_map[ln] = info["code"]
                else:
                    rank = sorting_rank.get(ln)
                    if rank is not None and (info["block_attn"] > 0 or info["line_attn"] > 0):
                        final_lines_map[ln] = f"{info['code']}  // defect_sorting={rank}"
                    else:
                        final_lines_map[ln] = info["code"]
            else:
                final_lines_map[ln] = ""

        # 写入文件
        flat_name = sanitize_output_name(filename)
        out_path = unique_path(out_dir / flat_name)
        write_java_file(final_lines_map, out_path, encoding=encoding)

    print(f"[输出目录] {out_dir.resolve()}")



if __name__ == '__main__':
    # 提取真实和预测文件级为TRUE的文件，保留line-attention-score和block-attention-score
    results_data_dir = '../output/prediction/LineBB/within-release/'
    pre_data_dir = '../datasets/llm_preprocessed_data/'
    save_dir = '../datasets/ours_input_data/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # code_line有"{"和"}"，保留filename code_line new_line_number
    for proj in list(all_releases.keys()):
        preprocessing_llm_data(proj, results_data_dir, pre_data_dir, save_dir)

    llm_data_path = '../datasets/ours_input_data/'
    for proj in list(all_releases.keys()):
        for rel in all_releases[proj]:
            csv_path = os.path.join(llm_data_path, rel + '.csv')
            save_java_path = os.path.join(llm_data_path, proj, rel)

            if not os.path.exists(save_java_path):
                os.makedirs(save_java_path)

            process_csv(csv_path, save_java_path)
