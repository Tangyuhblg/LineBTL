import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import os
import pandas as pd
import re
from typing import List, Dict, Any
from my_util import *
import re
import os
from typing import List, Dict, Any, Tuple, Set


class JavaBasicBlockAnalyzer:
    def __init__(self):
        self.structured_pattern = re.compile(
            r'^\s*(if|for|while|do|switch|try|catch|finally|else|synchronized)\b'
        )
        self.jump_pattern = re.compile(
            r'^\s*(return|break|continue|throw)\b'
        )
        self.label_pattern = re.compile(
            r'^\s*((case\s+.*)|(default)|([a-zA-Z_]\w*))\s*:'
        )
        self.brace_pattern = re.compile(r'^\s*[\{\}]\s*$')

    _KW_PAT = {
        "if": re.compile(r'^\s*if\b', re.I),
        "else": re.compile(r'^\s*else\b', re.I),
        "switch": re.compile(r'^\s*switch\b', re.I),
        "case": re.compile(r'^\s*case\b', re.I),
        "default": re.compile(r'^\s*default\b', re.I),
        "try": re.compile(r'^\s*try\b', re.I),
        "catch": re.compile(r'^\s*catch\b', re.I),
        "finally": re.compile(r'^\s*finally\b', re.I),
        "for": re.compile(r'^\s*for\b', re.I),
        "while": re.compile(r'^\s*while\b', re.I),
        "do": re.compile(r'^\s*do\b', re.I),
    }

    def _block_text(self, block: Dict[str, Any]) -> str:
        lines = block.get('clean_lines', None)
        if lines is None:
            lines = block.get('lines', []) or []
        # 去掉空白行，避免空行/缩进影响模式匹配
        txt = ' '.join(str(x) for x in lines if str(x).strip() != '')
        return txt.strip().lower()


    def _starts_with(self, txt: str, key: str) -> bool:
        pat = self._KW_PAT.get(key)
        return bool(pat.search(txt)) if pat else False

    def _block_ends_with_uncond_jump(self, block: Dict[str, Any]) -> bool:
        lines = block.get('clean_lines', None)
        if lines is None:
            lines = block.get('lines', []) or []
        for x in reversed(lines):
            s = str(x).strip()
            if s != '':
                return self.is_unconditional_jump(s)
        return False

    def _find_next_block(self, blocks: List[Dict[str, Any]], start_idx: int, max_ahead: int, key: str) -> int | None:
        n = len(blocks)
        for j in range(start_idx + 1, min(n, start_idx + 1 + max_ahead)):
            if self._starts_with(self._block_text(blocks[j]), key):
                return j
        return None

    def build_cfg_edges(self, blocks: List[Dict[str, Any]]) -> Set[Tuple[int, int]]:
        edges: Set[Tuple[int, int]] = set()
        n = len(blocks)
        if n <= 1:
            return edges

        for i in range(n - 1):
            if not self._block_ends_with_uncond_jump(blocks[i]):
                edges.add((i, i + 1))

        for i in range(n):
            txt = self._block_text(blocks[i])

            if self._starts_with(txt, "if"):
                if i + 1 < n: edges.add((i, i + 1))
                j_else = self._find_next_block(blocks, i, max_ahead=8, key="else")
                if j_else is not None: edges.add((i, j_else))

            if self._starts_with(txt, "switch"):
                for j in range(i + 1, min(n, i + 12)):
                    t2 = self._block_text(blocks[j])
                    if self._starts_with(t2, "case") or self._starts_with(t2, "default"):
                        edges.add((i, j))

            if self._starts_with(txt, "try"):
                for j in range(i + 1, min(n, i + 6)):
                    t2 = self._block_text(blocks[j])
                    if self._starts_with(t2, "catch") or self._starts_with(t2, "finally"):
                        edges.add((i, j))

            if (self._starts_with(txt, "for") or
                    self._starts_with(txt, "while") or
                    self._starts_with(txt, "do")):
                if i + 1 < n:
                    edges.add((i, i + 1))
                    edges.add((i + 1, i))

        preds = [set() for _ in range(n)]
        succs = [set() for _ in range(n)]
        for u, v in edges:
            if 0 <= u < n and 0 <= v < n:
                succs[u].add(v);
                preds[v].add(u)

        for i in range(n):
            if i > 0 and len(preds[i]) == 0:
                edges.add((i - 1, i))  # 至少有一个前驱
                preds[i].add(i - 1);
                succs[i - 1].add(i)
            if i < n - 1 and len(succs[i]) == 0:
                edges.add((i, i + 1))  # 至少有一个后继
                succs[i].add(i + 1);
                preds[i + 1].add(i)

        return edges

    def build_pred_succ_lists(self, num_blocks: int, edges: Set[Tuple[int, int]]) -> Dict[int, Dict[str, List[int]]]:
        preds = [set() for _ in range(num_blocks)]
        succs = [set() for _ in range(num_blocks)]
        for u, v in edges:
            if 0 <= u < num_blocks and 0 <= v < num_blocks:
                succs[u].add(v)
                preds[v].add(u)
        out = {}
        for i in range(num_blocks):
            out[i] = {
                "pred": sorted(preds[i]),
                "succ": sorted(succs[i]),
            }
        return out

    def is_structured_jump_start(self, code: str) -> bool:

        code = code.strip()

        if not code or code == ';':
            return False

        if code.startswith('//'):
            return False

        return bool(self.structured_pattern.match(code))

    def is_unconditional_jump(self, code: str) -> bool:

        code = code.strip()

        if not code or code == ';' or code.startswith('//'):
            return False

        if self.jump_pattern.match(code):
            return True

        if code.endswith(';'):
            code_body = code[:-1].strip()
            return bool(self.jump_pattern.match(code_body))

        return False

    def is_label_or_case(self, code: str) -> bool:

        code = code.strip()

        if not code or code.startswith('//'):
            return False

        return bool(self.label_pattern.match(code))

    def is_brace_only(self, code: str) -> bool:
        return bool(self.brace_pattern.match(code))

    def preprocess_code_lines(self, lines: pd.DataFrame) -> pd.DataFrame:
        processed_lines = []
        in_block_comment = False

        def strip_block_comments_and_line_comments(s: str) -> str:
            nonlocal in_block_comment
            i, n = 0, len(s)
            out = []
            while i < n:
                if not in_block_comment and i + 1 < n and s[i] == '/' and s[i + 1] == '*':
                    in_block_comment = True
                    i += 2
                    continue
                if in_block_comment:
                    if i + 1 < n and s[i] == '*' and s[i + 1] == '/':
                        in_block_comment = False
                        i += 2
                    else:
                        i += 1
                    continue
                if i + 1 < n and s[i] == '/' and s[i + 1] == '/':
                    break
                out.append(s[i])
                i += 1
            return ''.join(out)

        for _, row in lines.sort_values('line_number').iterrows():
            ln = int(row['line_number'])
            raw = row.get('code_line', '')
            if pd.isna(raw):
                raw = ''
            clean = strip_block_comments_and_line_comments(str(raw)).strip()
            processed_lines.append({'line_number': ln, 'code_line': clean})

        return pd.DataFrame(processed_lines, columns=['line_number', 'code_line'])

    def split_basic_blocks(self, lines: pd.DataFrame) -> List[Dict[str, Any]]:

        if lines.empty:
            return []

        raw_lines = lines.sort_values('line_number').copy()
        raw_lines['code_line'] = raw_lines['code_line'].apply(lambda x: '' if pd.isna(x) else str(x))
        raw_lines = raw_lines.reset_index(drop=True)

        processed_lines = self.preprocess_code_lines(raw_lines[['line_number', 'code_line']])
        processed_lines = processed_lines.reset_index(drop=True)
        n = len(processed_lines)

        if n == 0:
            return []

        processed_lines = processed_lines.reset_index(drop=True)
        leaders = {0}

        for i in range(n):
            code_line = processed_lines.at[i, 'code_line']

            if self.is_structured_jump_start(code_line):
                leaders.add(i)

                if i + 1 < n:
                    leaders.add(i + 1)

            if self.is_label_or_case(code_line):
                leaders.add(i)

            if self.is_unconditional_jump(code_line):
                if i + 1 < n:
                    leaders.add(i + 1)

            if code_line.strip() == '}' and i + 1 < n:
                leaders.add(i + 1)

            if code_line.strip() == '{' and i > 0:
                leaders.add(i)

        leaders = sorted(leaders)

        basic_blocks = []

        for idx, leader_idx in enumerate(leaders):
            start_index = leader_idx

            if idx + 1 < len(leaders):
                end_index = leaders[idx + 1] - 1
            else:
                end_index = n - 1

            actual_end = start_index
            for j in range(start_index, end_index + 1):
                code_line = processed_lines.at[j, 'code_line']
                actual_end = j
                if self.is_unconditional_jump(code_line):
                    break

            block_lines = processed_lines.loc[start_index:actual_end]
            raw_block_lines = raw_lines.loc[start_index:actual_end]

            if len(block_lines) == 0:
                continue

            basic_blocks.append({
                "start_line": int(raw_block_lines.iloc[0]['line_number']),
                "end_line": int(raw_block_lines.iloc[-1]['line_number']),
                "lines": raw_block_lines['code_line'].tolist(),
                "clean_lines": block_lines['code_line'].tolist(),
                "line_numbers": raw_block_lines['line_number'].astype(int).tolist(),  # 每一行的原始行号
                "block_size": len(raw_block_lines)
            })

        return basic_blocks

    def analyze_file(self, proj_name, input_file: str, output_file: str):
        try:
            proj_all_rel = all_releases[proj_name]

            for rel in proj_all_rel:
                df = pd.read_csv(input_file + rel + '.csv')
                df = df.reset_index(drop=True)
                df['_orig_row_idx'] = df.index

                if 'code_line' in df.columns:
                    df['code_line'] = df['code_line'].apply(lambda x: '' if pd.isna(x) else x)
                for _bcol, _default in [('is_comment', False), ('is_blank', False)]:
                    if _bcol not in df.columns:
                        if _bcol == 'is_comment':
                            raise ValueError("数据集中缺少 'is_comment' 列")
                        df[_bcol] = _default
                    df[_bcol] = df[_bcol].fillna(_default).astype(bool)

                if 'code_line' in df.columns:
                    _code = df['code_line'].astype(str)
                    _code_no_brace = _code.str.replace('{', '', regex=False).str.replace('}', '', regex=False)

                    _became_blank = _code_no_brace.str.strip().eq('') & _code.str.contains(r'[{}]', regex=True)
                    df.loc[_became_blank, 'code_line'] = ''
                    df.loc[_became_blank, 'is_blank'] = True
                    df.loc[~_became_blank, 'code_line'] = _code_no_brace.loc[~_became_blank]


                print(f"成功读取文件: {input_file + rel}, 共 {len(df)} 行")

                label_cols = ['is_test_file', 'file-label', 'line-label']

                existing_label_cols = [c for c in label_cols if c in df.columns]

                if 'line_number' not in df.columns:
                    raise ValueError("数据集中缺少 'line_number' 列，无法对齐行级标签")

                bool_columns = ['is_comment', 'is_blank']
                line_meta_df = df[['filename', 'line_number', '_orig_row_idx'] + existing_label_cols + bool_columns].copy()

                # 过滤注释行和空行
                # code_df = df[(df['is_comment'] == False) & (df['is_blank'] == False)].copy()
                code_df = df.copy()
                # print(f"过滤后代码行数: {len(code_df)}")

                # 按文件分组处理
                group_col = 'filename'
                if group_col not in code_df.columns:
                    raise ValueError(f"缺少分组列 '{group_col}'")

                grouped = code_df.groupby(group_col, sort=False)
                blocks_data = []

                # 处理每个文件
                for filename, file_group in grouped:

                    file_sorted = file_group.sort_values('line_number')
                    # 划分基本块
                    basic_blocks = self.split_basic_blocks(file_sorted[['line_number', 'code_line']])
                    # print(f"  - 生成 {len(basic_blocks)} 个基本块")

                    edges = self.build_cfg_edges(basic_blocks)  # {(u,v)}
                    num_blocks = len(basic_blocks)
                    pred_succ_map = self.build_pred_succ_lists(num_blocks, edges)  # {i: {'pred': [...], 'succ': [...]}}

                    if 'cfg_rows' not in locals():
                        cfg_rows = []
                    for bid in range(num_blocks):
                        cfg_rows.append({
                            'filename': filename,
                            'block_id': bid,
                            'start_line': basic_blocks[bid]['start_line'],
                            'end_line': basic_blocks[bid]['end_line'],
                            'block_size': basic_blocks[bid]['block_size'],
                            'pred_blocks': '|'.join(map(str, pred_succ_map[bid]['pred'])) if pred_succ_map[bid][
                                'pred'] else '',
                            'succ_blocks': '|'.join(map(str, pred_succ_map[bid]['succ'])) if pred_succ_map[bid][
                                'succ'] else '',
                        })

                    for block_id, block in enumerate(basic_blocks):
                        preds = pred_succ_map[block_id]['pred']
                        succs = pred_succ_map[block_id]['succ']
                        pred_str = '|'.join(map(str, preds)) if preds else ''
                        succ_str = '|'.join(map(str, succs)) if succs else ''
                        for orig_line_number, line_content in zip(block['line_numbers'], block['lines']):
                            blocks_data.append({
                                'filename': filename,
                                'block_id': block_id,
                                'start_line': block['start_line'],
                                'end_line': block['end_line'],
                                'block_size': block['block_size'],
                                'code_line': line_content,
                                'orig_line_number': int(orig_line_number),
                                'pred_blocks': pred_str,  # <-- 新增
                                'succ_blocks': succ_str,  # <-- 新增
                            })

                if blocks_data:
                    blocks_df = pd.DataFrame(blocks_data)
                    unique_files = df['filename'].unique()
                    file_order_map = {filename: idx for idx, filename in enumerate(unique_files)}
                    blocks_df['file_order'] = blocks_df['filename'].map(file_order_map)
                    required_columns = ['filename', 'block_id', 'start_line', 'end_line', 'block_size', 'code_line']
                    missing_columns = [col for col in required_columns if col not in blocks_df.columns]
                    if missing_columns:
                        print(f"警告: 缺少必要的列: {missing_columns}")

                    # print(f"开始处理数据，原始行数: {len(blocks_df)}")

                    # 1 保障 code_line：保留注释原始格式；空行保持为空字符串（不要变成 'nan'）
                    blocks_df['code_line'] = blocks_df['code_line'].apply(lambda x: '' if pd.isna(x) else x)

                    # 删除多余空格
                    # blocks_df['code_line'] = re.sub(r'\s+', ' ', blocks_df['code_line']).strip()
                    blocks_df['code_line'] = blocks_df['code_line'].fillna('').astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

                    # 2 检查并删除空行
                    blocks_df['is_empty'] = blocks_df['code_line'].str.strip().eq('')
                    empty_count = blocks_df['is_empty'].sum()
                    # print(f"发现空行数量: {empty_count}")

                    # 保存删除空行前的数据用于重建CFG
                    before_clean_df = blocks_df.copy()

                    # # 删除空行
                    # df_clean = blocks_df[~blocks_df['is_empty']].copy()
                    # df_clean = df_clean.drop('is_empty', axis=1)
                    # df_clean = df_clean.reset_index(drop=True)

                    # 不删除空行：直接保留所有行进入后续流程
                    df_clean = blocks_df.copy()
                    df_clean = df_clean.drop('is_empty', axis=1)
                    df_clean = df_clean.reset_index(drop=True)

                    # 为每个文件重新计算行号
                    result_dfs = []

                    for filename in df_clean['filename'].unique():
                        file_df = df_clean[df_clean['filename'] == filename].copy()
                        # print(f"处理文件: {filename}, 行数: {len(file_df)}")

                        # 按block_id分组处理
                        new_start_line = 1

                        for block_id in sorted(file_df['block_id'].unique()):
                            block_df = file_df[file_df['block_id'] == block_id].copy()
                            block_size = len(block_df)

                            # 计算新的结束行
                            new_end_line = new_start_line + block_size - 1

                            # 更新block内的所有行
                            block_df.loc[:, 'start_line'] = new_start_line
                            block_df.loc[:, 'end_line'] = new_end_line
                            block_df.loc[:, 'block_size'] = block_size

                            # 更新下一个block的起始行
                            new_start_line = new_end_line + 1

                            result_dfs.append(block_df)

                    final_df = pd.concat(result_dfs, ignore_index=True)
                    final_df = final_df.sort_values(['file_order', 'start_line'], ascending=[True, True]).reset_index(
                        drop=True)
                    final_df['old_block_id'] = final_df['block_id']
                    final_df['block_id'] = (
                        final_df
                        .groupby('filename')['old_block_id']
                        .transform(lambda s: pd.factorize(s)[0])  # 0,1,2,... 连续
                    )
                    updated_blocks_data = []

                    for filename in final_df['filename'].unique():
                        file_df = final_df[final_df['filename'] == filename].copy()

                        file_blocks = []
                        for block_id in sorted(file_df['block_id'].unique()):
                            block_rows = file_df[file_df['block_id'] == block_id]
                            if len(block_rows) == 0:
                                continue

                            block = {
                                "start_line": int(block_rows.iloc[0]['start_line']),
                                "end_line": int(block_rows.iloc[-1]['end_line']),
                                "lines": block_rows['code_line'].tolist(),
                                "line_numbers": block_rows['orig_line_number'].tolist(),
                                "block_size": len(block_rows)
                            }
                            file_blocks.append(block)

                        edges = self.build_cfg_edges(file_blocks)
                        num_blocks = len(file_blocks)
                        pred_succ_map = self.build_pred_succ_lists(num_blocks, edges)

                        # 更新每个块的前驱后继
                        for block_id, block in enumerate(file_blocks):
                            preds = pred_succ_map[block_id]['pred']
                            succs = pred_succ_map[block_id]['succ']
                            pred_str = '|'.join(map(str, preds)) if preds else ''
                            succ_str = '|'.join(map(str, succs)) if succs else ''

                            # 获取该块对应的所有行
                            block_rows = file_df[file_df['block_id'] == block_id]
                            for idx, row in block_rows.iterrows():
                                updated_row = {
                                    'filename': filename,
                                    'block_id': block_id,
                                    'start_line': block['start_line'],
                                    'end_line': block['end_line'],
                                    'block_size': block['block_size'],
                                    'code_line': row['code_line'],
                                    'orig_line_number': int(row['orig_line_number']),
                                    'pred_blocks': pred_str,
                                    'succ_blocks': succ_str,
                                }
                                updated_blocks_data.append(updated_row)

                    # 用更新后的数据替换final_df
                    final_df = pd.DataFrame(updated_blocks_data)

                    # 重新添加文件顺序
                    final_df['file_order'] = final_df['filename'].map(file_order_map)

                    # 删除临时列
                    if 'old_block_id' in final_df.columns:
                        final_df = final_df.drop('old_block_id', axis=1)

                    # # 三列合并
                    # final_df = final_df.merge(line_meta_df, how='left', left_on=['filename', 'orig_line_number'],
                    #                           right_on=['filename', 'line_number'])
                    # final_df = final_df.drop(columns=['line_number'])
                    #
                    # # 删除文件顺序列
                    # final_df = final_df.drop('file_order', axis=1)
                    #
                    # # 重新排序，保持原始顺序
                    # final_df = final_df.sort_values(['filename', 'start_line'], ascending=[True, True]).reset_index(
                    #     drop=True)

                    # 三列合并
                    final_df = final_df.merge(
                        line_meta_df,
                        how='left',
                        left_on=['filename', 'orig_line_number'],
                        right_on=['filename', 'line_number']
                    ).drop(columns=['line_number'])

                    # 关键：严格按输入 CSV 的原始行顺序输出（稳定排序）
                    final_df = final_df.sort_values('_orig_row_idx', kind='stable').reset_index(drop=True)

                    # 清理辅助列
                    final_df = final_df.drop(columns=['_orig_row_idx', 'file_order'], errors='ignore')

                    # 添加代码行号
                    final_df['new_line_number'] = 0

                    for filename in final_df['filename'].unique():
                        file_mask = final_df['filename'] == filename
                        file_row_count = file_mask.sum()
                        final_df.loc[file_mask, 'new_line_number'] = range(1, file_row_count + 1)

                    # 重新排列列的顺序，将 new_line_number 放在合适的位置
                    columns = list(final_df.columns)
                    # 将 new_line_number 移到 code_line 后面
                    code_line_idx = columns.index('code_line')
                    columns.insert(code_line_idx + 1, 'new_line_number')
                    columns.remove('new_line_number')
                    final_df = final_df[columns]

                    # 确保输出目录存在
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)

                    # 保存结果前做一次兜底清洗
                    # 确保 code_line 为空行时输出为空字符串（不要输出 NaN / 'nan'）
                    if 'code_line' in final_df.columns:
                        final_df['code_line'] = final_df['code_line'].apply(lambda x: '' if pd.isna(x) else x)
                    # merge 带回来的布尔列缺失值兜底
                    for _bcol in ['is_comment', 'is_blank']:
                        if _bcol in final_df.columns:
                            final_df[_bcol] = final_df[_bcol].fillna(False).astype(bool)

                    final_df.to_csv(output_file + rel + '.csv', index=False, encoding='utf-8')
                    print(f"{rel}.csv基本块信息已保存到: {os.path.abspath(output_file)}")
                    # print(f"总共生成 {len(blocks_df)} 行基本块数据")
                else:
                    print("警告: 没有生成任何基本块数据")

        except Exception as e:
            print(f"处理文件时出错: {str(e)}")
            raise


if __name__ == '__main__':

    original_data_dir = '../datasets/preprocessed_data/'
    save_dir = '../datasets/basic_block_data/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    analyzer = JavaBasicBlockAnalyzer()

    for proj in list(all_releases.keys()):
        analyzer.analyze_file(proj, original_data_dir, save_dir)