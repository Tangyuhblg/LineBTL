import os
import json
import re
from pathlib import Path
from typing import Any, List

import pandas as pd
from tqdm import tqdm


def load_file_probs_from_json_dir(json_root: str) -> pd.DataFrame:
    """
    扫描 json_root 下所有 .json 文件，解析文件级 has_defect/has_bug 和 defective_line_numbers 信息，
    返回一个 DataFrame: [filename, file-prob, defective_line_numbers]。

    - filename：要与 CSV 中的 filename 对齐；
    - file-prob：字符串 "TRUE" 或 "FALSE"；
    - defective_line_numbers：List[int]（可为空列表）。
    """
    json_root = Path(json_root)
    if not json_root.exists():
        return pd.DataFrame(columns=["filename", "file-prob", "defective_line_numbers"])

    json_files = list(json_root.rglob("*.json"))
    if not json_files:
        return pd.DataFrame(columns=["filename", "file-prob", "defective_line_numbers"])

    def _to_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        if isinstance(v, str):
            return v.strip().lower() in {"true", "yes", "y", "1", "bug", "defect"}
        return False

    def _parse_line_numbers(raw: Any, data: dict) -> List[int]:
        # 1) 直接是 list
        if isinstance(raw, list):
            nums = []
            for x in raw:
                try:
                    nums.append(int(x))
                except Exception:
                    continue
            return sorted(set(nums))

        # 2) 字符串：可能是 "[1,2]" 或 "1, 2" 或 "line: 1 2"
        if isinstance(raw, str):
            s = raw.strip()
            # 尝试按 JSON 解析
            try:
                obj = json.loads(s)
                return _parse_line_numbers(obj, data)
            except Exception:
                pass
            # 提取所有数字
            nums = [int(x) for x in re.findall(r"\d+", s)]
            return sorted(set(nums))

        # 3) 缺失 defective_line_numbers，但有 lines: [{"line_number":...}, ...]
        lines = data.get("lines")
        if isinstance(lines, list):
            nums = []
            for item in lines:
                if isinstance(item, dict) and "line_number" in item:
                    try:
                        nums.append(int(item["line_number"]))
                    except Exception:
                        continue
            if nums:
                return sorted(set(nums))

        return []

    records: list[tuple[str, str, List[int]]] = []

    for jf in tqdm(json_files, desc="扫描 JSON 文件(文件级)"):
        # 读取 JSON
        try:
            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # 若顶层不是 dict（极少数情况），尽量兜底
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if not isinstance(data, dict):
            continue

        # ==== 文件名规范化：逻辑和之前保持一致，并兼容 filename/file_name ====
        filename = None
        file_name_in_json = data.get("file_name")
        if not (isinstance(file_name_in_json, str) and file_name_in_json.strip()):
            file_name_in_json = data.get("filename")  # 兼容字段

        if isinstance(file_name_in_json, str) and file_name_in_json.strip():
            tmp = file_name_in_json.strip()
            if "/" in tmp:
                filename = tmp
            else:
                filename = tmp.replace("+", "/")
        else:
            # fallback：根据 JSON 文件路径推断
            rel = jf.relative_to(json_root)
            if len(rel.parts) == 1:
                stem = rel.stem
                if stem.endswith(".java"):
                    filename = stem.replace("+", "/")
                else:
                    filename = stem.replace("+", "/") + ".java"
            else:
                filename = str(rel.with_suffix(".java")).replace(os.sep, "/")

        # 补上 .java 后缀
        if not filename.endswith(".java"):
            filename = filename + ".java"

        # ==== 解析 has_defect / has_bug → "TRUE" / "FALSE" ====
        has_defect = data.get("has_defect", None)
        if has_defect is None:
            has_defect = data.get("has_bug", False)  # 兼容字段

        has_defect = _to_bool(has_defect)
        file_prob = "TRUE" if has_defect else "FALSE"

        # ==== 解析 defective_line_numbers ====
        raw_lines = data.get("defective_line_numbers", None)
        defective_line_numbers = _parse_line_numbers(raw_lines, data)

        records.append((str(filename).strip(), file_prob, defective_line_numbers))

    if not records:
        return pd.DataFrame(columns=["filename", "file-prob", "defective_line_numbers"])

    file_df = pd.DataFrame(records, columns=["filename", "file-prob", "defective_line_numbers"])
    file_df["filename"] = file_df["filename"].astype(str).str.strip()

    # 同一文件只保留最后一条（保持你原来的策略）
    file_df = file_df.drop_duplicates(subset=["filename"], keep="last").reset_index(drop=True)
    return file_df


def extract_gt_lines_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    df["file-level-ground-truth"] = df["file-level-ground-truth"].astype(int)
    df["line-level-ground-truth"] = df["line-level-ground-truth"].astype(int)
    df["origin-line-number"] = df["origin-line-number"].astype(int)

    # 根据filename聚合
    file_gt = (df.groupby("filename", as_index=False)["file-level-ground-truth"].max())

    # 文件级和行级都为TRUE
    hit = df[(df["file-level-ground-truth"] == 1) & (df["line-level-ground-truth"] == 1)]
    line_list = (
        hit.groupby("filename")["origin-line-number"]
        .apply(lambda s: sorted(set(s.tolist())))
        .reset_index(name="origin-line-number")
    )

    # Others
    out = file_gt.merge(line_list, on="filename", how="left")
    out["origin-line-number"] = out["origin-line-number"].apply(lambda x: x if isinstance(x, list) else [])

    out["file-level-ground-truth"] = out["file-level-ground-truth"].map(lambda b: "TRUE" if b else "FALSE")

    return out


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
    all_releases = {
        'camel': ['camel-2.10.0', 'camel-2.11.0']
    }

    for proj, rels in all_releases.items():
        for rel in rels:
            # 处理json
            json_dir = os.path.join("../datasets/ours_output_data/", proj, rel)
            out_dir = os.path.join("../results/ours/", proj)
            os.makedirs(out_dir, exist_ok=True)
            out_csv = os.path.join(out_dir, rel + ".csv")
            df = load_file_probs_from_json_dir(json_dir)

            # 处理csv
            csv_path = os.path.join('../output/prediction/LineBB/within-release/', rel + '.csv')
            gt_df = extract_gt_lines_from_csv(csv_path)

            # json与csv合并
            merged_df = pd.merge(
                df,
                gt_df,
                on="filename",
                how="outer",  # 如果你只想保留 JSON 中出现的文件，改成 how="left"
                suffixes=("_pred", "_gt")  # 同名列冲突时保留两份
            )

            # 调整列顺序：filename + df的其它列 + gt_df的其它列
            df_cols = [c for c in df.columns if c != "filename"]
            gt_cols = [c for c in gt_df.columns if c != "filename"]
            merged_df = merged_df[["filename"] + df_cols + gt_cols]

            # 保存合并后的结果
            merged_df.to_csv(out_csv, index=False, encoding="utf-8")
