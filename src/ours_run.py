import os, sys
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import openai
import requests
from openai import OpenAI
import threading
import argparse
import concurrent.futures as cf
import time


def choose_max_tokens(n_lines: int, n_chars: int, hard_cap: int = 4096) -> int:
    # Completion budget heuristic (keep it safe/capped)
    est = int(900 + n_lines * 3 + n_chars / 450)  # 估算最大token数，基于行数和字符数
    return max(1200, min(hard_cap, est))  # 返回最大token数，确保不超过硬性限制（默认为4096）


def iter_java_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.java") if p.is_file()])


def read_java(input_root: str) -> str:
    p = Path(input_root)
    if not p.exists():
        raise FileNotFoundError(f"Not exist: {p}")

    # 如果是目录：递归找第一个 .java
    if p.is_dir():
        java_files = sorted(p.rglob("*.java"))
        if not java_files:
            raise FileNotFoundError(f"No .java file found under directory: {p}")
        p = java_files[0]

    # 如果是文件：确保是 .java
    if p.suffix.lower() != ".java":
        raise ValueError(f"Expected a .java file, got: {p}")

    # 以纯文本读取 .java
    return p.read_text(encoding="utf-8", errors="replace")

def add_line_numbers(code_text: str) -> str:
    lines = code_text.splitlines()
    # 使用固定宽度对齐，方便模型引用
    width = max(4, len(str(len(lines))))
    return "\n".join([f"{str(i+1).rjust(width)} | {line}" for i, line in enumerate(lines)])


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extract and parse the first JSON object from model output.
    Repairs common wrappers like ```json ... ``` and leading/trailing text.
    """
    if not isinstance(text, str):
        raise TypeError(f"extract_json_object expects str, got {type(text)}")

    s = text.strip()

    # strip markdown fences
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)

    m = _JSON_RE.search(s)
    if not m:
        raise ValueError("No JSON object found in model output.")

    json_str = m.group(0).strip()
    data = json.loads(json_str)

    # normalize has_bug to bool
    hb = data.get("has_bug", False)
    if isinstance(hb, str):
        hb_norm = hb.strip().lower()
        if hb_norm in ("true"):
            hb = True
        elif hb_norm in ("false"):
            hb = False
        else:
            hb = False
    data["has_bug"] = bool(hb)

    # enforce schema rules
    if not data["has_bug"]:
        data["defective_line_numbers"] = []
        data.pop("lines", None)
    else:
        dln = data.get("defective_line_numbers", [])
        if not isinstance(dln, list):
            dln = []
        data["defective_line_numbers"] = [int(x) for x in dln if str(x).strip().isdigit()]

        lines_arr = data.get("lines", [])
        if not isinstance(lines_arr, list):
            lines_arr = []
        cleaned = []
        for item in lines_arr:
            if not isinstance(item, dict):
                continue
            if "line_number" not in item or "code_line" not in item:
                continue
            try:
                ln = int(item["line_number"])
            except Exception:
                continue
            cleaned.append({"line_number": ln, "code_line": str(item["code_line"])})

        if cleaned:
            data["lines"] = cleaned
            data["defective_line_numbers"] = [x["line_number"] for x in cleaned]
        else:
            if not data["defective_line_numbers"]:
                data["defective_line_numbers"] = []

    return data


def cot_prompt(clients, code_document, model, max_tokens):
    """

    """
    conversation_history = []
    numbered_code = add_line_numbers(code_document)

    # 引入注意力先验排序
    hint_rows = []
    for ln, raw_line in enumerate(code_document.splitlines(), start=1):
        m = re.search(r"//\s*defect_sorting\s*=\s*([0-9eE+\-\.]+)", raw_line)
        if not m:
            continue
        try:
            ds = float(m.group(1))
        except Exception:
            continue
        hint_rows.append((ds, ln, raw_line))

    hint_rows.sort(key=lambda x: x[0])  # smaller => more suspicious (rank 1 best)
    # num_line = len(code_document.splitlines())
    # top_k = int(num_line * 0.2)
    # top_k = 30
    # top_hint = hint_rows[:top_k]
    top_hint = hint_rows[:]

    if top_hint:
        risk_hint = "\n".join(
            [f"- line {ln}: defect_sorting={ds:g} | {raw_line}" for ds, ln, raw_line in top_hint]
        )
    else:
        risk_hint = "(No `// defect_sorting=...` hints found in this file.)"

    # prompt_1 = (
    #     "You are a professional Java code reviewer.\n"
    #     "Below is a Java source file with line numbers. Please:\n"
    #     "1) Summarize what the code is trying to do.\n"
    #     "2) Identify potential bugs / vulnerabilities / spec violations / compile-time issues.\n"
    #     "3) For each issue, reference the most relevant line numbers.\n\n"
    #     "Java code (with line numbers):\n"
    #     f"{numbered_code}\n"
    # )

    prompt_1 = (
        f"Here is the Java code with line numbers (the number before '|' is the line number):\n"
        f"Code:\n{numbered_code}\n"
        "Please provide a detailed summary of the code's functionality, analyze the code structure"
    )

    conversation_history.append({"role": "user", "content": prompt_1}) # 用户说提示
    response_1 = clients.chat.completions.create(model=model, messages=conversation_history, stream=False, max_tokens=max_tokens) # 创建对话
    # analysis_text1 = response_1.choices[0].message # LLM回答文本
    conversation_history.append(response_1.choices[0].message) # 保存LLM的所有回答
    code_analysis1 = (response_1.choices[0].message.content or "").strip() # LLM输出

    prompt_2 = (
        f"Based on the previous analysis:\n'{code_analysis1}'\n"
        "Evaluate whether the code has any significant vulnerabilities. Only answer 'true' or 'false'"
    )
    conversation_history.append({"role": "user", "content": prompt_2})
    response_2 = clients.chat.completions.create(model=model, messages=conversation_history, stream=False, max_tokens=max_tokens)
    conversation_history.append(response_2.choices[0].message)
    code_analysis2 = (response_2.choices[0].message.content or "").strip()

    prompt_3 = (
        f"Based on the previous analysis:\n'{code_analysis1}'\n and '{code_analysis2}'\n"
        "You can get an attention-based prior list (risk hint) derived from `// defect_sorting=...`.\n\n"
        "Risk hint (attention prior; smaller defect_sorting => more suspicious):\n"
        f"{risk_hint}\n\n"
        "Output a SINGLE JSON object ONLY (no extra text).\n"
        "Schema requirements:\n"
        "{\n"
        f'  "has_bug": {code_analysis2},\n'
        '  "defective_line_numbers": integer[],\n'
        '  "lines"?: [\n'
        "     {\n"
        '       "line_number": integer,\n'
        '       "code_line": string,\n'
        "     }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- If has_bug is false, set defective_line_numbers to [] and OMIT the field \"lines\".\n"
        "- If has_bug is true, provide the most suspicious lines (at least one line)."
        "- Output between 20 and 30 suspicious lines (prefer 25). This is recall-first."
        "- Prioritize choosing line numbers from the Risk hint list.\n"
        "- You may add adjacent lines (+/- 1 or +/- 2) around a risky line to capture multi-line defects.\n"
        "defective_line_numbers must match the line_number values in \"lines\".\n"
        "- The line_number values in defective_line_numbers must be unique.\n"
        "- code_line MUST be copied exactly from the given code (the content after the '|').\n"
        "- Only output JSON.\n"
    ) # "- code_line MUST be copied exactly from the given code (the content after the '|').\n"
    conversation_history.append({"role": "user", "content": prompt_3})
    response_3 = clients.chat.completions.create(model=model, messages=conversation_history, stream=False, max_tokens=max_tokens)
    code_analysis3 = (response_3.choices[0].message.content or "").strip()
    data = extract_json_object(code_analysis3)
    return data


def save_results(record: Dict[str, Any], java_path: Path, input_root: Path, output_root: Path) -> Path:
    """
    为每个 .java 写出一个对应 .json：
      output_root / 相对路径 / 同名.json
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    input_root = Path(input_root)
    java_path = Path(java_path)

    rel = java_path.relative_to(input_root)                 # e.g., org/apache/.../X.java
    out_path = (output_root / rel).with_suffix(".json")     # e.g., org/apache/.../X.json
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 规则：has_bug==False 时不输出 lines
    payload = {
        "filename": str(rel).replace("/", "+"),
        "has_bug": bool(record.get("has_bug", False)),
        "defective_line_numbers": record.get("defective_line_numbers", []),
    }
    if payload["has_bug"] and "lines" in record:
        payload["lines"] = record["lines"]

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)



def call_with_retries(fn, max_retries: int, base_sleep: float = 1.0):
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            sleep_s = base_sleep * (2 ** attempt) + (0.05 * attempt)
            time.sleep(min(sleep_s, 20))
    raise last_err


def process_files_parallel(input_root, output_root, client, workers, max_inflight_requests, max_retries):

    java_files = iter_java_files(input_root)
    total = len(java_files)
    if total == 0:
        return []

    inflight_sem = threading.Semaphore(max_inflight_requests)
    results: List[Optional[Dict[str, Any]]] = [None] * total

    def worker(idx_path: Tuple[int, Path]) -> Dict[str, Any]:
        idx, java_path = idx_path
        rel_name = str(java_path.relative_to(input_root)).replace("\\", "/")

        try:
            code_text = read_java(java_path)
            n_lines = code_text.count("\n") + 1
            n_chars = len(code_text)
            max_tokens = choose_max_tokens(n_lines, n_chars)

            def _do_call():
                with inflight_sem:
                    return cot_prompt(client, code_text, model=model, max_tokens=max_tokens)

            data = call_with_retries(_do_call, max_retries=max_retries)

            record = {
                "filename": rel_name,
                "has_bug": data.get("has_bug", False),
                "defective_line_numbers": data.get("defective_line_numbers", []),
            }
            if data.get("has_bug") and "lines" in data:
                record["lines"] = data["lines"]

        except Exception as e:
            record = {
                "filename": rel_name,
                "has_bug": False,
                "defective_line_numbers": [],
                "error": f"{type(e).__name__}: {e}",
            }

        save_results(record, java_path, input_root, output_root)
        return {"idx": idx, **record}

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, (i, p)) for i, p in enumerate(java_files)]
        for fut in tqdm(cf.as_completed(futures), total=total, desc="LLM analyzing", unit="file"):
            r = fut.result()
            results[r["idx"]] = r

    return [r for r in results if r is not None]


if __name__ == '__main__':

    input_root = Path('../datasets/ours_input_data/camel/camel-2.11.0/')
    output_root = Path('../datasets/ours_output_data/camel/camel-2.11.0/')

    if not output_root.exists(): os.makedirs(output_root)

    base_url = 'https://api.deepseek.com'
    model = 'deepseek-chat'
    api_key = '' # your key
    client = OpenAI(api_key=api_key, base_url=base_url)

    workers = 32 # 线程数
    max_inflight_requests = 32 # 同时进行LLM请求数量
    max_retries = 4 # 重试次数

    if not api_key:
        raise SystemExit("Missing API key. Set DMX_API_KEY (or OPENAI_API_KEY) or pass --api_key.")

    process_files_parallel(input_root, output_root, client, workers, max_inflight_requests, max_retries)

