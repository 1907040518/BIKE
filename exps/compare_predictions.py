#!/usr/bin/env python3
"""
compare_predictions.py
对比两个模型的 video_predictions.txt，输出详细差异分析报告。

使用方法：
    python /home/neimedia/gmk/BIKE/exps/compare_predictions.py \
        --file_a /home/neimedia/gmk/BIKE/exps/hmdb51/ViT-B/16/Video-Text交互/my_1_predictions.txt \
        --file_b /home/neimedia/gmk/BIKE/exps/hmdb51/ViT-B/16/Video-Text不交互/my_1_predictions.txt \
        --output /home/neimedia/gmk/BIKE/exps/hmdb51/comparison_report.txt \
        --name_a "CoAPT" \
        --name_b "CoAPT+fution"
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path


# ============================================================
# 解析 video_predictions.txt
# ============================================================
def parse_predictions(filepath: str) -> dict:
    """
    解析 video_predictions.txt，返回以 video_path 为 key 的字典。

    每条记录结构：
    {
        "dataset_index": int,
        "video_path":    str,
        "ground_truth":  str,   # 类别名称
        "predicted":     str,   # 类别名称
        "confidence":    float,
        "correct":       bool,
    }
    """
    records = {}          # key: video_path -> record dict
    index_to_path = {}    # key: dataset_index -> video_path（备用）

    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"预测文件不存在: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 找到表头行之后的数据行
    # 格式: ListIdx   Correct?   GT Label   Pred Label   Confidence   Video Path
    in_table = False
    for line in lines:
        line = line.rstrip("\n")

        # 识别表头分隔线（连续 '-' 超过 30 个）
        if re.match(r"^-{30,}$", line.strip()):
            in_table = True
            continue

        # 遇到空行或 '=' 开头的分隔线，退出表格区域
        if in_table and (line.strip() == "" or line.strip().startswith("=")):
            in_table = False
            continue

        # 跳过表头行（含 "ListIdx" 关键字）
        if "ListIdx" in line:
            continue

        if not in_table:
            continue

        # 解析数据行
        # 格式（空格对齐）：
        # 0         ✓          archery                       archery                      0.8821       /data/...
        # 用正则：前5列固定宽度，最后一列是路径（可含空格）
        m = re.match(
            r"^(\d+)\s+"           # ListIdx
            r"([✓✗])\s+"           # Correct?
            r"(\S.*?\S|\S+)\s{2,}" # GT Label  (至少2空格分隔)
            r"(\S.*?\S|\S+)\s{2,}" # Pred Label
            r"(\d+\.\d+)\s+"       # Confidence
            r"(.+)$",              # Video Path
            line,
        )
        if m is None:
            # 备用：按列宽切割（与 save_video_predictions 的格式对齐）
            # ListIdx:<9  Correct?:<10  GT:<28  Pred:<28  Conf:<12  Path
            try:
                ds_idx    = int(line[0:9].strip())
                correct   = line[9:19].strip() == "✓"
                gt_label  = line[19:47].strip()
                pred_label= line[47:75].strip()
                confidence= float(line[75:87].strip())
                video_path= line[87:].strip()
            except (ValueError, IndexError):
                continue
        else:
            ds_idx     = int(m.group(1))
            correct    = m.group(2) == "✓"
            gt_label   = m.group(3).strip()
            pred_label = m.group(4).strip()
            confidence = float(m.group(5))
            video_path = m.group(6).strip()

        record = {
            "dataset_index": ds_idx,
            "video_path":    video_path,
            "ground_truth":  gt_label,
            "predicted":     pred_label,
            "confidence":    confidence,
            "correct":       correct,
        }
        records[video_path]  = record
        index_to_path[ds_idx] = video_path

    return records, index_to_path


# ============================================================
# 生成对比报告
# ============================================================
def compare_and_report(
    records_a: dict,
    records_b: dict,
    index_to_path_a: dict,
    index_to_path_b: dict,
    name_a: str,
    name_b: str,
    output_path: str,
):
    # 以 video_path 为主键对齐；若路径不一致则尝试用 dataset_index 对齐
    all_paths_a = set(records_a.keys())
    all_paths_b = set(records_b.keys())
    common_paths = all_paths_a & all_paths_b
    only_in_a    = all_paths_a - all_paths_b
    only_in_b    = all_paths_b - all_paths_a

    # 若路径完全不重叠，尝试用 dataset_index 对齐（路径前缀不同的情况）
    if len(common_paths) == 0 and len(records_a) > 0 and len(records_b) > 0:
        print("[WARNING] 两个文件的 video_path 无交集，尝试按 dataset_index 对齐...")
        common_indices = set(index_to_path_a.keys()) & set(index_to_path_b.keys())
        aligned_pairs = [
            (records_a[index_to_path_a[i]], records_b[index_to_path_b[i]])
            for i in sorted(common_indices)
        ]
    else:
        aligned_pairs = [
            (records_a[p], records_b[p])
            for p in sorted(common_paths, key=lambda p: records_a[p]["dataset_index"])
        ]

    # 分类
    both_correct   = []   # A✓ B✓
    a_only_correct = []   # A✓ B✗
    b_only_correct = []   # A✗ B✓
    both_wrong     = []   # A✗ B✗  (但预测类别可能不同)
    both_wrong_same_pred = []   # A✗ B✗ 且预测相同
    both_wrong_diff_pred = []   # A✗ B✗ 且预测不同

    for rec_a, rec_b in aligned_pairs:
        ca, cb = rec_a["correct"], rec_b["correct"]
        if ca and cb:
            both_correct.append((rec_a, rec_b))
        elif ca and not cb:
            a_only_correct.append((rec_a, rec_b))
        elif not ca and cb:
            b_only_correct.append((rec_a, rec_b))
        else:
            both_wrong.append((rec_a, rec_b))
            if rec_a["predicted"] == rec_b["predicted"]:
                both_wrong_same_pred.append((rec_a, rec_b))
            else:
                both_wrong_diff_pred.append((rec_a, rec_b))

    total = len(aligned_pairs)
    acc_a = sum(1 for r, _ in aligned_pairs if r["correct"]) / total * 100 if total else 0
    acc_b = sum(1 for _, r in aligned_pairs if r["correct"]) / total * 100 if total else 0

    # ---- 按类别统计 A✓B✗ 和 A✗B✓ ----
    class_a_wins  = defaultdict(int)   # A✓B✗ 按 GT 类别
    class_b_wins  = defaultdict(int)   # A✗B✓ 按 GT 类别
    class_total   = defaultdict(int)

    for rec_a, rec_b in aligned_pairs:
        gt = rec_a["ground_truth"]
        class_total[gt] += 1
        if rec_a["correct"] and not rec_b["correct"]:
            class_a_wins[gt] += 1
        elif not rec_a["correct"] and rec_b["correct"]:
            class_b_wins[gt] += 1

    # ---- 写报告 ----
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    W = 130  # 分隔线宽度

    def sep(char="="):
        return char * W + "\n"

    def header(title):
        return sep() + f"{title}\n" + sep()

    with open(output_path, "w", encoding="utf-8") as f:

        # ── 总览 ──────────────────────────────────────────────
        f.write(header("PREDICTION COMPARISON REPORT"))
        f.write(f"  Model A : {name_a}\n")
        f.write(f"  Model B : {name_b}\n")
        f.write(f"  Aligned samples  : {total}\n")
        if only_in_a:
            f.write(f"  Only in A        : {len(only_in_a)} samples (skipped)\n")
        if only_in_b:
            f.write(f"  Only in B        : {len(only_in_b)} samples (skipped)\n")
        f.write("\n")
        f.write(f"  {name_a} Accuracy : {acc_a:.2f}%\n")
        f.write(f"  {name_b} Accuracy : {acc_b:.2f}%\n")
        f.write(f"  Accuracy Delta   : {acc_a - acc_b:+.2f}%  ({name_a} vs {name_b})\n")
        f.write("\n")
        f.write(f"  {'Category':<40} {'Count':>8}   {'% of total':>10}\n")
        f.write("  " + "-" * 62 + "\n")
        categories = [
            (f"Both Correct  (A✓ B✓)",          len(both_correct)),
            (f"{name_a} only correct  (A✓ B✗)", len(a_only_correct)),
            (f"{name_b} only correct  (A✗ B✓)", len(b_only_correct)),
            (f"Both Wrong, same pred (A✗ B✗ →same)", len(both_wrong_same_pred)),
            (f"Both Wrong, diff pred (A✗ B✗ →diff)", len(both_wrong_diff_pred)),
        ]
        for cat_name, cnt in categories:
            pct = cnt / total * 100 if total else 0
            f.write(f"  {cat_name:<40} {cnt:>8}   {pct:>9.2f}%\n")
        f.write("\n")

        # ── 辅助：列表写入函数 ────────────────────────────────
        col_w = {"idx": 8, "path": 55, "gt": 25, "pred": 25, "conf": 10}

        def write_table_header(f, label_a, label_b):
            f.write(
                f"  {'#':<5} {'ListIdx':<8} "
                f"{'GT Label':<25} "
                f"[{label_a}] {'Pred':<22} {'Conf':>8}   "
                f"[{label_b}] {'Pred':<22} {'Conf':>8}   "
                f"Video Path\n"
            )
            f.write("  " + "-" * (W - 2) + "\n")

        def write_row(f, idx, rec_a, rec_b):
            f.write(
                f"  {idx:<5} {rec_a['dataset_index']:<8} "
                f"{rec_a['ground_truth']:<25} "
                f"{'✓' if rec_a['correct'] else '✗'} {rec_a['predicted']:<22} {rec_a['confidence']:>8.4f}   "
                f"{'✓' if rec_b['correct'] else '✗'} {rec_b['predicted']:<22} {rec_b['confidence']:>8.4f}   "
                f"{rec_a['video_path']}\n"
            )

        # ── Section 1: A✓ B✗ ─────────────────────────────────
        f.write(header(f"[SECTION 1]  {name_a} CORRECT  /  {name_b} WRONG   "
                       f"(A✓ B✗)  —  {len(a_only_correct)} samples"))
        if not a_only_correct:
            f.write("  (none)\n\n")
        else:
            write_table_header(f, name_a, name_b)
            for idx, (ra, rb) in enumerate(
                sorted(a_only_correct, key=lambda x: x[0]["dataset_index"]), 1
            ):
                write_row(f, idx, ra, rb)
            f.write("\n")

        # ── Section 2: A✗ B✓ ─────────────────────────────────
        f.write(header(f"[SECTION 2]  {name_a} WRONG  /  {name_b} CORRECT   "
                       f"(A✗ B✓)  —  {len(b_only_correct)} samples"))
        if not b_only_correct:
            f.write("  (none)\n\n")
        else:
            write_table_header(f, name_a, name_b)
            for idx, (ra, rb) in enumerate(
                sorted(b_only_correct, key=lambda x: x[0]["dataset_index"]), 1
            ):
                write_row(f, idx, ra, rb)
            f.write("\n")

        # ── Section 3: A✗ B✗ 预测不同 ────────────────────────
        f.write(header(f"[SECTION 3]  BOTH WRONG but DIFFERENT predictions   "
                       f"(A✗ B✗ →diff)  —  {len(both_wrong_diff_pred)} samples"))
        if not both_wrong_diff_pred:
            f.write("  (none)\n\n")
        else:
            write_table_header(f, name_a, name_b)
            for idx, (ra, rb) in enumerate(
                sorted(both_wrong_diff_pred, key=lambda x: x[0]["dataset_index"]), 1
            ):
                write_row(f, idx, ra, rb)
            f.write("\n")

        # ── Section 4: A✗ B✗ 预测相同 ────────────────────────
        f.write(header(f"[SECTION 4]  BOTH WRONG and SAME prediction   "
                       f"(A✗ B✗ →same)  —  {len(both_wrong_same_pred)} samples"))
        if not both_wrong_same_pred:
            f.write("  (none)\n\n")
        else:
            write_table_header(f, name_a, name_b)
            for idx, (ra, rb) in enumerate(
                sorted(both_wrong_same_pred, key=lambda x: x[0]["dataset_index"]), 1
            ):
                write_row(f, idx, ra, rb)
            f.write("\n")

        # ── Section 5: 按类别统计差异 ─────────────────────────
        f.write(header("[SECTION 5]  PER-CLASS DIFFERENCE SUMMARY"))
        f.write(
            f"  {'GT Class':<30} {'Total':>7} "
            f"{'A✓B✗':>7} {'A✗B✓':>7} "
            f"{'Net(A-B)':>10}   Advantage\n"
        )
        f.write("  " + "-" * 75 + "\n")

        all_classes = sorted(
            set(list(class_a_wins.keys()) + list(class_b_wins.keys()) + list(class_total.keys()))
        )
        for cls in all_classes:
            tot  = class_total[cls]
            a_w  = class_a_wins[cls]
            b_w  = class_b_wins[cls]
            net  = a_w - b_w
            adv  = f"← {name_a}" if net > 0 else (f"← {name_b}" if net < 0 else "tie")
            f.write(
                f"  {cls:<30} {tot:>7} {a_w:>7} {b_w:>7} {net:>+10}   {adv}\n"
            )
        f.write("\n")

        # ── Section 6: 仅 A 有 / 仅 B 有 的样本 ──────────────
        if only_in_a or only_in_b:
            f.write(header("[SECTION 6]  UNMATCHED SAMPLES"))
            if only_in_a:
                f.write(f"  --- Only in {name_a} ({len(only_in_a)} samples) ---\n")
                for p in sorted(only_in_a):
                    r = records_a[p]
                    f.write(f"  [{r['dataset_index']}] {p}\n")
                f.write("\n")
            if only_in_b:
                f.write(f"  --- Only in {name_b} ({len(only_in_b)} samples) ---\n")
                for p in sorted(only_in_b):
                    r = records_b[p]
                    f.write(f"  [{r['dataset_index']}] {p}\n")
                f.write("\n")

    print(f"[✓] 对比报告已保存至: {output_path}")
    print(f"    对齐样本数: {total}")
    print(f"    {name_a} Acc: {acc_a:.2f}%  |  {name_b} Acc: {acc_b:.2f}%  |  Delta: {acc_a - acc_b:+.2f}%")
    print(f"    A✓B✗: {len(a_only_correct)}  |  A✗B✓: {len(b_only_correct)}  |  "
          f"Both✗(diff): {len(both_wrong_diff_pred)}  |  Both✗(same): {len(both_wrong_same_pred)}")


# ============================================================
# 入口
# ============================================================
def get_parser():
    parser = argparse.ArgumentParser(description="Compare two model prediction txt files")
    parser.add_argument("--file_a",  type=str, required=True, help="Model A 的 video_predictions.txt")
    parser.add_argument("--file_b",  type=str, required=True, help="Model B 的 video_predictions.txt")
    parser.add_argument("--output",  type=str, default="comparison_report.txt", help="输出报告路径")
    parser.add_argument("--name_a",  type=str, default="ModelA", help="Model A 的显示名称")
    parser.add_argument("--name_b",  type=str, default="ModelB", help="Model B 的显示名称")
    return parser


def main():
    args = get_parser().parse_args()

    print(f"[*] 解析 {args.name_a}: {args.file_a}")
    records_a, idx2path_a = parse_predictions(args.file_a)
    print(f"    读取到 {len(records_a)} 条记录")

    print(f"[*] 解析 {args.name_b}: {args.file_b}")
    records_b, idx2path_b = parse_predictions(args.file_b)
    print(f"    读取到 {len(records_b)} 条记录")

    compare_and_report(
        records_a, records_b,
        idx2path_a, idx2path_b,
        args.name_a, args.name_b,
        args.output,
    )


if __name__ == "__main__":
    main()
