"""真值交叉验证：人工肉眼标注 与 程序化真值（photo_manifest.json）的一致率。

用途（C 分工的测试与文档工作链一环）：
    分组算法的"真值"来自 data/photo_manifest.json——它是按文件名规则程序化生成的
    （photo_XXXX_{变换}.jpg，同一 photo_XXXX 属于同一来源组），标注无歧义但可能
    "过于理想"。为验证这套真值本身符合人工直觉，生成一份随机乱序的标注模板，
    由人工独立肉眼判断哪些图片属于同一组，再与本脚本对比计算一致率。

用法：
    1. 生成标注模板（500 张，行序随机打乱，固定种子可复现）：
         py cross_validate_labels.py --make-template data/evaluation/annotation_template.csv
    2. 人工填写模板：给"同一组的图片"填相同的组号（可留空，留空行不参与统计）。
    3. 计算人工标注与程序真值的一致率：
         py cross_validate_labels.py data/evaluation/annotation_template.csv

统计口径（以程序真值为基准，人工标注为预测）：
    仅统计"人工已标注且存在于真值清单"的图片对：
        Precision = 人工判同组 ∩ 真值同组 / 人工判同组      （人工合并是否准确）
        Recall    = 人工判同组 ∩ 真值同组 / 真值同组        （真值组是否被人工找回）
        F1        = 2PR/(P+R)
        Agreement = (真值同组且人工同组 + 真值异组且人工异组) / 全部有效图片对
    并输出：有效图片覆盖率、未标注文件清单、分歧最多的真值组 TOP5。
"""
import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / "data" / "library"
MANIFEST = ROOT / "data" / "photo_manifest.json"
DEFAULT_TEMPLATE = ROOT / "data" / "evaluation" / "annotation_template.csv"


def load_ground_truth():
    """读取程序真值：{文件名: 来源组}。"""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    truth = {}
    for group in manifest["groups"]:
        for member in group["members"]:
            truth[member] = group["source_id"]
    return truth


def make_template(out_path):
    """生成标注模板：全部素材文件名随机乱序，附 来源目录 提示，供人工标注。"""
    names = sorted(p.name for p in LIBRARY.iterdir() if p.is_file())
    # 固定种子打乱行序：同一份模板可复现，且文件名顺序不泄露来源编号规律。
    random.Random(20261021).shuffle(names)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["序号", "文件名", "来源提示", "分组标注（同组填相同编号，可留空）", "备注"])
        for index, name in enumerate(names, 1):
            writer.writerow([index, name, str(LIBRARY), "", ""])
    print(f"已生成标注模板：{out_path}（{len(names)} 行）")


def validate(csv_path):
    """计算人工标注与程序真值的一致率，输出指标与待补标清单。"""
    truth = load_ground_truth()
    label = {}  # 文件名 -> 人工组号
    unlabeled = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.reader(stream):
            if not row or row[0] == "序号":
                continue
            name, value = row[1], (row[3].strip() if len(row) > 3 else "")
            if value:
                label[name] = value
            else:
                unlabeled.append(name)
    # 仅统计"人工已标注 且 真值存在"的图片（部分标注也支持）。
    covered = {name for name in label if name in truth}
    names = sorted(covered)
    total = len(truth)
    print(f"素材总数 {total}，已标注 {len(label)}，有效参与统计 {len(covered)}，覆盖率 {len(covered) / total:.1%}")
    if not names:
        print("没有可统计的标注，请先在模板中填写分组标注。")
        return
    tp = fp = fn = tn = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            same_truth = truth[names[i]] == truth[names[j]]
            same_label = label[names[i]] == label[names[j]]
            if same_truth and same_label:
                tp += 1
            elif not same_truth and same_label:
                fp += 1
            elif same_truth and not same_label:
                fn += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    agreement = (tp + tn) / (tp + fp + fn + tn)
    print(f"\n人工标注 vs 程序真值（{len(names)} 张有效图片，共 {tp + fp + fn + tn} 对）")
    print(f"  查准率 Precision = {precision:.2%}（人工判同组的准确程度）")
    print(f"  召回率 Recall    = {recall:.2%}（真值同组被人工找回的比例）")
    print(f"  F1               = {f1:.4f}")
    print(f"  一致率 Agreement = {agreement:.2%}（含异组一致的总体一致率）")
    # 分歧最多的真值组 TOP5：展示人工在哪些来源上最容易判错，供复核。
    disagreement = []
    for source_id in sorted(set(truth.values())):
        members = [name for name in names if truth[name] == source_id]
        if len(members) < 2:
            continue
        wrong = 0
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if label[members[i]] != label[members[j]]:
                    wrong += 1
        if wrong:
            disagreement.append((source_id, len(members), wrong))
    if disagreement:
        print("\n分歧最多的真值来源组 TOP5（成员数, 未判为同组对子数）：")
        for source_id, count, wrong in sorted(disagreement, key=lambda item: -item[2])[:5]:
            print(f"  {source_id}（{count} 张，{wrong} 对未判同组）")
    if unlabeled:
        print(f"\n未标注文件 {len(unlabeled)} 个，可留空继续使用（已标注部分会照常统计）。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-template", nargs="?", const=DEFAULT_TEMPLATE,
                        metavar="CSV", help="生成标注模板（默认输出到 data/evaluation/annotation_template.csv）")
    parser.add_argument("csv", nargs="?", default=None,
                        help="已填写的人工标注 CSV，用于计算一致率")
    args = parser.parse_args()
    if args.make_template:
        make_template(Path(args.make_template))
    elif args.csv:
        validate(Path(args.csv))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
