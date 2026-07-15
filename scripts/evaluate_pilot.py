"""
從去識別化 Firestore snapshot 產生單店試行 KPI。

第一版只讀取本機 JSON；不直接連線 Firebase，也不會寫入正式資料。

使用方式：
    python scripts/evaluate_pilot.py --snapshot "C:\\path\\to\\snapshot"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """讀取命令列提供的 snapshot 資料夾路徑。"""
    parser = argparse.ArgumentParser(
        description="從匿名化 Firestore snapshot 計算試行 KPI。"
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="包含 manifest.json 與 collection JSON 的資料夾。",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict | list:
    """讀取一個 JSON 檔，並在資料夾不完整時中止。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到必要檔案：{path}")

    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON 格式錯誤：{path}") from error


def load_snapshot(snapshot_dir: Path) -> dict[str, object]:
    """載入第一版 KPI 所需的四份 snapshot 資料。"""
    if not snapshot_dir.is_dir():
        raise NotADirectoryError(f"snapshot 資料夾不存在：{snapshot_dir}")

    return {
        "manifest": load_json(snapshot_dir / "manifest.json"),
        "replenishment": load_json(snapshot_dir / "replenishment.json"),
        "arrival_history": load_json(snapshot_dir / "arrivalHistory.json"),
        "issue_reports": load_json(snapshot_dir / "issueReports.json"),
    }


def calculate_abc_xyz_coverage(records: object) -> tuple[int, int, float | None]:
    """計算同時有 ABC 與 XYZ 分級的 SKU 覆蓋率。"""
    if not isinstance(records, list):
        raise ValueError("replenishment.json 的最外層必須是 JSON 陣列。")

    total_skus = len(records)
    classified_skus = sum(
        1
        for record in records
        if isinstance(record, dict)
        and record.get("ABC_Class")
        and record.get("XYZ_Class")
    )

    coverage_pct = (
        round(classified_skus / total_skus * 100, 1) if total_skus else None
    )
    return total_skus, classified_skus, coverage_pct


def calculate_target_stock_coverage(records: object) -> tuple[int, int, float | None]:
    """計算有目標庫存數值的 SKU 覆蓋率。"""
    if not isinstance(records, list):
        raise ValueError("replenishment.json 的最外層必須是 JSON 陣列。")

    total_skus = len(records)
    target_stock_skus = sum(
        1
        for record in records
        if isinstance(record, dict)
        and record.get("Target_Stock") is not None
        and record.get("Target_Stock") != ""
    )

    coverage_pct = (
        round(target_stock_skus / total_skus * 100, 1) if total_skus else None
    )
    return total_skus, target_stock_skus, coverage_pct


def calculate_replenishment_gap(records: object) -> tuple[int, int, float | None]:
    """計算目標庫存高於現有庫存的補貨候選 SKU。"""
    if not isinstance(records, list):
        raise ValueError("replenishment.json 的最外層必須是 JSON 陣列。")

    total_skus = len(records)
    gap_skus = sum(
        1
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("Target_Stock"), (int, float))
        and isinstance(record.get("CurrStock"), (int, float))
        and record["Target_Stock"] > max(0, record["CurrStock"])
    )

    gap_pct = round(gap_skus / total_skus * 100, 1) if total_skus else None
    return total_skus, gap_skus, gap_pct


def calculate_arrival_activity(records: object) -> tuple[int, int]:
    """計算到貨紀錄數與不同操作者標籤數。"""
    if not isinstance(records, list):
        raise ValueError("arrivalHistory.json 的最外層必須是 JSON 陣列。")

    operator_labels = {
        record.get("createdBy")
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("createdBy"), str)
        and record.get("createdBy") != "operator_unknown"
    }
    return len(records), len(operator_labels)


def main() -> None:
    args = parse_args()
    snapshot = load_snapshot(args.snapshot)

    manifest = snapshot["manifest"]
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json 的最外層必須是 JSON 物件。")

    observation_window = manifest.get("observationWindow")
    if not isinstance(observation_window, dict):
        raise ValueError("manifest.json 缺少 observationWindow。")

    start_date = observation_window.get("startDate", "unknown")
    end_date = observation_window.get("endDate", "unknown")
    total_skus, classified_skus, coverage_pct = calculate_abc_xyz_coverage(
        snapshot["replenishment"]
    )
    _, target_stock_skus, target_stock_coverage_pct = (
        calculate_target_stock_coverage(snapshot["replenishment"])
    )
    _, gap_skus, gap_pct = calculate_replenishment_gap(snapshot["replenishment"])
    arrival_records, arrival_operators = calculate_arrival_activity(
        snapshot["arrival_history"]
    )

    print(f"Snapshot: {args.snapshot}")
    print(f"觀測期間：{start_date} 至 {end_date}")
    print(f"補貨資料筆數：{total_skus}")
    print(
        "ABC/XYZ 分級覆蓋率："
        f"{coverage_pct}%（{classified_skus} / {total_skus} SKU）"
    )
    print(
        "目標庫存覆蓋率："
        f"{target_stock_coverage_pct}%（{target_stock_skus} / {total_skus} SKU）"
    )
    print(f"補貨候選 SKU：{gap_skus}（{gap_pct}%）")
    print(f"到貨紀錄筆數：{arrival_records}")
    print(f"不同操作者標籤：{arrival_operators}")
    print(f"問題回報筆數：{len(snapshot['issue_reports'])}")

if __name__ == "__main__":
    main()