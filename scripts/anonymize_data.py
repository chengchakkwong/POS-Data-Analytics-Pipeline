"""
scripts/anonymize_data.py

從真實 POS 資料中抽取 200 SKU，脫敏後輸出至 sample_data/。
所有 mapping 只存記憶體，不落地成檔案。
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------

MISC_PRODUCT_CODES: set[str] = {"202320232023"}

MISC_NAME_KEYWORDS: list[str] = ["Deleted", "膠袋徵費", "塑膠袋", "五金家品"]

MISC_BARCODE_VALUES: set[str] = {"", "0", "0000000000000", "DELETED"}

MISC_BARCODE_PLACEHOLDER = "0000000000000"

INPUT_STOCK_FILENAME = "DetailGoodsStockToday.csv"
INPUT_SALES_DIRNAME = "sales_daily_parquet"

# ---------------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="從真實 POS 資料抽樣並脫敏，產出 sample_data/。"
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("sample_data"))
    parser.add_argument("--sku-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-shift-days", type=int, default=90)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不寫檔，只印 sanity report。",
    )
    return parser.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=level,
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# 2. 資料讀取
# ---------------------------------------------------------------------------


def load_stock(input_dir: Path) -> pd.DataFrame:
    path = input_dir / INPUT_STOCK_FILENAME
    for enc in ("utf-8-sig", "cp950", "gbk"):
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.debug("stock 以 %s 讀取，共 %d 列", enc, len(df))
            return df
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"無法解碼 {path}，已嘗試 utf-8-sig / cp950 / gbk")


def load_sales(input_dir: Path) -> pd.DataFrame:
    path = input_dir / INPUT_SALES_DIRNAME
    df = pd.read_parquet(path)
    logger.debug("sales 讀取完成，共 %d 列", len(df))
    return df


# ---------------------------------------------------------------------------
# 3. Misc 辨識
# ---------------------------------------------------------------------------


def detect_misc(df_stock: pd.DataFrame, df_sales: pd.DataFrame) -> set[int]:
    """回傳判定為 misc 的 GoodsID 集合。"""
    stock_gids = set(df_stock["GoodsID"].dropna().astype(int))
    sales_gids = set(df_sales["GoodsID"].dropna().astype(int))

    misc: set[int] = set()

    for _, row in df_stock.iterrows():
        gid = int(row["GoodsID"])
        # 去掉 float 讀入時產生的 ".0" 尾綴，再比對
        product_code = str(row.get("ProductCode", "")).strip().split(".")[0]
        name = str(row.get("Name", ""))
        barcode = str(row.get("Barcode", "")).strip().split(".")[0]

        if product_code in MISC_PRODUCT_CODES:
            misc.add(gid)
            continue
        if any(kw in name for kw in MISC_NAME_KEYWORDS):
            misc.add(gid)
            continue
        if barcode in MISC_BARCODE_VALUES:
            misc.add(gid)
            continue

    # 在 sales 出現但 stock 主檔不存在的 GoodsID
    ghost_gids = sales_gids - stock_gids
    misc.update(ghost_gids)

    logger.debug("detect_misc：辨識出 %d 個 misc GoodsID", len(misc))
    return misc


# ---------------------------------------------------------------------------
# 4. SKU 抽樣
# ---------------------------------------------------------------------------


def sample_skus(
    df_stock: pd.DataFrame,
    df_sales: pd.DataFrame,
    n: int,
    seed: int,
    misc_gids: set[int],
) -> set[int]:
    """
    分層抽樣：
    - misc  ≈ 10%（最多 20 個）
    - 正常商品按銷售貢獻度切頭 / 中 / 尾，各佔剩餘名額的 1/3
    """
    rng = np.random.default_rng(seed)

    n_misc = min(int(n * 0.10), len(misc_gids))
    n_normal = n - n_misc

    # --- misc 抽樣 ---
    # stock 裡有主檔的 misc 具有 demo 價值（例如「膠袋徵費」「五金家品」），
    # 先強制保留，再用 ghost GID 補足 misc 名額。
    stock_gids = set(df_stock["GoodsID"].dropna().astype(int))
    stock_misc = sorted(misc_gids & stock_gids)
    selected_misc = stock_misc[:n_misc]

    remaining_misc_slots = n_misc - len(selected_misc)
    if remaining_misc_slots > 0:
        misc_pool = np.array(sorted(misc_gids - set(selected_misc)))
        selected_misc.extend(
            rng.choice(
                misc_pool,
                size=min(remaining_misc_slots, len(misc_pool)),
                replace=False,
            ).tolist()
        )

    # --- 正常商品：按貢獻度分三段 ---
    contribution = (
        df_sales.groupby("GoodsID")["TotalAmt"].sum().rename("contribution")
    )
    normal_pool = (
        df_stock[~df_stock["GoodsID"].isin(misc_gids)]
        .merge(contribution, on="GoodsID", how="left")
        .fillna({"contribution": 0})
        .sort_values("contribution", ascending=False)
    )
    all_normal_gids = normal_pool["GoodsID"].astype(int).tolist()

    def split_thirds(lst: list) -> tuple[list, list, list]:
        k = len(lst)
        a = lst[: k // 3]
        b = lst[k // 3 : 2 * k // 3]
        c = lst[2 * k // 3 :]
        return a, b, c

    head, mid, tail = split_thirds(all_normal_gids)

    each = n_normal // 3
    remainder = n_normal - each * 3

    selected_normal: list[int] = []
    for i, pool in enumerate([head, mid, tail]):
        count = each + (1 if i < remainder else 0)
        count = min(count, len(pool))
        selected_normal.extend(
            rng.choice(np.array(pool), size=count, replace=False).tolist()
        )

    result = set(int(g) for g in selected_misc + selected_normal)
    logger.info(
        "sample_skus：選出 %d SKU（misc=%d，正常=%d）",
        len(result),
        len(selected_misc),
        len(selected_normal),
    )
    return result


# ---------------------------------------------------------------------------
# 5. Master mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MasterMaps:
    goods_id: dict[int, int]
    name: dict[int, str]
    barcode: dict[int, str]
    product_code: dict[int, str]
    supplier: dict[str, str]
    location: dict[str, str]
    category: dict[str, str]
    price_shift: dict[int, float]
    is_misc: dict[int, bool]


def _get_barcode(df_stock: pd.DataFrame, gid: int) -> str:
    rows = df_stock[df_stock["GoodsID"] == gid]["Barcode"]
    if rows.empty:
        return ""
    return str(rows.iloc[0]).strip()


def build_maps(
    df_stock: pd.DataFrame,
    df_sales: pd.DataFrame,
    misc_gids: set[int],
    seed: int,
) -> MasterMaps:
    fake = Faker("zh_TW")
    Faker.seed(seed)
    np_rng = np.random.default_rng(seed)

    all_gids = sorted(
        int(g)
        for g in pd.concat(
            [df_stock["GoodsID"], df_sales["GoodsID"]]
        ).dropna().unique()
    )

    goods_id_map: dict[int, int] = {
        old: 10_000 + i for i, old in enumerate(all_gids)
    }

    name_map = _build_name_map(df_stock, all_gids, misc_gids, fake)

    barcode_map: dict[int, str] = {}
    for g in all_gids:
        if g in misc_gids:
            barcode_map[g] = MISC_BARCODE_PLACEHOLDER
        else:
            barcode_map[g] = _generate_unique_ean13(fake, set(barcode_map.values()))

    product_code_map: dict[int, str] = {
        g: fake.bothify(text="SKU-#####-??").upper() for g in all_gids
    }

    supplier_map = _build_label_map(df_stock, "Supplier", "Supplier")
    location_map = _build_label_map(df_stock, "InboundLocation", "LOC")
    category_map = _build_label_map(df_stock, "Category", "CAT")

    price_shift_map: dict[int, float] = {
        g: float(np_rng.uniform(0.8, 1.2)) for g in all_gids
    }

    is_misc_map: dict[int, bool] = {g: (g in misc_gids) for g in all_gids}

    return MasterMaps(
        goods_id=goods_id_map,
        name=name_map,
        barcode=barcode_map,
        product_code=product_code_map,
        supplier=supplier_map,
        location=location_map,
        category=category_map,
        price_shift=price_shift_map,
        is_misc=is_misc_map,
    )


def _build_label_map(df: pd.DataFrame, column: str, prefix: str) -> dict[str, str]:
    if column not in df.columns:
        return {}
    values = sorted(str(v) for v in df[column].dropna().unique())
    return {value: f"{prefix}-{i:04d}" for i, value in enumerate(values, start=1)}


def _build_name_map(
    df_stock: pd.DataFrame,
    all_gids: list[int],
    misc_gids: set[int],
    fake: Faker,
) -> dict[int, str]:
    name_map: dict[int, str] = {}
    stock_by_gid = df_stock.set_index("GoodsID") if "GoodsID" in df_stock.columns else None

    for gid in all_gids:
        if gid in misc_gids and stock_by_gid is not None and gid in stock_by_gid.index:
            row = stock_by_gid.loc[gid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            name_map[gid] = _generic_demo_name(row)
        else:
            name_map[gid] = fake.bothify(text="Type-??##").upper()
    return name_map


def _generic_demo_name(row: pd.Series) -> str:
    name = str(row.get("Name", ""))
    product_code = str(row.get("ProductCode", "")).strip().split(".")[0]

    if "膠袋徵費" in name or "塑膠袋" in name:
        return "膠袋徵費"
    if "五金家品" in name:
        return "五金家品雜項"
    if product_code in MISC_PRODUCT_CODES:
        return "五金家品雜項"
    return "Deleted Generic Item"


def _generate_unique_ean13(fake: Faker, existing: set[str]) -> str:
    for _ in range(100):
        code = fake.ean13()
        if code not in existing:
            return code
    return fake.ean13()


# ---------------------------------------------------------------------------
# 6. 資料轉換
# ---------------------------------------------------------------------------


def anonymize_stock(df: pd.DataFrame, maps: MasterMaps) -> pd.DataFrame:
    original_gid = df["GoodsID"].astype(int)
    out = df.copy()

    out["GoodsID"] = original_gid.map(maps.goods_id)
    out["Name"] = original_gid.map(maps.name)
    out["Barcode"] = original_gid.map(maps.barcode)

    if "ProductCode" in out.columns:
        out["ProductCode"] = original_gid.map(maps.product_code)
    if "Supplier" in out.columns:
        out["Supplier"] = out["Supplier"].astype(str).map(maps.supplier).fillna("Supplier-0000")
    if "InboundLocation" in out.columns:
        out["InboundLocation"] = (
            out["InboundLocation"].astype(str).map(maps.location).fillna("LOC-0000")
        )
    if "Category" in out.columns:
        out["Category"] = out["Category"].astype(str).map(maps.category).fillna("CAT-0000")

    shift = original_gid.map(maps.price_shift)
    for col in ("LastInCost", "AvgCost", "RetailPrice"):
        if col in out.columns:
            out[col] = (out[col] * shift).round(2)

    if "CurrStock" in out.columns:
        out["CurrStock"] = out["CurrStock"].fillna(0).round().astype(int)

    if "Note" in out.columns:
        out["Note"] = ""

    return out


def anonymize_sales(
    df: pd.DataFrame,
    maps: MasterMaps,
    time_shift_days: int,
) -> pd.DataFrame:
    original_gid = df["GoodsID"].astype(int)
    out = df.copy()

    out["GoodsID"] = original_gid.map(maps.goods_id)

    if "rDate" in out.columns:
        col = out["rDate"]
        if pd.api.types.is_datetime64_any_dtype(col):
            dates = col
        else:
            numeric = pd.to_numeric(col, errors="coerce")
            dates = pd.to_datetime(
                numeric.where(numeric.notna(), other=pd.NA)
                .dropna()
                .astype(int)
                .astype(str)
                .reindex(col.index),
                format="%Y%m%d",
                errors="coerce",
            )
        shifted = dates - pd.Timedelta(days=time_shift_days)
        out["rDate"] = (
            shifted.dt.strftime("%Y%m%d")
            .pipe(pd.to_numeric, errors="coerce")
            .astype("Int64")
        )

    shift = original_gid.map(maps.price_shift)
    if "TotalAmt" in out.columns:
        out["TotalAmt"] = (out["TotalAmt"] * shift).round(2)

    return out


# ---------------------------------------------------------------------------
# 7. 一致性修正
# ---------------------------------------------------------------------------


def recompute_total_amt(
    df_sales: pd.DataFrame, df_stock: pd.DataFrame
) -> pd.DataFrame:
    """強制 TotalAmt = TotalQty × RetailPrice，確保內部自洽。"""
    if "RetailPrice" not in df_stock.columns or "TotalQty" not in df_sales.columns:
        return df_sales
    price_lookup = df_stock.set_index("GoodsID")["RetailPrice"].to_dict()
    out = df_sales.copy()
    out["TotalAmt"] = (
        out["TotalQty"] * out["GoodsID"].map(price_lookup).fillna(0)
    ).round(2)
    return out


# ---------------------------------------------------------------------------
# 8. Sanity check
# ---------------------------------------------------------------------------


def run_sanity_checks(
    stock_raw: pd.DataFrame,
    sales_raw: pd.DataFrame,
    stock_anon: pd.DataFrame,
    sales_anon: pd.DataFrame,
) -> str:
    lines = ["=" * 40, "  Anonymization Sanity Report", "=" * 40]

    lines.append(
        f"SKU count       : {stock_raw.shape[0]:>6} → {stock_anon.shape[0]}"
    )
    lines.append(
        f"Sales rows      : {sales_raw.shape[0]:>6} → {sales_anon.shape[0]}"
    )

    # Misc：stock 裡被辨識為 misc 的（ProductCode / Name / Barcode 規則）
    def _is_misc_row(row: pd.Series) -> bool:
        pc = str(row.get("ProductCode", "")).strip().split(".")[0]
        nm = str(row.get("Name", ""))
        bc = str(row.get("Barcode", "")).strip().split(".")[0]
        return (
            pc in MISC_PRODUCT_CODES
            or any(kw in nm for kw in MISC_NAME_KEYWORDS)
            or bc in MISC_BARCODE_VALUES
        )

    misc_raw_n = stock_raw.apply(_is_misc_row, axis=1).sum()
    misc_anon_n = (stock_anon["Barcode"].astype(str) == MISC_BARCODE_PLACEHOLDER).sum()

    # 加上 ghost GID（只在 sales、不在 stock）
    stock_raw_ids = set(stock_raw["GoodsID"].dropna().astype(int))
    sales_raw_ids = set(sales_raw["GoodsID"].dropna().astype(int))
    ghost_raw_n = len(sales_raw_ids - stock_raw_ids)

    stock_anon_ids = set(stock_anon["GoodsID"].dropna().astype(int))
    sales_anon_ids = set(sales_anon["GoodsID"].dropna().astype(int))
    ghost_anon_n = len(sales_anon_ids - stock_anon_ids)

    lines.append(
        f"Misc in stock   : {misc_raw_n:>6} → {misc_anon_n}"
        f"  (barcode placeholder)"
    )
    lines.append(
        f"Ghost GID       : {ghost_raw_n:>6} → {ghost_anon_n}"
        f"  (sales only, no stock master)"
    )

    for col in ("LastInCost", "RetailPrice"):
        if col in stock_raw.columns and col in stock_anon.columns:
            lines.append(
                f"{col:<16}: mean {stock_raw[col].mean():.2f} → {stock_anon[col].mean():.2f}"
                f"  std {stock_raw[col].std():.2f} → {stock_anon[col].std():.2f}"
            )

    if "rDate" in sales_anon.columns:
        lines.append(
            f"Date range      : {sales_anon['rDate'].min()} → {sales_anon['rDate'].max()}"
        )

    lines.append("=" * 40)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. 寫出
# ---------------------------------------------------------------------------


def write_outputs(
    stock: pd.DataFrame, sales: pd.DataFrame, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stock.to_csv(output_dir / "stock.csv", index=False, encoding="utf-8-sig")
    sales.to_parquet(output_dir / "sales.parquet", index=False)
    sales.to_csv(output_dir / "sales.csv", index=False, encoding="utf-8-sig")
    logger.info("已寫出：%s", output_dir)


# ---------------------------------------------------------------------------
# 10. Main orchestrator
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)

    logger.info("讀取真實資料（%s）...", args.input_dir)
    df_stock_raw = load_stock(args.input_dir)
    df_sales_raw = load_sales(args.input_dir)

    logger.info("辨識 misc / generic 商品...")
    misc_gids = detect_misc(df_stock_raw, df_sales_raw)

    logger.info("分層抽樣 %d SKU...", args.sku_count)
    selected_gids = sample_skus(
        df_stock_raw,
        df_sales_raw,
        n=args.sku_count,
        seed=args.seed,
        misc_gids=misc_gids,
    )

    df_stock = df_stock_raw[df_stock_raw["GoodsID"].isin(selected_gids)].copy()
    df_sales = df_sales_raw[df_sales_raw["GoodsID"].isin(selected_gids)].copy()

    logger.info("建立 master mapping（seed=%d）...", args.seed)
    maps = build_maps(df_stock, df_sales, misc_gids=misc_gids & selected_gids, seed=args.seed)

    logger.info("脫敏轉換...")
    df_stock_anon = anonymize_stock(df_stock, maps)
    df_sales_anon = anonymize_sales(df_sales, maps, time_shift_days=args.time_shift_days)
    df_sales_anon = recompute_total_amt(df_sales_anon, df_stock_anon)

    report = run_sanity_checks(df_stock_raw, df_sales_raw, df_stock_anon, df_sales_anon)
    print(report)

    if args.dry_run:
        logger.info("--dry-run 模式，不寫出檔案。")
    else:
        logger.info("寫出至 %s ...", args.output_dir)
        write_outputs(df_stock_anon, df_sales_anon, args.output_dir)
        logger.info("完成。")


if __name__ == "__main__":
    main()
    
