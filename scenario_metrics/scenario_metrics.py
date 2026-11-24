import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

VISUAL_PREFIX = "visual_"


def load_metadata(tsv_path: Path) -> dict:
    """photo_id -> {'source_group': str, 'scenarios': [str, ...]}"""
    meta = {}
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = row["photo_id"]
            scenarios = [
                key[len(VISUAL_PREFIX):]
                for key in row
                if key.startswith(VISUAL_PREFIX) and row[key] == "1"
            ]
            meta[pid] = {
                "source_group": row["source_group"],
                "scenarios": scenarios,
            }
    return meta


def discover_configs(results_dir: Path) -> list:
    """Находит все подпапки results_dir, где есть per_image.jsonl (= конфигурация модели)."""
    configs = []
    for child in sorted(results_dir.iterdir()):
        if child.is_dir() and (child / "per_image.jsonl").is_file():
            configs.append(child.name)
    return configs


def load_per_image(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_div(numer, denom):
    return numer / denom if denom else None


def f1(tp, fp, fn):
    return safe_div(2 * tp, 2 * tp + fp + fn)


def accumulate(per_image_rows: list, meta: dict, key_fn, config_name: str) -> dict:
    """
    key_fn(meta_entry) -> список ключей (групп/сценариев), к которым относится фото
    (для сценариев фото может попасть в несколько ключей сразу — так же, как в dataset_summary).
    Возвращает dict: ключ -> агрегированные счётчики.
    """
    agg = defaultdict(lambda: {
        "photos": 0, "n_golden": 0, "n_predicted": 0,
        "tp5": 0, "fp5": 0, "fn5": 0,
        "tp75": 0, "fp75": 0, "fn75": 0,
        "photos_with_fp5": 0,
    })
    missing_meta = 0
    skipped_errors = 0
    for row in per_image_rows:
        pid = row["photo_id"]
        m = meta.get(pid)
        if m is None:
            missing_meta += 1
            continue
        if row.get("error") is not None:
            # техническая ошибка обработки фото — не считаем как "нашёл 0 бабочек",
            # пропускаем из агрегатов, чтобы не искажать метрику (см. statement.md)
            skipped_errors += 1
            continue

        keys = key_fn(m)
        iou5 = row.get("iou_0.5", {})
        iou75 = row.get("iou_0.75", {})
        tp5, fp5, fn5 = iou5.get("tp", 0), iou5.get("fp", 0), iou5.get("fn", 0)
        tp75, fp75, fn75 = iou75.get("tp", 0), iou75.get("fp", 0), iou75.get("fn", 0)

        for k in keys:
            a = agg[k]
            a["photos"] += 1
            a["n_golden"] += row.get("n_golden", 0)
            a["n_predicted"] += row.get("n_predicted", 0)
            a["tp5"] += tp5; a["fp5"] += fp5; a["fn5"] += fn5
            a["tp75"] += tp75; a["fp75"] += fp75; a["fn75"] += fn75
            if fp5 > 0:
                a["photos_with_fp5"] += 1

    if missing_meta:
        print(f"  [{config_name}] предупреждение: {missing_meta} фото не нашлись в dataset_inaturalist.tsv", file=sys.stderr)
    if skipped_errors:
        print(f"  [{config_name}] предупреждение: {skipped_errors} фото с технической ошибкой исключены из агрегатов", file=sys.stderr)

    return agg


def rows_for_output(agg: dict, config: str, dimension: str) -> list:
    out = []
    for key, a in sorted(agg.items()):
        out.append({
            "dimension": dimension,
            "key": key,
            "config": config,
            "photos": a["photos"],
            "n_golden_boxes": a["n_golden"],
            "n_predicted_boxes": a["n_predicted"],
            "tp@0.5": a["tp5"], "fp@0.5": a["fp5"], "fn@0.5": a["fn5"],
            "precision@0.5": safe_div(a["tp5"], a["tp5"] + a["fp5"]),
            "recall@0.5": safe_div(a["tp5"], a["tp5"] + a["fn5"]),
            "f1@0.5": f1(a["tp5"], a["fp5"], a["fn5"]),
            "tp@0.75": a["tp75"], "fp@0.75": a["fp75"], "fn@0.75": a["fn75"],
            "f1@0.75": f1(a["tp75"], a["fp75"], a["fn75"]),
            "photos_with_fp@0.5": a["photos_with_fp5"],
            "fp_rate@0.5": safe_div(a["photos_with_fp5"], a["photos"]),
        })
    return out


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        print(f"  нечего писать в {path} (пустой результат)", file=sys.stderr)
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset-tsv", default="dataset_inaturalist.tsv",
                     help="путь к dataset_inaturalist.tsv (по умолчанию: в текущей папке)")
    ap.add_argument("--results-dir", default="models/results/golden",
                     help="папка с подпапками <config>/per_image.jsonl (по умолчанию: models/results/golden)")
    ap.add_argument("--out-dir", default=None,
                     help="куда писать итоговые CSV (по умолчанию = --results-dir)")
    ap.add_argument("--configs", nargs="*", default=None,
                     help="считать только эти конфигурации (по умолчанию — все найденные в --results-dir)")
    args = ap.parse_args()

    dataset_tsv = Path(args.dataset_tsv)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir

    if not dataset_tsv.is_file():
        sys.exit(f"Не найден {dataset_tsv}. Укажите путь через --dataset-tsv.")
    if not results_dir.is_dir():
        sys.exit(f"Не найдена папка {results_dir}. Укажите путь через --results-dir.")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Читаю метаданные: {dataset_tsv}")
    meta = load_metadata(dataset_tsv)
    print(f"  фото в метаданных: {len(meta)}")

    configs = args.configs or discover_configs(results_dir)
    if not configs:
        sys.exit(f"В {results_dir} не найдено ни одной папки с per_image.jsonl")
    print(f"Конфигурации моделей: {', '.join(configs)}")

    by_group_rows = []
    by_scenario_rows = []

    for cfg in configs:
        per_image_path = results_dir / cfg / "per_image.jsonl"
        if not per_image_path.is_file():
            print(f"  [{cfg}] нет файла {per_image_path}, пропускаю", file=sys.stderr)
            continue

        rows = load_per_image(per_image_path)
        print(f"  [{cfg}] фото в per_image.jsonl: {len(rows)}")

        group_agg = accumulate(rows, meta, key_fn=lambda m: [m["source_group"]], config_name=cfg)
        scenario_agg = accumulate(rows, meta, key_fn=lambda m: m["scenarios"], config_name=cfg)

        by_group_rows += rows_for_output(group_agg, cfg, "source_group")
        by_scenario_rows += rows_for_output(scenario_agg, cfg, "visual_scenario")

    group_out = out_dir / "metrics_by_group.csv"
    scenario_out = out_dir / "metrics_by_scenario.csv"
    write_csv(group_out, by_group_rows)
    write_csv(scenario_out, by_scenario_rows)

    print(f"\nГотово:\n  {group_out}\n  {scenario_out}")


if __name__ == "__main__":
    main()