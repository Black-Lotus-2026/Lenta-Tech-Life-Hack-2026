#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO_IGNORE = shutil.ignore_patterns(
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "outputs",
    "runs",
    "datasets",
    "data",
    "kaggle_work*",
    "*.pyc",
)


def read_kaggle_username(kaggle_json: Path) -> str:
    data = json.loads(kaggle_json.read_text(encoding="utf-8"))
    username = str(data.get("username") or "").strip()
    if not username:
        raise ValueError(f"No username in {kaggle_json}")
    return username


def copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=REPO_IGNORE)
    # Keep all compact local assets required by evaluation-only kernels:
    # whole-tag/field-zone YOLO weights, optional ensemble weights, WeChat QR
    # models and local product catalogs. Do this explicitly so
    # the public video data can stay outside the repo bundle while inference
    # assets remain reproducible.
    for rel in ["models", "data/catalogs"]:
        src_assets = src / rel
        if not src_assets.exists():
            continue
        dst_assets = dst / rel
        if dst_assets.exists():
            shutil.rmtree(dst_assets)
        shutil.copytree(src_assets, dst_assets, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Kaggle dataset and GPU kernel upload folders")
    parser.add_argument("--repo-dir", default=".", help="Local solution repository")
    parser.add_argument("--data-dir", required=True, help="Directory containing the extracted Данные folder")
    parser.add_argument("--kaggle-json", default=str(Path.home() / ".kaggle" / "kaggle.json"))
    parser.add_argument("--work-dir", default="kaggle_work")
    parser.add_argument("--dataset-slug", default="lenta-shelf-ai-bundle")
    parser.add_argument("--kernel-slug", default="lenta-shelf-ai-gpu-experiment")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    kaggle_json = Path(args.kaggle_json).resolve()
    work_dir = Path(args.work_dir).resolve()
    username = read_kaggle_username(kaggle_json)

    dataset_dir = work_dir / "dataset"
    kernel_dir = work_dir / "kernel"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    dataset_dir.mkdir(parents=True)
    kernel_dir.mkdir(parents=True)

    copytree_clean(repo_dir, dataset_dir / "repo")
    data_dst = dataset_dir / "data" / data_dir.name
    shutil.copytree(data_dir, data_dst)

    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Lenta Shelf AI Bundle",
                "id": f"{username}/{args.dataset_slug}",
                "licenses": [{"name": "CC0-1.0"}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    shutil.copy2(repo_dir / "kaggle" / "gpu_experiment.py", kernel_dir / "gpu_experiment.py")
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{username}/{args.kernel_slug}",
                "title": "Lenta Shelf AI GPU Experiment",
                "code_file": "gpu_experiment.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [f"{username}/{args.dataset_slug}"],
                "competition_sources": [],
                "kernel_sources": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[DONE] dataset_dir={dataset_dir}")
    print(f"[DONE] kernel_dir={kernel_dir}")
    print(f"[NEXT] kaggle datasets create -p {dataset_dir} -r zip")
    print(f"[NEXT] kaggle kernels push -p {kernel_dir} --accelerator gpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
