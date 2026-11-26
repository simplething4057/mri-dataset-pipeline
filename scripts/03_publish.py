"""
03_publish.py — 학습 데이터셋 버전 발행 (manifest.json + split CSV).

발행 순서:
  1. visit_series_links × preprocessed_outputs 조인으로 labeled sample 목록 구성
  2. subject 단위 class-stratified split (70/15/15, seed=42)
  3. dataset_versions / dataset_samples 테이블 기록
  4. datasets/{version}/ 에 manifest.json + train/val/test.csv 저장

사용법:
    python scripts/03_publish.py --version v1
    python scripts/03_publish.py --version v2
"""

import argparse
import csv as _csv
import hashlib
import json
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "db" / "registry.db"

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42
PIPELINE_VERSION_FOR = {"v1": "1.0", "v2": "1.1"}


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_pipeline_commit() -> str:
    """현재 git 커밋 해시 반환. git 미설치·미초기화 시 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


# ── Split ─────────────────────────────────────────────────────────────────────

def stratified_subject_split(
    subjects: list[str],
    subject_to_label: dict[str, str],
    seed: int,
) -> dict[str, str]:
    """subject → split 딕셔너리 반환.

    알고리즘: 클래스별 독립 split
      각 클래스 그룹 내에서 개별적으로 70/15/15 배정.
      n≥3: val·test 각 최소 1명 보장 → 소수 클래스(Dementia 등)가
           val/test에서 완전히 빠지는 현상 방지.
      n=2: train 1, test 1.
      n=1: train.
    """
    rng = np.random.default_rng(seed)

    label_to_subjects: dict[str, list[str]] = {}
    for subj in sorted(subjects):
        lbl = subject_to_label.get(subj, "UNKNOWN")
        label_to_subjects.setdefault(lbl, []).append(subj)

    assignment: dict[str, str] = {}

    for lbl in sorted(label_to_subjects.keys()):
        arr = np.array(label_to_subjects[lbl])
        shuffled = arr[rng.permutation(len(arr))].tolist()
        n = len(shuffled)

        if n == 1:
            assignment[shuffled[0]] = "train"
        elif n == 2:
            assignment[shuffled[0]] = "train"
            assignment[shuffled[1]] = "test"
        else:
            n_val  = max(1, round(n * SPLIT_RATIOS["val"]))
            n_test = max(1, round(n * SPLIT_RATIOS["test"]))
            # val+test 합이 전체를 초과하지 않도록 보정
            if n_val + n_test >= n:
                n_val, n_test = 1, 1
            n_train = n - n_val - n_test

            for i, subj in enumerate(shuffled):
                if i < n_train:
                    assignment[subj] = "train"
                elif i < n_train + n_val:
                    assignment[subj] = "val"
                else:
                    assignment[subj] = "test"

    return assignment


# ── Publish ───────────────────────────────────────────────────────────────────

def collect_samples(conn: sqlite3.Connection, pipeline_version: str) -> list[dict]:
    """visit_series_links와 preprocessed_outputs를 조인해 샘플 목록 구성."""
    rows = conn.execute(
        """SELECT
               s.series_uid,
               s.image_id,
               s.subject_id,
               st.study_date,
               s.image_type,
               s.modality,
               po.id              AS preprocessed_id,
               po.pipeline_version,
               po.code_commit,
               po.output_path,
               po.sha256,
               h.diagnosis,
               h.viscode2         AS diagnosis_viscode,
               h.exam_date        AS diagnosis_exam_date,
               h.update_stamp     AS diagnosis_stamp
           FROM visit_series_links vsl
           JOIN series              s  ON s.series_uid  = vsl.series_uid
           JOIN studies             st ON st.study_uid  = s.study_uid
           JOIN clinical_visit_history h ON h.id        = vsl.visit_history_id
           JOIN preprocessed_outputs po ON po.series_uid = s.series_uid
                                       AND po.pipeline_version = ?
                                       AND po.output_role      = 'image'
           ORDER BY s.subject_id, st.study_date""",
        (pipeline_version,),
    ).fetchall()
    return [dict(r) for r in rows]


def publish(version: str) -> None:
    if version not in PIPELINE_VERSION_FOR:
        raise ValueError(f"알 수 없는 버전: {version}. PIPELINE_VERSION_FOR에 추가하세요.")

    pipeline_ver = PIPELINE_VERSION_FOR[version]
    out_dir = ROOT / "datasets" / version
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = get_connection()

    if conn.execute(
        "SELECT 1 FROM dataset_versions WHERE version=?", (version,)
    ).fetchone():
        print(f"[publish] {version} 이미 발행됨. 덮어쓰려면 DB에서 해당 버전을 먼저 삭제하세요.")
        conn.close()
        return

    samples = collect_samples(conn, pipeline_ver)
    if not samples:
        print(f"[publish] 샘플 없음 (pipeline_version={pipeline_ver} 미등록?)")
        conn.close()
        return

    subjects = list({s["subject_id"] for s in samples})
    first_diag: dict[str, str] = {}
    for s in sorted(samples, key=lambda x: x["study_date"]):
        first_diag.setdefault(s["subject_id"], s["diagnosis"])

    split_map = stratified_subject_split(subjects, first_diag, SPLIT_SEED)

    created_at = now_utc()
    pipeline_commit = get_pipeline_commit()

    sample_rows = [{**s, "split": split_map[s["subject_id"]]} for s in samples]

    per_split = {sp: sum(1 for r in sample_rows if r["split"] == sp)
                 for sp in ("train", "val", "test")}
    per_class: dict[str, int] = {}
    for r in sample_rows:
        per_class[r["diagnosis"]] = per_class.get(r["diagnosis"], 0) + 1

    clinical_snapshot = conn.execute(
        """SELECT h.subject_id, h.viscode2, h.diagnosis, h.update_stamp
           FROM clinical_visit_history h
           INNER JOIN (
               SELECT subject_id, viscode2, MAX(update_stamp) AS ms
               FROM clinical_visit_history GROUP BY subject_id, viscode2
           ) latest ON h.subject_id=latest.subject_id
             AND h.viscode2=latest.viscode2 AND h.update_stamp=latest.ms
           ORDER BY h.subject_id, h.viscode2"""
    ).fetchall()
    clinical_str = json.dumps([dict(r) for r in clinical_snapshot], sort_keys=True)
    clinical_hash = hashlib.sha256(clinical_str.encode()).hexdigest()

    manifest = {
        "version": version,
        "published_at": created_at,
        "pipeline_version": pipeline_ver,
        "pipeline_commit": pipeline_commit,
        "link_rule": "nearest_within_30d",
        "split_seed": SPLIT_SEED,
        "split_ratios": SPLIT_RATIOS,
        "total_samples": len(sample_rows),
        "per_split_counts": per_split,
        "per_class_counts": per_class,
        "clinical_snapshot_hash": clinical_hash,
        "samples": sample_rows,
    }

    # 임시 디렉터리에 파일 모두 작성 후 out_dir로 이동 (partial write 방지)
    # 실패 시 finally에서 tmp 정리, DB는 commit 전이므로 롤백됨
    tmp_dir = Path(tempfile.mkdtemp(dir=out_dir.parent, prefix=f".tmp_{version}_"))
    try:
        with open(tmp_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        for split_name in ("train", "val", "test"):
            rows = [r for r in sample_rows if r["split"] == split_name]
            if rows:
                with open(tmp_dir / f"{split_name}.csv", "w", newline="", encoding="utf-8") as f:
                    writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

        # 파일 쓰기 모두 성공 → out_dir로 이동
        for src in tmp_dir.iterdir():
            dest = out_dir / src.name
            if dest.exists():
                dest.unlink()
            shutil.move(str(src), str(dest))

        # 파일 이동 성공 → DB 커밋
        conn.execute(
            "INSERT INTO dataset_versions (version, created_at, manifest_path) VALUES (?, ?, ?)",
            (version, created_at, str(out_dir / "manifest.json")),
        )
        for s in sample_rows:
            conn.execute(
                """INSERT INTO dataset_samples
                   (dataset_version, series_uid, preprocessed_id,
                    diagnosis, diagnosis_viscode, diagnosis_exam_date,
                    diagnosis_stamp, split)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    version,
                    s["series_uid"],
                    s["preprocessed_id"],
                    s["diagnosis"],
                    s["diagnosis_viscode"],
                    s["diagnosis_exam_date"],
                    s["diagnosis_stamp"],
                    s["split"],
                ),
            )
        conn.commit()

    except Exception:
        conn.rollback()
        conn.close()
        raise
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    conn.close()
    print(f"[publish] {version} 발행 완료")
    print(f"  pipeline_commit : {pipeline_commit}")
    print(f"  샘플 수: {len(sample_rows)}  |  split: {per_split}  |  class: {per_class}")
    print(f"  → {out_dir}")


def main(version: str) -> None:
    publish(version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="발행할 버전 (예: v1)")
    args = parser.parse_args()
    main(version=args.version)
