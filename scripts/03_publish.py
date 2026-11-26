"""
03_publish.py — 학습 데이터셋 버전 발행 (manifest.json + split CSV).

발행 순서:
  1. visit_series_links × preprocessed_outputs 조인으로 labeled sample 목록 구성
  2. subject 단위 stratified split (70/15/15, seed=42)
  3. dataset_versions / dataset_samples 테이블 기록
  4. datasets/{version}/ 에 manifest.json + train/val/test.csv 저장

사용법:
    python scripts/03_publish.py --version v1
    python scripts/03_publish.py --version v2
"""

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "db" / "registry.db"

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42
PIPELINE_VERSION_FOR = {"v1": "1.0", "v2": "1.1"}  # 버전별 사용 파이프라인


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Split ─────────────────────────────────────────────────────────────────────

def stratified_subject_split(
    subjects: list[str],
    subject_to_label: dict[str, str],
    seed: int,
) -> dict[str, str]:
    """subject → split 딕셔너리 반환.

    알고리즘:
      1. 레이블별로 그룹화 후 각 그룹을 seed로 셔플
      2. 라운드로빈 인터리빙 → class-balanced 순서 형성
      3. 전체 N 기준으로 70/15/15 경계 계산
         - 소규모 데이터셋에서도 test가 최소 1명 이상이 되도록 보정
    """
    rng = np.random.default_rng(seed)

    label_to_subjects: dict[str, list[str]] = {}
    for subj in sorted(subjects):  # 정렬 → 재현성
        lbl = subject_to_label.get(subj, "UNKNOWN")
        label_to_subjects.setdefault(lbl, []).append(subj)

    # 각 레이블 그룹 내 셔플 (공유 rng → 클래스 간 순서도 seed에 종속)
    for lbl in sorted(label_to_subjects.keys()):
        arr = np.array(label_to_subjects[lbl])
        label_to_subjects[lbl] = arr[rng.permutation(len(arr))].tolist()

    # 라운드로빈 인터리빙
    labels = sorted(label_to_subjects.keys())
    ptrs = {lbl: 0 for lbl in labels}
    interleaved: list[str] = []
    while len(interleaved) < len(subjects):
        for lbl in labels:
            if ptrs[lbl] < len(label_to_subjects[lbl]):
                interleaved.append(label_to_subjects[lbl][ptrs[lbl]])
                ptrs[lbl] += 1

    n = len(interleaved)
    n_train = max(1, round(n * SPLIT_RATIOS["train"]))
    n_val   = max(0, round(n * SPLIT_RATIOS["val"]))
    # n ≥ 3이면 test에 최소 1명 확보
    if n >= 3:
        while n_train + n_val >= n and n_val > 0:
            n_val -= 1

    assignment: dict[str, str] = {}
    for i, subj in enumerate(interleaved):
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

    # 이미 발행된 버전이면 중단
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

    # subject 단위 split
    subjects = list({s["subject_id"] for s in samples})
    # 첫 번째 visit 진단으로 계층화
    first_diag: dict[str, str] = {}
    for s in sorted(samples, key=lambda x: x["study_date"]):
        first_diag.setdefault(s["subject_id"], s["diagnosis"])

    split_map = stratified_subject_split(subjects, first_diag, SPLIT_SEED)

    created_at = now_utc()
    manifest_path = str(out_dir / "manifest.json")

    sample_rows = []
    for s in samples:
        sample_rows.append({**s, "split": split_map[s["subject_id"]]})

    per_split = {sp: sum(1 for r in sample_rows if r["split"] == sp)
                 for sp in ("train", "val", "test")}
    per_class: dict[str, int] = {}
    for r in sample_rows:
        per_class[r["diagnosis"]] = per_class.get(r["diagnosis"], 0) + 1

    # clinical 스냅샷 해시: 발행 시점의 유효 진단 전체를 해시
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
        "link_rule": "nearest_within_30d",
        "split_seed": SPLIT_SEED,
        "split_ratios": SPLIT_RATIOS,
        "total_samples": len(sample_rows),
        "per_split_counts": per_split,
        "per_class_counts": per_class,
        "clinical_snapshot_hash": clinical_hash,
        "samples": sample_rows,
    }

    # manifest + CSV 먼저 파일로 저장, 성공 후에만 DB 커밋
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    import csv as _csv
    for split_name in ("train", "val", "test"):
        rows = [r for r in sample_rows if r["split"] == split_name]
        if rows:
            with open(out_dir / f"{split_name}.csv", "w", newline="", encoding="utf-8") as f:
                writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    conn.execute(
        "INSERT INTO dataset_versions (version, created_at, manifest_path) VALUES (?, ?, ?)",
        (version, created_at, manifest_path),
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
    conn.close()
    print(f"[publish] {version} 발행 완료")
    print(f"  샘플 수: {len(sample_rows)}  |  split: {per_split}  |  class: {per_class}")
    print(f"  → {out_dir}")


def main(version: str) -> None:
    publish(version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="발행할 버전 (예: v1)")
    args = parser.parse_args()
    main(version=args.version)
