"""
04_validate.py — 발행된 데이터셋 검증.

통과 → exit 0  /  실패 → exit 1

검증 항목:
  1. manifest.json 존재
  2. 필수 필드 완결성
  3. 허용 진단값 (CN / MCI / Dementia)
  4. split 간 subject 중복 없음 (leakage 검사)
  5. split 비율 허용 오차 ±5%
  6. manifest samples ↔ split CSV 일치
  7. sha256 무결성 (파일 존재 시)

사용법:
    python scripts/04_validate.py --version v1
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

ALLOWED_DIAGNOSES = {"CN", "MCI", "Dementia"}
SPLIT_RATIO_TOLERANCE = 0.05
EXPECTED_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
REQUIRED_SAMPLE_FIELDS = {
    "series_uid", "image_id", "subject_id", "study_date",
    "image_type", "output_path", "sha256",
    "diagnosis", "diagnosis_viscode", "diagnosis_exam_date",
    "diagnosis_stamp", "split",
}


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}", file=sys.stderr)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def validate(version: str) -> bool:
    out_dir = ROOT / "datasets" / version
    manifest_path = out_dir / "manifest.json"
    errors = 0

    # ── 1. manifest 존재 ────────────────────────────────────────────────────
    if not manifest_path.exists():
        fail(f"manifest.json 없음: {manifest_path}")
        return False
    ok("manifest.json 존재")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    samples: list[dict] = manifest.get("samples", [])

    # ── 2. 필수 필드 ────────────────────────────────────────────────────────
    missing_fields = set()
    for s in samples:
        missing_fields |= REQUIRED_SAMPLE_FIELDS - s.keys()
    if missing_fields:
        fail(f"샘플에 필수 필드 누락: {missing_fields}")
        errors += 1
    else:
        ok(f"필수 필드 완결 ({len(REQUIRED_SAMPLE_FIELDS)}개)")

    # ── 3. 허용 진단값 ──────────────────────────────────────────────────────
    bad_diag = {s["diagnosis"] for s in samples if s["diagnosis"] not in ALLOWED_DIAGNOSES}
    if bad_diag:
        fail(f"허용되지 않는 진단값: {bad_diag}")
        errors += 1
    else:
        ok(f"진단값 모두 허용 집합 내 (CN/MCI/Dementia)")

    # ── 4. split 간 subject 중복 (leakage) ─────────────────────────────────
    split_subjects: dict[str, set] = {"train": set(), "val": set(), "test": set()}
    for s in samples:
        split_subjects[s["split"]].add(s["subject_id"])

    train_val  = split_subjects["train"] & split_subjects["val"]
    train_test = split_subjects["train"] & split_subjects["test"]
    val_test   = split_subjects["val"]   & split_subjects["test"]

    if train_val or train_test or val_test:
        if train_val:
            fail(f"train-val subject 중복: {train_val}")
        if train_test:
            fail(f"train-test subject 중복: {train_test}")
        if val_test:
            fail(f"val-test subject 중복: {val_test}")
        errors += 1
    else:
        ok("split 간 subject 중복 없음 (leakage 없음)")

    # ── 5. split 비율 (subject 단위) ─────────────────────────────────────────
    # split이 subject 단위로 적용되므로 비율 검증도 subject 단위로 수행.
    # sample 수 비율은 subject별 방문 수 차이로 인해 target과 달라질 수 있음.
    subj_per_split: dict[str, set] = {"train": set(), "val": set(), "test": set()}
    for s in samples:
        subj_per_split[s["split"]].add(s["subject_id"])
    n_subjects = sum(len(v) for v in subj_per_split.values())

    if n_subjects > 0:
        for sp, expected in EXPECTED_RATIOS.items():
            n_sp = len(subj_per_split[sp])
            actual = n_sp / n_subjects
            # 부동소수점 오차를 허용하기 위해 epsilon 추가
            if abs(actual - expected) > SPLIT_RATIO_TOLERANCE + 1e-9:
                fail(f"{sp} subject 비율 오차 초과: {n_sp}/{n_subjects}={actual:.2f} vs 기대 {expected:.2f} (허용 ±{SPLIT_RATIO_TOLERANCE})")
                errors += 1
            else:
                ok(f"{sp} subject 비율 {actual:.2f} ({n_sp}/{n_subjects}명, 기대 {expected:.2f})")

    # ── 6. manifest samples ↔ split CSV 일치 ───────────────────────────────
    for sp in ("train", "val", "test"):
        csv_path = out_dir / f"{sp}.csv"
        if not csv_path.exists():
            fail(f"{sp}.csv 없음")
            errors += 1
            continue

        import csv
        with open(csv_path, encoding="utf-8", newline="") as f:
            csv_ids = {row["series_uid"] for row in csv.DictReader(f)}
        manifest_ids = {s["series_uid"] for s in samples if s["split"] == sp}

        if csv_ids != manifest_ids:
            fail(f"{sp}.csv와 manifest 불일치: 차이 {csv_ids ^ manifest_ids}")
            errors += 1
        else:
            ok(f"{sp}.csv ↔ manifest 일치 ({len(csv_ids)}개)")

    # ── 7. sha256 무결성 (파일 존재 시) ────────────────────────────────────
    checked = 0
    corrupted = 0
    for s in samples:
        path = ROOT / s["output_path"]
        if not path.exists():
            continue  # 합성 데이터: 파일 없으면 hash 검증 스킵
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != s["sha256"]:
            fail(f"sha256 불일치: {path.name}")
            corrupted += 1
        checked += 1

    if corrupted:
        errors += 1
    else:
        if checked > 0:
            ok(f"sha256 무결성 확인 ({checked}개 파일)")
        else:
            ok("sha256: 실제 파일 없음, 메타데이터 완결성만 확인")

    # sha256 필드 자체가 비어 있는 샘플 확인
    empty_hash = [s["series_uid"] for s in samples if not s.get("sha256")]
    if empty_hash:
        fail(f"sha256 필드 없는 샘플: {empty_hash}")
        errors += 1

    # ── 결과 ────────────────────────────────────────────────────────────────
    return errors == 0


def main(version: str) -> None:
    print(f"[validate] {version} 검증 시작")
    passed = validate(version)

    if passed:
        print(f"[validate] PASS - {version}")
        sys.exit(0)
    else:
        print(f"[validate] FAIL - {version}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="검증할 버전 (예: v1)")
    args = parser.parse_args()
    main(version=args.version)
