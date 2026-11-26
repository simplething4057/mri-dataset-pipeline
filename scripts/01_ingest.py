"""
01_ingest.py — series/, derived/, clinical/diagnosis.csv 를 SQLite에 적재.

사용법:
    python scripts/01_ingest.py                # 초기 적재
    python scripts/01_ingest.py --changes      # changes/ 반영 (v2용)
"""

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "sample_bundle"
DB_PATH = ROOT / "db" / "registry.db"


# ── 연결 ──────────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


# ── 스키마 ────────────────────────────────────────────────────────────────────

def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS subjects (
            subject_id TEXT PRIMARY KEY,
            rid        INTEGER
        );

        CREATE TABLE IF NOT EXISTS studies (
            study_uid  TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL REFERENCES subjects(subject_id),
            study_date TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS series (
            series_uid          TEXT PRIMARY KEY,
            image_id            TEXT UNIQUE NOT NULL,
            study_uid           TEXT NOT NULL REFERENCES studies(study_uid),
            subject_id          TEXT NOT NULL REFERENCES subjects(subject_id),  -- 비정규화: JOIN 없이 subject 직접 접근
            modality            TEXT NOT NULL,
            image_type          TEXT NOT NULL,
            series_description  TEXT,
            manufacturer        TEXT,
            manufacturer_model  TEXT,
            radiopharmaceutical TEXT,
            source_json_path    TEXT NOT NULL
        );

        -- 전처리 결과 (pipeline 버전별로 coexist)
        CREATE TABLE IF NOT EXISTS preprocessed_outputs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id         TEXT NOT NULL,
            series_uid       TEXT NOT NULL REFERENCES series(series_uid),
            pipeline_name    TEXT NOT NULL,
            pipeline_version TEXT NOT NULL,
            code_commit      TEXT,
            output_role      TEXT NOT NULL,
            output_path      TEXT NOT NULL,
            sha256           TEXT NOT NULL,
            source_json_path TEXT NOT NULL,
            UNIQUE(series_uid, pipeline_version, output_role)
        );

        -- 진단 기록 전체 이력 (append-only)
        CREATE TABLE IF NOT EXISTS clinical_visit_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id   TEXT NOT NULL REFERENCES subjects(subject_id),
            rid          INTEGER,
            viscode      TEXT NOT NULL,
            viscode2     TEXT NOT NULL,
            exam_date    TEXT NOT NULL,
            diagnosis    TEXT NOT NULL,
            update_stamp TEXT NOT NULL,
            ingested_at  TEXT NOT NULL,
            source_file  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS visit_series_links (
            series_uid       TEXT PRIMARY KEY REFERENCES series(series_uid),
            visit_history_id INTEGER NOT NULL REFERENCES clinical_visit_history(id),
            date_diff_days   INTEGER NOT NULL,
            link_rule        TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS dataset_versions (
            version       TEXT PRIMARY KEY,
            created_at    TEXT NOT NULL,
            description   TEXT,
            manifest_path TEXT
        );

        CREATE TABLE IF NOT EXISTS dataset_samples (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_version     TEXT NOT NULL REFERENCES dataset_versions(version),
            series_uid          TEXT NOT NULL REFERENCES series(series_uid),
            preprocessed_id     INTEGER NOT NULL REFERENCES preprocessed_outputs(id),
            diagnosis           TEXT NOT NULL,
            diagnosis_viscode   TEXT NOT NULL,
            diagnosis_exam_date TEXT NOT NULL,
            diagnosis_stamp     TEXT NOT NULL,
            split               TEXT NOT NULL CHECK(split IN ('train', 'val', 'test')),
            UNIQUE(dataset_version, series_uid)
        );

        -- 쿼리 성능: JOIN/WHERE에 자주 쓰이는 컬럼
        CREATE INDEX IF NOT EXISTS idx_series_subject   ON series(subject_id);
        CREATE INDEX IF NOT EXISTS idx_series_study     ON series(study_uid);
        CREATE INDEX IF NOT EXISTS idx_studies_subject  ON studies(subject_id);
        CREATE INDEX IF NOT EXISTS idx_clinical_subject ON clinical_visit_history(subject_id, viscode2);
        CREATE INDEX IF NOT EXISTS idx_clinical_stamp   ON clinical_visit_history(subject_id, viscode2, update_stamp);
        CREATE INDEX IF NOT EXISTS idx_preproc_series   ON preprocessed_outputs(series_uid, pipeline_version, output_role);
        CREATE INDEX IF NOT EXISTS idx_samples_version  ON dataset_samples(dataset_version);
    """)
    conn.commit()


# ── 보조 함수 ─────────────────────────────────────────────────────────────────

def infer_image_type(modality: str, description: str) -> str:
    if modality == "PT":
        return "PET"
    d = description.upper()
    if any(k in d for k in ["MPRAGE", "IR-FSPGR", "SAG T1"]):
        return "T1"
    if any(k in d for k in ["DTI", "DWI", "DIFFUSION"]):
        return "DWI"
    return "UNKNOWN"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── 적재 함수 ─────────────────────────────────────────────────────────────────

def load_series(conn: sqlite3.Connection) -> None:
    series_dir = DATA / "series"
    count = 0

    for path in sorted(series_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            s = json.load(f)

        subject_id = s["PatientID"]
        study_uid  = s["StudyInstanceUID"]
        series_uid = s["SeriesInstanceUID"]
        image_type = infer_image_type(s["Modality"], s.get("SeriesDescription", ""))
        if image_type == "UNKNOWN":
            print(f"  [warn] UNKNOWN image_type: {s['ImageID']} "
                  f"(Modality={s['Modality']}, SeriesDescription={s.get('SeriesDescription')})")

        conn.execute(
            "INSERT OR IGNORE INTO subjects (subject_id) VALUES (?)",
            (subject_id,),
        )
        raw_date = s["StudyDate"]
        # YYYYMMDD → YYYY-MM-DD (ISO 8601)로 정규화
        study_date = (
            f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
            if len(raw_date) == 8 and "-" not in raw_date
            else raw_date
        )
        conn.execute(
            "INSERT OR IGNORE INTO studies (study_uid, subject_id, study_date) VALUES (?, ?, ?)",
            (study_uid, subject_id, study_date),
        )
        conn.execute(
            """INSERT OR IGNORE INTO series
               (series_uid, image_id, study_uid, subject_id,
                modality, image_type, series_description,
                manufacturer, manufacturer_model, radiopharmaceutical, source_json_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                series_uid,
                s["ImageID"],
                study_uid,
                subject_id,
                s["Modality"],
                image_type,
                s.get("SeriesDescription"),
                s.get("Manufacturer"),
                s.get("ManufacturerModelName"),
                s.get("Radiopharmaceutical"),
                str(path),
            ),
        )
        count += 1

    conn.commit()
    print(f"  [series]  {count}개 처리 ({series_dir})")


def load_derived(conn: sqlite3.Connection, derived_dir: Path) -> None:
    count = 0

    for path in sorted(derived_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        series_uid = d["SourceSeriesInstanceUID"]

        # series가 DB에 없으면 적재 불가 (참조 무결성)
        if not conn.execute(
            "SELECT 1 FROM series WHERE series_uid = ?", (series_uid,)
        ).fetchone():
            print(f"  [warn] series 미등록, 건너뜀: {series_uid}")
            continue

        for output in d.get("Outputs", []):
            conn.execute(
                """INSERT OR REPLACE INTO preprocessed_outputs
                   (image_id, series_uid, pipeline_name, pipeline_version,
                    code_commit, output_role, output_path, sha256, source_json_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    d["SourceImageID"],
                    series_uid,
                    d["PipelineName"],
                    d["PipelineVersion"],
                    d.get("CodeCommit"),
                    output["role"],
                    str((derived_dir / output["path"]).relative_to(ROOT)),
                    output["sha256"],
                    str(path),
                ),
            )
            count += 1

    conn.commit()
    print(f"  [derived] {count}개 output 처리 ({derived_dir})")


def load_clinical(conn: sqlite3.Connection, csv_path: Path) -> None:
    now = now_utc()
    inserted = 0

    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            # (subject_id, viscode2, update_stamp) 이 같은 행은 중복이므로 skip
            exists = conn.execute(
                """SELECT 1 FROM clinical_visit_history
                   WHERE subject_id=? AND viscode2=? AND update_stamp=?""",
                (row["PTID"], row["VISCODE2"], row["update_stamp"]),
            ).fetchone()
            if exists:
                continue

            conn.execute(
                """INSERT INTO clinical_visit_history
                   (subject_id, rid, viscode, viscode2, exam_date,
                    diagnosis, update_stamp, ingested_at, source_file)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["PTID"],
                    int(row["RID"]),
                    row["VISCODE"],
                    row["VISCODE2"],
                    row["EXAMDATE"],
                    row["DIAGNOSIS"],
                    row["update_stamp"],
                    now,
                    str(csv_path),
                ),
            )
            inserted += 1

    # subjects 테이블에 rid 보완 (sidecar에 없고 clinical CSV에만 있음)
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            conn.execute(
                "UPDATE subjects SET rid=? WHERE subject_id=? AND rid IS NULL",
                (int(row["RID"]), row["PTID"]),
            )

    conn.commit()
    print(f"  [clinical] {inserted}행 추가 ({csv_path})")


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main(apply_changes: bool) -> None:
    conn = get_connection()

    if not apply_changes:
        print("[ingest] 초기 적재 시작")
        create_schema(conn)
        load_series(conn)
        load_derived(conn, DATA / "derived" / "v1.0")
        load_clinical(conn, DATA / "clinical" / "diagnosis.csv")
    else:
        print("[ingest] changes 적재 시작")
        load_clinical(conn, DATA / "changes" / "diagnosis.csv")
        load_derived(conn, DATA / "changes" / "derived" / "v1.1")

    conn.close()
    print("[ingest] 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--changes",
        action="store_true",
        help="changes/ 디렉터리의 진단·전처리 변경분을 추가 적재",
    )
    args = parser.parse_args()
    main(apply_changes=args.changes)
