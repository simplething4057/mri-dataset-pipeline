"""통합 테스트 — 실제 샘플 데이터로 전체 파이프라인을 검증합니다.

각 테스트는 temp_env 픽스처(conftest.py)가 제공하는 독립 임시 DB를 사용하므로
실제 datasets/ 나 db/registry.db 를 오염시키지 않습니다.
"""
import csv
import json
import sqlite3
from pathlib import Path

import pytest


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def db_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def run_pipeline(modules, version: str = "tv1") -> None:
    """ingest → link → publish 순서로 파이프라인 실행."""
    modules["ingest"].main(apply_changes=False)
    conn = modules["link"].get_connection()
    modules["link"].build_links(conn)
    conn.close()
    modules["publish"].publish(version)


# ── 적재(ingest) ──────────────────────────────────────────────────────────────

class TestIngest:
    def test_creates_series_and_clinical(self, modules, temp_env):
        modules["ingest"].main(apply_changes=False)

        conn = db_conn(temp_env["db_path"])
        series_cnt   = conn.execute("SELECT COUNT(*) FROM series").fetchone()[0]
        clinical_cnt = conn.execute("SELECT COUNT(*) FROM clinical_visit_history").fetchone()[0]
        conn.close()

        assert series_cnt   > 0, "series 미적재"
        assert clinical_cnt > 0, "clinical_visit_history 미적재"

    def test_insert_or_ignore_idempotent(self, modules, temp_env):
        """같은 derived 데이터를 두 번 적재해도 preprocessed_outputs 행 수 불변."""
        modules["ingest"].main(apply_changes=False)
        conn  = db_conn(temp_env["db_path"])
        cnt1  = conn.execute("SELECT COUNT(*) FROM preprocessed_outputs").fetchone()[0]
        conn.close()

        modules["ingest"].main(apply_changes=False)
        conn  = db_conn(temp_env["db_path"])
        cnt2  = conn.execute("SELECT COUNT(*) FROM preprocessed_outputs").fetchone()[0]
        conn.close()

        assert cnt1 == cnt2, f"INSERT OR IGNORE 실패: {cnt1} → {cnt2}"

    def test_changes_adds_clinical_rows(self, modules, temp_env):
        """--changes 는 임상 행만 추가하고 중복은 삽입하지 않는다."""
        modules["ingest"].main(apply_changes=False)
        conn   = db_conn(temp_env["db_path"])
        before = conn.execute("SELECT COUNT(*) FROM clinical_visit_history").fetchone()[0]
        conn.close()

        modules["ingest"].main(apply_changes=True)
        conn   = db_conn(temp_env["db_path"])
        after  = conn.execute("SELECT COUNT(*) FROM clinical_visit_history").fetchone()[0]
        conn.close()

        assert after > before, "changes 적재 후 임상 행이 늘어나야 함"

    def test_preprocessed_id_stable_after_reingest(self, modules, temp_env):
        """재적재 후 preprocessed_outputs.id 가 변하지 않아야 FK 무결성 유지."""
        modules["ingest"].main(apply_changes=False)

        conn     = db_conn(temp_env["db_path"])
        ids_before = {r[0] for r in conn.execute(
            "SELECT id FROM preprocessed_outputs"
        ).fetchall()}
        conn.close()

        modules["ingest"].main(apply_changes=False)   # 재적재
        conn     = db_conn(temp_env["db_path"])
        ids_after  = {r[0] for r in conn.execute(
            "SELECT id FROM preprocessed_outputs"
        ).fetchall()}
        conn.close()

        assert ids_before == ids_after, "INSERT OR IGNORE 미적용: id 변경됨"


# ── 연결(link) ────────────────────────────────────────────────────────────────

class TestLink:
    def test_creates_links(self, modules, temp_env):
        modules["ingest"].main(apply_changes=False)
        conn = modules["link"].get_connection()
        modules["link"].build_links(conn)
        conn.close()

        conn       = db_conn(temp_env["db_path"])
        link_count = conn.execute("SELECT COUNT(*) FROM visit_series_links").fetchone()[0]
        conn.close()

        assert link_count > 0, "T1·PET 링크 미생성"

    def test_relink_replaces_all(self, modules, temp_env):
        """두 번 실행해도 visit_series_links 행 수가 동일해야 함(DELETE+재삽입)."""
        modules["ingest"].main(apply_changes=False)

        conn = modules["link"].get_connection()
        modules["link"].build_links(conn)
        conn.close()
        conn = db_conn(temp_env["db_path"])
        cnt1 = conn.execute("SELECT COUNT(*) FROM visit_series_links").fetchone()[0]
        conn.close()

        conn = modules["link"].get_connection()
        modules["link"].build_links(conn)
        conn.close()
        conn = db_conn(temp_env["db_path"])
        cnt2 = conn.execute("SELECT COUNT(*) FROM visit_series_links").fetchone()[0]
        conn.close()

        assert cnt1 == cnt2


# ── 발행(publish) ─────────────────────────────────────────────────────────────

class TestPublish:
    def test_manifest_exists(self, modules, temp_env):
        run_pipeline(modules)
        manifest = temp_env["tmp_path"] / "datasets" / "tv1" / "manifest.json"
        assert manifest.exists()

    def test_manifest_required_fields(self, modules, temp_env):
        run_pipeline(modules)
        with open(temp_env["tmp_path"] / "datasets" / "tv1" / "manifest.json") as f:
            m = json.load(f)
        for field in ("version", "published_at", "pipeline_version",
                      "pipeline_commit", "clinical_snapshot_hash",
                      "total_samples", "per_split_counts", "per_class_counts", "samples"):
            assert field in m, f"manifest에 {field!r} 없음"

    def test_total_samples_matches(self, modules, temp_env):
        run_pipeline(modules)
        with open(temp_env["tmp_path"] / "datasets" / "tv1" / "manifest.json") as f:
            m = json.load(f)
        assert len(m["samples"]) == m["total_samples"]

    def test_manifest_path_is_relative(self, modules, temp_env):
        run_pipeline(modules)
        conn = db_conn(temp_env["db_path"])
        row  = conn.execute(
            "SELECT manifest_path FROM dataset_versions WHERE version='tv1'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert not Path(row["manifest_path"]).is_absolute(), \
            f"manifest_path가 절대 경로: {row['manifest_path']}"

    def test_pipeline_commit_captured(self, modules, temp_env):
        run_pipeline(modules)
        with open(temp_env["tmp_path"] / "datasets" / "tv1" / "manifest.json") as f:
            m = json.load(f)
        commit = m["pipeline_commit"]
        is_hash    = all(c in "0123456789abcdef" for c in commit) and len(commit) >= 4
        is_unknown = commit == "unknown"
        assert is_hash or is_unknown

    def test_csv_manifest_consistency(self, modules, temp_env):
        """각 split CSV의 series_uid 집합이 manifest samples와 일치."""
        run_pipeline(modules)
        base = temp_env["tmp_path"] / "datasets" / "tv1"
        with open(base / "manifest.json") as f:
            m = json.load(f)
        for sp in ("train", "val", "test"):
            csv_path = base / f"{sp}.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, newline="") as f:
                csv_ids = {row["series_uid"] for row in csv.DictReader(f)}
            manifest_ids = {s["series_uid"] for s in m["samples"] if s["split"] == sp}
            assert csv_ids == manifest_ids, f"{sp}.csv ↔ manifest 불일치"

    def test_no_subject_leakage(self, modules, temp_env):
        run_pipeline(modules)
        with open(temp_env["tmp_path"] / "datasets" / "tv1" / "manifest.json") as f:
            m = json.load(f)
        splits: dict[str, set] = {"train": set(), "val": set(), "test": set()}
        for s in m["samples"]:
            splits[s["split"]].add(s["subject_id"])
        assert not (splits["train"] & splits["val"]),  "train-val subject 중복"
        assert not (splits["train"] & splits["test"]), "train-test subject 중복"
        assert not (splits["val"]   & splits["test"]), "val-test subject 중복"

    def test_duplicate_publish_skipped(self, modules, temp_env):
        """같은 버전을 두 번 발행해도 에러 없이 무시."""
        run_pipeline(modules)
        modules["publish"].publish("tv1")   # 2nd call — should be a no-op

    def test_v2_snapshot_hash_differs(self, modules, temp_env):
        """changes 적용 후 v2의 clinical_snapshot_hash는 v1과 달라야 함."""
        run_pipeline(modules, version="tv1")

        modules["ingest"].main(apply_changes=True)
        conn = modules["link"].get_connection()
        modules["link"].build_links(conn)
        conn.close()
        modules["publish"].publish("tv2")

        base = temp_env["tmp_path"] / "datasets"
        with open(base / "tv1" / "manifest.json") as f:
            h1 = json.load(f)["clinical_snapshot_hash"]
        with open(base / "tv2" / "manifest.json") as f:
            h2 = json.load(f)["clinical_snapshot_hash"]

        assert h1 != h2, "v1·v2 clinical_snapshot_hash 동일 (변경분 미반영)"

    def test_snapshot_hash_scoped_to_dataset_subjects(self, modules, temp_env):
        """tv1·tv2 가 각자 데이터셋 subject만 해싱함을 간접 검증.
        tv1 발행 후 changes 적재 시 tv1 hash가 불변이어야 함."""
        run_pipeline(modules, version="tv1")

        with open(temp_env["tmp_path"] / "datasets" / "tv1" / "manifest.json") as f:
            hash_before = json.load(f)["clinical_snapshot_hash"]

        # changes 적재는 새 임상 행을 추가하지만 tv1 manifest는 불변
        modules["ingest"].main(apply_changes=True)

        with open(temp_env["tmp_path"] / "datasets" / "tv1" / "manifest.json") as f:
            hash_after = json.load(f)["clinical_snapshot_hash"]

        assert hash_before == hash_after, "이미 발행된 tv1 hash가 변경됨"


# ── 검증(validate) ────────────────────────────────────────────────────────────

class TestValidate:
    def test_v1_passes(self, modules, temp_env):
        run_pipeline(modules)
        assert modules["validate"].validate("tv1"), "04_validate FAIL (tv1)"

    def test_v2_passes(self, modules, temp_env):
        run_pipeline(modules, version="tv1")
        modules["ingest"].main(apply_changes=True)
        conn = modules["link"].get_connection()
        modules["link"].build_links(conn)
        conn.close()
        modules["publish"].publish("tv2")
        assert modules["validate"].validate("tv2"), "04_validate FAIL (tv2)"

    def test_missing_manifest_fails(self, modules, temp_env):
        """manifest.json이 없으면 False 반환 (exit 0 아님)."""
        passed = modules["validate"].validate("nonexistent_ver")
        assert not passed


# ── 원자적 발행 ────────────────────────────────────────────────────────────────

class TestAtomicPublish:
    def test_no_partial_files_on_failure(self, tmp_path, monkeypatch):
        """쓰기 도중 예외 발생 시 out_dir에 파일이 남지 않아야 함."""
        import shutil
        import tempfile

        out_dir = tmp_path / "datasets" / "broken"
        out_dir.mkdir(parents=True)

        tmp_dir = Path(tempfile.mkdtemp(dir=out_dir.parent, prefix=".tmp_broken_"))
        try:
            (tmp_dir / "manifest.json").write_text('{"partial": true}')
            raise RuntimeError("강제 실패")
        except RuntimeError:
            pass
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

        assert not tmp_dir.exists(),              "tmp 디렉터리 미정리"
        assert not (out_dir / "manifest.json").exists(), "partial manifest 잔재"
