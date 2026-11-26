"""
02_link.py — T1·PET series와 임상 방문을 시간적 근접성으로 연결.

규칙 (WINDOW_DAYS=30, LINK_RULE 상수로 제어):
  - |StudyDate − EXAMDATE| ≤ 30일
  - 동점 시: 촬영 이전 방문 우선 (label leakage 방지)
  - 범위 내 방문 없으면 미연결 (labeled dataset에서 제외)

결과: visit_series_links 테이블 갱신 (기존 행 모두 교체)

사용법:
    python scripts/02_link.py
"""

import sqlite3
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "db" / "registry.db"

WINDOW_DAYS = 30
LINK_RULE = "nearest_within_30d"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def parse_date(text: str) -> date:
    """YYYYMMDD 또는 YYYY-MM-DD → date (두 형식 모두 허용)"""
    if len(text) == 8 and "-" not in text:
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    return date.fromisoformat(text)


def build_links(conn: sqlite3.Connection) -> None:
    # 기존 링크 초기화 (재실행 시 최신 진단 기준으로 재계산)
    conn.execute("DELETE FROM visit_series_links")

    # T1·PET 대상 (각각 ±30일 근접 매칭으로 임상 방문에 직접 연결)
    # DWI는 T1과 동일 StudyInstanceUID이므로 T1의 연결을 상속 (별도 링크 불필요)
    anchor_series = conn.execute(
        """SELECT s.series_uid, s.subject_id, s.image_id,
                  s.series_description, s.image_type, st.study_date
           FROM series s
           JOIN studies st ON st.study_uid = s.study_uid
           WHERE s.image_type IN ('T1', 'PET')"""
    ).fetchall()

    linked = 0
    unlinked = 0

    for sr in anchor_series:
        study_date = parse_date(sr["study_date"])
        subject_id = sr["subject_id"]

        # 해당 subject의 모든 임상 방문 (최신 update_stamp 기준 현재 유효 진단)
        visits = conn.execute(
            """SELECT h.id, h.viscode2, h.exam_date, h.diagnosis, h.update_stamp
               FROM clinical_visit_history h
               INNER JOIN (
                   SELECT subject_id, viscode2, MAX(update_stamp) AS max_stamp
                   FROM clinical_visit_history
                   WHERE subject_id = ?
                   GROUP BY subject_id, viscode2
               ) latest
                 ON h.subject_id = latest.subject_id
                AND h.viscode2  = latest.viscode2
                AND h.update_stamp = latest.max_stamp""",
            (subject_id,),
        ).fetchall()

        best_visit_id = None
        best_diff = None
        best_is_prior = None  # True = 촬영 이전 방문

        for v in visits:
            exam_date = parse_date(v["exam_date"])
            diff = abs((study_date - exam_date).days)

            if diff > WINDOW_DAYS:
                continue

            is_prior = exam_date <= study_date

            # 선택 기준:
            #   1) 차이가 더 작은 것 우선
            #   2) 동점이면 촬영 이전 방문 우선
            if best_visit_id is None:
                better = True
            elif diff < best_diff:
                better = True
            elif diff == best_diff and is_prior and not best_is_prior:
                better = True
            else:
                better = False

            if better:
                best_visit_id = v["id"]
                best_diff = diff
                best_is_prior = is_prior

        if best_visit_id is not None:
            conn.execute(
                """INSERT OR REPLACE INTO visit_series_links
                   (series_uid, visit_history_id, date_diff_days, link_rule)
                   VALUES (?, ?, ?, ?)""",
                (sr["series_uid"], best_visit_id, best_diff, LINK_RULE),
            )
            linked += 1
        else:
            unlinked += 1

    conn.commit()
    print(f"[link] T1·PET 연결 완료: {linked}개 연결, {unlinked}개 미연결 (±{WINDOW_DAYS}일 범위 초과)")


def main() -> None:
    conn = get_connection()
    build_links(conn)
    conn.close()
    print("[link] 완료")


if __name__ == "__main__":
    main()
