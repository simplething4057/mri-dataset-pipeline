# MRI Dataset Pipeline

Brain MRI (T1) 기반 알츠하이머 분류 모델 학습용 데이터 엔지니어링 파이프라인.

## 목차

 [1. 실행 방법](#1-실행-방법)<br>
 [2. 파이프라인 개요](#2-파이프라인-개요)<br>
&nbsp;&nbsp;&nbsp;[2.1. 스크립트 처리 흐름](#21-스크립트-처리-흐름)<br>
&nbsp;&nbsp;&nbsp;[2.2. 모달리티별 데이터 흐름](#22-모달리티별-데이터-흐름)<br>
 [3. 설계 결정](#3-설계-결정)<br>
&nbsp;&nbsp;&nbsp;[3.1. 최소 단위: `series`](#31-최소-단위-series)<br>
&nbsp;&nbsp;&nbsp;[3.2. 임상 방문과 영상 종류(T1·DWI·PET)의 관계](#32-임상-방문과-영상-종류t1dwipet의-관계)<br>
&nbsp;&nbsp;&nbsp;[3.3. Train / Val / Test 분할](#33-train--val--test-분할)<br>
&nbsp;&nbsp;&nbsp;[3.4. 기술 스택](#34-기술-스택)<br>
&nbsp;&nbsp;&nbsp;[3.5. 미채택 대안](#35-미채택-대안)<br>
&nbsp;&nbsp;&nbsp;[3.6. 가정한 것](#36-가정한-것)<br>
&nbsp;&nbsp;&nbsp;[3.7. 데이터셋 식별자·재현 보장·학습 파이프라인 통합](#37-데이터셋-식별자재현-보장학습-파이프라인-통합)<br>
 [4. 문제 Q1–Q6](#4-문제-q1q6)<br>
&nbsp;&nbsp;&nbsp;[4.1. Q1. v1과 v2의 차이](#41-q1-v1과-v2의-차이)<br>
&nbsp;&nbsp;&nbsp;[4.2. Q2. v1 파일 목록과 진단 시점](#42-q2-v1-파일-목록과-진단-시점)<br>
&nbsp;&nbsp;&nbsp;[4.3. Q3. 진단 수정 1건이 영향을 주는 데이터셋 버전](#43-q3-진단-수정-1건이-영향을-주는-데이터셋-버전)<br>
&nbsp;&nbsp;&nbsp;[4.4. Q4. 학습·평가 데이터 오염 여부 확인 방법과 결과](#44-q4-학습평가-데이터-오염-여부-확인-방법과-결과)<br>
&nbsp;&nbsp;&nbsp;[4.5. Q5. 특정 영상에 어떤 진단 기록이 왜 연결됐는지](#45-q5-특정-영상에-어떤-진단-기록이-왜-연결됐는지)<br>
&nbsp;&nbsp;&nbsp;[4.6. Q6. S001의 임상 방문마다 대응하는 T1·DWI·PET와 대응 규칙](#46-q6-s001의-임상-방문마다-대응하는-t1dwipet와-대응-규칙)<br>
 [5. 디렉토리 구조](#5-디렉토리-구조)

---

## 1. 실행 방법

```bash
pip install -r requirements.txt

python scripts/01_ingest.py          # DB 적재
python scripts/02_link.py            # 방문-영상 연결
python scripts/03_publish.py --version v1  # v1 발행
python scripts/04_validate.py --version v1 # v1 검증

# v2: changes/ 반영 후 재발행
python scripts/01_ingest.py --changes
python scripts/02_link.py
python scripts/03_publish.py --version v2
python scripts/04_validate.py --version v2
```

---

## 2. 파이프라인 개요

### 2.1. 스크립트 처리 흐름

| 스크립트 | 역할 | 입력 | 출력 (DB 테이블 / 파일) |
|---------|------|------|----------------------|
| `01_ingest.py` | 원시 데이터 적재 | `series/`, `derived/`, `clinical/` | `subjects` · `studies` · `series` · `preprocessed_outputs` · `clinical_visit_history` |
| `02_link.py` | 방문-영상 연결 | DB (series, clinical_visit_history) | `visit_series_links` (T1·PET ±30일 매칭) |
| `03_publish.py` | 데이터셋 발행 | DB (`visit_series_links` ∩ `preprocessed_outputs`) | `dataset_versions` · `dataset_samples` + `manifest.json` + `train/val/test.csv` |
| `04_validate.py` | 무결성 검증 | `datasets/{version}/` | exit 0 (PASS) / exit 1 (FAIL) |

### 2.2. 모달리티별 데이터 흐름

샘플 데이터 기준 (series 36개: T1×15, DWI×15, PET×6):

| 모달리티 | series | preprocessed_outputs | visit_series_links | dataset_samples |
|---------|:------:|:-------------------:|:-----------------:|:---------------:|
| **T1** | ✓ 15개 | ✓ 15개 | ✓ 직접 연결 | ✓ **학습 데이터** |
| DWI | ✓ 15개 | ✗ | ✗ (T1 study_uid 상속) | ✗ |
| PET | ✓ 6개 | ✗ | ✓ 직접 연결 | ✗ (전처리 출력 없음) |

> 현재 파이프라인은 T1-only 학습 데이터셋입니다. DWI·PET는 임상 연결 맥락으로만 활용되며, 전처리 출력이 추가되면 자동으로 `dataset_samples`에 포함됩니다.

---

## 3. 설계 결정

### 3.1. 최소 단위: `series`

모델 입력 = NIfTI 볼륨 1개 = series 1개. Subject·visit은 레이블 단위이고 직접적인 모델 입력이 아니므로 series를 최소 단위로 선택.

### 3.2. 임상 방문과 영상 종류(T1·DWI·PET)의 관계

**T1·PET는 임상 방문에 직접 연결, DWI는 T1의 연결을 상속합니다.**

```
임상 방문 (bl, m06, m12…)
   ├── ±30일 근접 매칭 ──▶ T1 series  (visit_series_links)
   │                           ↕  동일 StudyInstanceUID
   │                         DWI series  ←── T1 연결 상속
   └── ±30일 근접 매칭 ──▶ PET series (visit_series_links, 별도 study)
```

| 모달리티 | 연결 방식 | 근거 |
|---------|-----------|------|
| T1 | `visit_series_links`에 명시적 등록 | 구조적 뇌 영상이 진단의 주요 근거 |
| DWI | 동일 `StudyInstanceUID`로 T1과 묶임 | T1과 같은 MRI 세션 → T1의 임상 방문 맥락을 공유 |
| PET | `visit_series_links`에 명시적 등록 (별도 study) | MRI와 별도 PET 세션으로 촬영 → 직접 ±30일 매칭 필요 |

T1이 미연결인 study의 DWI는 labeled dataset에서 제외됩니다. PET는 T1과 독립적으로 연결되므로 T1 연결 여부와 무관합니다.

**방문-영상 연결 규칙 (T1·PET 공통):**

`|StudyDate − EXAMDATE| ≤ 30일` 범위 내 가장 가까운 임상 방문을 연결.
동점 시 촬영 이전 방문 우선 (사후 방문 사용 시 label leakage 위험).
30일 이내 방문 없는 series는 labeled dataset 미포함.

### 3.3. Train / Val / Test 분할

- **단위**: subject (동일 subject의 모든 series는 같은 split)
- **비율**: 70 / 15 / 15
- **계층화**: 첫 방문 진단 기준 CN/MCI/Dementia 비율 유지
- **시드**: 42 (manifest에 기록 → 완전 재현 가능)

> 일반 ML의 80/10/10과 달리 15%씩 확보하는 이유: subject 단위 분할 시 실질 독립 샘플 수가 적고, Dementia 소수 클래스의 test 대표성 보장 필요.

### 3.4. 기술 스택

| 역할 | 선택 | 이유 |
|------|------|------|
| 메타데이터 저장·쿼리 | SQLite (`db/registry.db`) | 파일 1개, Python 내장, SQL로 복합 쿼리 가능 |
| 버전 불변 스냅샷 | JSON manifest (`datasets/vN/`) | MLOps 엔지니어가 직접 소비, Git 커밋 가능 |

### 3.5. 미채택 대안

- **DVC**: 외부 스토리지 연동 필수, 환경 의존성 큼
- **PostgreSQL**: 서버 프로세스 필요, 과도한 규모
- **순수 JSON 파일**: cross-file 집계 쿼리 불가 (Q3 등)

### 3.6. 가정한 것

- derived JSON의 `Outputs` 중 `role="image"` 파일만 학습 입력으로 사용
- `SeriesDescription` 대소문자 무관 keyword 매칭으로 image_type 판별
- 파일이 존재하지 않는 경우 sha256 검증은 메타데이터 완결성(필드 존재 여부)만 확인
- v2는 v1 발행 이후 changes/ 적재 → 재발행 순서로 진행
- **현재 학습 데이터셋은 T1-only**: `derived/`에 T1 전처리 출력만 존재. PET는 `visit_series_links`에 연결되지만 전처리 출력이 없어 `dataset_samples`에 미포함. DWI도 동일.
- **`dataset_config.json`은 참조 문서**: 스크립트가 직접 읽지 않음. 설계 근거·필드 명세를 기록한 문서 역할.
- **series sidecar는 불변으로 간주**: `01_ingest.py`는 series/studies를 `INSERT OR IGNORE`로 적재하므로 sidecar JSON이 수정되어도 재실행 시 반영되지 않음. `--changes` 모드는 clinical·derived 변경만 처리하며 series 수정은 대상 외.

---

### 3.7. 데이터셋 식별자·재현 보장·학습 파이프라인 통합

#### 식별자 형식

데이터셋 식별자 = **버전 문자열** (예: `v1`, `v2`).  
`python scripts/03_publish.py --version v1`으로 발행하면 `datasets/v1/` 디렉터리가 생성됩니다.

#### 재현 스냅샷

`datasets/{version}/manifest.json` 파일 1개가 재현에 필요한 모든 정보를 포함합니다.

| manifest 필드 | 재현에서의 역할 |
|--------------|---------------|
| `samples[].output_path` | 모델이 읽을 NIfTI 파일 경로 (ROOT 기준 상대경로) |
| `samples[].sha256` | 발행 당시 파일 내용의 지문 → 무결성 검증 |
| `samples[].diagnosis` | 발행 시점에 확정된 레이블 (이후 임상 수정과 무관) |
| `samples[].split` | train / val / test 소속 |
| `split_seed` | 분할 알고리즘 시드 → 완전 재현 가능 |
| `pipeline_version` | 전처리 파이프라인 버전 |
| `clinical_snapshot_hash` | 발행 당시 유효 진단 전체의 SHA-256 → 진단 상태 고정 증명 |

1년 뒤에도 `datasets/v1/manifest.json`만 있으면 `output_path`로 파일을 찾고 `sha256`으로 동일성을 확인해 완전히 같은 학습 입력을 구성할 수 있습니다.

#### 학습 파이프라인 통합 패턴

```bash
# 학습 시작 전 반드시 검증 → exit 0이면 학습, exit 1이면 중단
python scripts/04_validate.py --version v1 && python train.py --dataset v1
```

검증 스크립트는 사람이 출력을 읽지 않아도 종료 코드만으로 통과 여부를 판정합니다.  
**exit 0 = PASS (학습 가능), exit 1 = FAIL (학습 중단)**  
이 설계를 택한 이유: CI/CD나 오케스트레이터가 텍스트 파싱 없이 `&&` 연산자 또는 `if [ $? -eq 0 ]` 조건으로 다음 단계를 제어할 수 있기 때문입니다.

검증 항목:
1. manifest.json 존재
2. 필수 필드 완결성 (12개 필드)
3. 허용 진단값 (CN / MCI / Dementia)
4. split 간 subject 중복 없음 (leakage 없음)
5. split 비율 허용 오차 ±5% (subject 단위)
6. manifest samples ↔ split CSV 일치
7. sha256 무결성 (파일 존재 시 검증, 없으면 메타데이터 완결성만 확인)

> **주의**: 샘플 데이터는 가상의 경로만 포함하므로 항목 7은 메타데이터 완결성만 확인합니다. 실제 NIfTI 파일이 존재하는 환경에서는 sha256을 직접 비교합니다.

---

## 4. 문제 Q1–Q6

> 모든 쿼리는 `db/registry.db` 에 대해 실행합니다.  
> `sqlite3 db/registry.db < query.sql` 또는 DB Browser for SQLite에서 실행 가능.

---

### 4.1. Q1. v1과 v2의 차이

**Q. v1과 v2 데이터셋에서 샘플 목록과 진단이 어떻게 달라지는가?**

```sql
-- (1) v1·v2에서 진단이 달라진 샘플
SELECT
    s.subject_id,
    ds1.diagnosis_viscode     AS viscode,
    ds1.diagnosis             AS v1_diagnosis,
    ds2.diagnosis             AS v2_diagnosis
FROM dataset_samples ds1
-- 같은 series_uid로 v1·v2 행을 self-join → 같은 영상의 두 버전 진단을 한 행에 나란히 배치
JOIN dataset_samples ds2
    ON ds1.series_uid = ds2.series_uid
JOIN series s ON s.series_uid = ds1.series_uid
WHERE ds1.dataset_version = 'v1'
  AND ds2.dataset_version = 'v2'
  AND ds1.diagnosis <> ds2.diagnosis   -- 진단이 같으면 변경 없는 샘플이므로 제외
ORDER BY s.subject_id;

-- (2) 버전별 파이프라인·class 분포 요약
SELECT
    dv.version,
    dv.created_at,
    COUNT(ds.id)                                        AS total_samples,
    -- CASE WHEN으로 클래스별 카운트 (NULL은 SUM에서 무시됨)
    SUM(CASE WHEN ds.diagnosis='CN'       THEN 1 END)  AS cn,
    SUM(CASE WHEN ds.diagnosis='MCI'      THEN 1 END)  AS mci,
    SUM(CASE WHEN ds.diagnosis='Dementia' THEN 1 END)  AS dementia,
    -- 같은 버전 내 모든 샘플이 동일 파이프라인 버전이므로 MAX = 그 버전
    MAX(po.pipeline_version)                            AS pipeline_version
FROM dataset_versions dv
JOIN dataset_samples ds ON ds.dataset_version = dv.version
-- ds.preprocessed_id로 직접 JOIN → pipeline_version 정확히 특정
JOIN preprocessed_outputs po ON po.id = ds.preprocessed_id
GROUP BY dv.version;
```

**결과 요약:**

| subject | viscode | v1 진단 | v2 진단 | 변경 사유 |
|---------|---------|---------|---------|----------|
| S003 | bl | MCI | CN | 임상 재판정 (changes/diagnosis.csv) |
| S006 | m12 | MCI | Dementia | 임상 재판정 (changes/diagnosis.csv) |

| 버전 | 샘플 수 | CN | MCI | Dementia | 파이프라인 |
|------|---------|----|----|----------|-----------|
| v1 | 15 | 8 | 5 | 2 | 1.0 |
| v2 | 15 | 9 | 3 | 3 | 1.1 |

**v1은 발행 당시 snapshot(manifest.json)으로 고정**되며, v2를 발행한 뒤에도 `datasets/v1/manifest.json`에서 완전히 재현 가능합니다.

---

### 4.2. Q2. v1 파일 목록과 진단 시점

**Q. v1에 포함된 각 영상 파일 경로, 진단, 그 진단이 내려진 임상 시점은?**

```sql
SELECT
    s.subject_id,
    s.image_id,
    s.series_description,
    po.output_path                AS file_path,       -- 모델이 읽을 NIfTI 파일 경로
    po.pipeline_version,                              -- 전처리에 사용된 파이프라인 버전
    ds.diagnosis,                                     -- 발행 시점에 확정된 진단
    ds.diagnosis_viscode          AS diag_viscode,    -- 진단이 내려진 임상 방문 코드 (bl, m06…)
    ds.diagnosis_exam_date        AS diag_exam_date,  -- 임상 방문일 (촬영일과 최대 ±30일 차이)
    ds.diagnosis_stamp            AS diag_last_updated, -- 이 진단 레코드의 마지막 갱신 타임스탬프
    ds.split                                          -- train / val / test
FROM dataset_samples ds
-- ds.preprocessed_id로 JOIN → 해당 샘플에 실제 사용된 output만 가져옴
JOIN series              s  ON s.series_uid = ds.series_uid
JOIN preprocessed_outputs po ON po.id       = ds.preprocessed_id
WHERE ds.dataset_version = 'v1'
ORDER BY s.subject_id, ds.diagnosis_viscode;
```

**포인트:**
- `file_path`: 모델이 실제 읽을 NIfTI 파일 경로 (sha256으로 무결성 검증 가능)
- `diag_exam_date`: 영상 촬영일이 아닌 **임상 방문일** (최대 ±30일 허용)
- `diag_last_updated`: 이 진단 레코드가 마지막으로 갱신된 타임스탬프 → v1 발행 시점의 `clinical_snapshot_hash`로 묶여 있어 나중에 진단이 바뀌어도 추적 가능

---

### 4.3. Q3. 진단 수정 1건이 영향을 주는 데이터셋 버전

**Q. `changes/diagnosis.csv`의 수정 2건이 어떤 버전에 반영되고, v1은 보호되는가?**

```sql
-- (1) 버전별 발행 시점의 진단 스냅샷 해시 (manifest 고정 여부 확인)
-- clinical_snapshot_hash가 다르면 발행 시점의 진단 상태가 달랐음을 의미
SELECT version, created_at, manifest_path
FROM dataset_versions
ORDER BY created_at;

-- (2) 진단 이력 전체 조회: 원본 + 수정본이 모두 보존됨 (append-only)
-- update_stamp 오름차순 → 시간 순서대로 변경 이력 확인 가능
SELECT
    subject_id, viscode2, diagnosis, update_stamp, source_file
FROM clinical_visit_history
WHERE subject_id IN ('S003', 'S006')
ORDER BY subject_id, viscode2, update_stamp;

-- (3) 현재 유효 진단 (각 subject·viscode2 조합에서 가장 최신 update_stamp인 행)
SELECT
    h.subject_id, h.viscode2, h.diagnosis, h.update_stamp
FROM clinical_visit_history h
-- 서브쿼리: (subject_id, viscode2)별 최신 타임스탬프 추출
JOIN (
    SELECT subject_id, viscode2, MAX(update_stamp) AS ms
    FROM clinical_visit_history
    GROUP BY subject_id, viscode2
) latest USING (subject_id, viscode2)   -- USING → subject_id, viscode2 동시 매칭
WHERE h.update_stamp = latest.ms        -- 최신 stamp인 행만 선택
  AND h.subject_id IN ('S003', 'S006')
ORDER BY h.subject_id, h.viscode2;
```

**결과 해석:**

```
clinical_visit_history (append-only)
  S003 / bl : MCI  update_stamp=2021-03-07 10:00:00  ← v1 발행 당시
  S003 / bl : CN   update_stamp=2026-10-02 09:00:00  ← changes 적재 후
  S006 / m12: MCI  update_stamp=2022-06-11 10:00:00
  S006 / m12: Dem  update_stamp=2026-10-02 09:00:00
```

- **v1**: `manifest.json`의 `clinical_snapshot_hash`가 수정 전 진단으로 고정 → 불변
- **v2**: `02_link.py`가 `MAX(update_stamp)`로 최신 진단을 선택 → 수정 반영
- 이력이 모두 DB에 남아 있어 "언제 어떤 진단이었는가"를 언제든 재현 가능

**(4) 진단 수정이 어떤 데이터셋 버전에 영향을 주는가 — 영향도 조회**

학습 팀이 진단 오류를 발견했을 때 "이 수정이 어떤 버전에 반영됐는가"를 시스템에 물어 답을 받는 쿼리:

```sql
-- 특정 (subject_id, viscode2) 수정의 버전별 영향 확인
-- 예: S003 / bl 진단이 MCI → CN으로 수정된 경우
SELECT
    ds.dataset_version,
    ds.diagnosis             AS label_in_version,   -- 해당 버전에서 쓰인 진단
    ds.diagnosis_stamp       AS stamp_at_publish,    -- 발행 당시 진단 타임스탬프
    ds.split
FROM dataset_samples ds
JOIN series s ON s.series_uid = ds.series_uid
WHERE s.subject_id       = 'S003'       -- 수정 대상 subject
  AND ds.diagnosis_viscode = 'bl'       -- 수정 대상 방문
ORDER BY ds.dataset_version;
-- 기대 결과:
--   v1 | MCI | 2021-03-07 10:00:00 | train  ← 구 진단 사용, 모델 재학습 필요
--   v2 | CN  | 2026-10-02 09:00:00 | train  ← 신 진단 반영, 정상
```

이 쿼리로 "어떤 버전이 구 진단을 가지고 있는가"를 즉시 파악하고, 영향받는 버전의 모델을 재학습 대상으로 지정할 수 있습니다.

---

### 4.4. Q4. 학습·평가 데이터 오염 여부 확인 방법과 결과

**Q. train/val/test 간 subject 중복(data leakage)이 없는지 어떻게 확인하는가?**

```sql
-- (1) leakage 감지: 동일 subject가 2개 이상의 split에 속하는 경우 조회
SELECT
    s.subject_id,
    COUNT(DISTINCT ds.split)        AS split_count,    -- 1이면 정상, 2 이상이면 leakage
    GROUP_CONCAT(DISTINCT ds.split) AS splits_assigned, -- 어느 split에 중복됐는지 확인
    ds.dataset_version
FROM dataset_samples ds
JOIN series s ON s.series_uid = ds.series_uid
-- (dataset_version, subject_id) 단위로 집계 → subject가 여러 split에 걸쳐 있는지 확인
GROUP BY ds.dataset_version, s.subject_id
HAVING COUNT(DISTINCT ds.split) > 1;  -- 기대 결과: 0행 (leakage 없음)

-- (2) split별 subject 수·샘플 수 분포 확인
SELECT
    ds.dataset_version,
    ds.split,
    COUNT(DISTINCT s.subject_id)  AS n_subjects,       -- split 단위 고유 환자 수
    COUNT(*)                      AS n_samples,         -- 샘플 수 (1 subject = 여러 visit 가능)
    GROUP_CONCAT(DISTINCT s.subject_id) AS subjects    -- ORDER BY는 SQLite 3.44+ 전용; 호환성 위해 제거
FROM dataset_samples ds
JOIN series s ON s.series_uid = ds.series_uid
GROUP BY ds.dataset_version, ds.split
ORDER BY ds.dataset_version, ds.split;
```

**결과 (0행 = leakage 없음):**

| 버전 | split | subjects | 샘플 수 | subject 목록 |
|------|-------|:--------:|:-------:|-------------|
| v1 | train | 6 | 9 | S003 S004 S005 S006 S007 S008 |
| v1 | val | 1 | 3 | S002 |
| v1 | test | 1 | 3 | S001 |
| v2 | train | 6 | 9 | S003 S004 S005 S006 S007 S008 |
| v2 | val | 1 | 3 | S002 |
| v2 | test | 1 | 3 | S001 |

v1·v2 모두 동일 subject 구성 — 진단 수정은 split에 영향을 주지 않음.

leakage가 발생하는 대표적 패턴:
1. sample 단위 무작위 split → 동일 subject의 bl/m12가 각기 다른 split에 배정
2. 미래 방문 진단을 현재 영상의 레이블로 사용 (temporal leakage)

본 파이프라인은 **subject 단위 split** + **동점 시 촬영 이전 방문 우선** 규칙으로 두 가지 모두 방지합니다.

---

### 4.5. Q5. 특정 영상에 어떤 진단 기록이 왜 연결됐는지

**Q. `visit_series_links`에서 특정 series가 어떤 임상 방문에 연결됐고, 그 근거는?**

```sql
-- 모든 T1 series의 연결 근거 (날짜 차이 포함)
SELECT
    s.subject_id,
    s.image_id,
    s.series_description,
    st.study_date                   AS mri_study_date,      -- 영상 촬영일 (YYYY-MM-DD)
    h.viscode2                      AS linked_visit,         -- 연결된 임상 방문 코드
    h.exam_date                     AS clinical_exam_date,   -- 임상 방문일
    vsl.date_diff_days              AS date_diff,            -- |촬영일 − 방문일| 일수
    vsl.link_rule,                                           -- nearest_within_30d
    h.diagnosis,                                             -- 연결된 진단값
    h.update_stamp                  AS diagnosis_as_of       -- 진단 레코드 갱신 타임스탬프
FROM visit_series_links vsl
-- vsl.series_uid → series → studies 순서로 JOIN (study_date는 studies 테이블에만 있음)
JOIN series s             ON s.series_uid  = vsl.series_uid
JOIN studies st           ON st.study_uid  = s.study_uid
-- vsl.visit_history_id가 clinical_visit_history의 PK(id)를 직접 참조 → 연결 근거 명확
JOIN clinical_visit_history h ON h.id      = vsl.visit_history_id
ORDER BY s.subject_id, st.study_date;
```

**연결 로직 (02_link.py):**

```
링크 선택 기준 (우선순위):
  1순위: |StudyDate − EXAMDATE| 가 가장 작은 방문
  2순위: (동점 시) 촬영 이전 방문 (exam_date ≤ study_date)
  제외:  30일 범위 초과 → 미연결 (labeled dataset에서 제외)
```

예시: MRI StudyDate = 2019-08-01, 방문 A = 2019-07-20 (diff=12일, 이전), 방문 B = 2019-08-05 (diff=4일, 이후)
→ **방문 B 선택** (diff 4 < 12). 만약 diff 동일하면 이전 방문 A를 선택 (leakage 방지).

---

### 4.6. Q6. S001의 임상 방문마다 대응하는 T1·DWI·PET와 대응 규칙

**Q. S001의 각 임상 방문에 어떤 T1·DWI·PET 영상이 대응되며, 각 연결 방식은?**

```sql
-- S001의 visit별 모든 모달리티 영상
-- T1·DWI: T1의 visit_series_links 기준 (DWI는 동일 study_uid로 상속)
-- PET: 자체 visit_series_links 행 (별도 study)

-- Part 1: T1 + DWI (T1의 study_uid를 공유하는 모든 series)
SELECT
    h.viscode2             AS visit,
    h.exam_date,
    h.diagnosis,
    st.study_date          AS mri_study_date,
    s.image_type,          -- T1 / DWI
    s.modality,
    s.series_description,
    s.image_id,
    vsl.date_diff_days     AS link_diff,
    CASE s.image_type
        WHEN 'T1' THEN 'visit_series_links (±30일 근접 매칭)'
        ELSE           '동일 StudyInstanceUID (T1 연결 상속)'
    END AS link_basis
FROM visit_series_links vsl
-- T1 앵커만 선택 (PET는 자체 링크가 있으므로 Part 2에서 처리)
JOIN series s_t1 ON s_t1.series_uid = vsl.series_uid
                AND s_t1.image_type  = 'T1'
JOIN studies st  ON st.study_uid    = s_t1.study_uid
JOIN clinical_visit_history h ON h.id = vsl.visit_history_id
-- T1과 같은 study의 모든 series (T1 + DWI)
JOIN series s    ON s.study_uid     = st.study_uid
WHERE s_t1.subject_id = 'S001'
  AND h.update_stamp = (
      SELECT MAX(update_stamp) FROM clinical_visit_history
      WHERE subject_id = 'S001' AND viscode2 = h.viscode2
  )

UNION ALL

-- Part 2: PET (자체 visit_series_links 행, 별도 study)
SELECT
    h.viscode2,
    h.exam_date,
    h.diagnosis,
    st.study_date          AS mri_study_date,
    s.image_type,          -- PET
    s.modality,
    s.series_description,
    s.image_id,
    vsl.date_diff_days     AS link_diff,
    'visit_series_links (±30일 근접 매칭, 별도 PET study)' AS link_basis
FROM visit_series_links vsl
-- PET 직접 링크
JOIN series s    ON s.series_uid    = vsl.series_uid
                AND s.image_type    = 'PET'
JOIN studies st  ON st.study_uid    = s.study_uid
JOIN clinical_visit_history h ON h.id = vsl.visit_history_id
WHERE s.subject_id = 'S001'
  AND h.update_stamp = (
      SELECT MAX(update_stamp) FROM clinical_visit_history
      WHERE subject_id = 'S001' AND viscode2 = h.viscode2
  )

ORDER BY visit, image_type;
```

**대응 규칙 요약:**

```
visit (임상) ──[ ±30일 ]──▶ T1 series  (visit_series_links)
                                 │
                                 │ 동일 StudyInstanceUID
                                 ▼
                         DWI series (같은 MRI 세션)

visit (임상) ──[ ±30일 ]──▶ PET series (visit_series_links, 별도 study)
```

- **T1**: `visit_series_links`에 명시적으로 등록된 MRI 앵커 모달리티
- **DWI**: T1과 같은 `StudyInstanceUID`를 공유 → T1의 임상 방문 연결을 상속
  - 별도 링크 없이 `JOIN series ON study_uid`로 조회 가능
  - T1이 미연결인 study의 DWI는 labeled dataset에서 제외됨
- **PET**: MRI와 별도 세션으로 촬영 (다른 `StudyInstanceUID`) → `visit_series_links`에 직접 등록
  - T1·DWI와 독립적으로 ±30일 매칭 → T1 연결 여부와 무관하게 처리됨

---

## 5. 디렉토리 구조

```
mri-dataset-pipeline/
├── data/
│   └── sample_bundle/
│       ├── series/          # sidecar JSON (영상 메타데이터)
│       ├── derived/v1.0/    # 전처리 결과 JSON (파이프라인 v1.0)
│       ├── clinical/        # diagnosis.csv
│       └── changes/         # v2용 변경분
│           ├── diagnosis.csv      # 수정된 임상 진단 (2건)
│           └── derived/v1.1/      # 파이프라인 v1.1 전처리 출력 JSON
├── db/
│   └── registry.db          # SQLite (스크립트 실행 후 생성)
├── datasets/
│   ├── v1/                  # manifest.json + train/val/test.csv
│   └── v2/
├── scripts/
│   ├── 01_ingest.py
│   ├── 02_link.py
│   ├── 03_publish.py
│   └── 04_validate.py
├── dataset_config.json      # 설계 근거·필드 명세 참조 문서 (스크립트가 직접 읽지 않음)
├── requirements.txt
└── README.md
```
