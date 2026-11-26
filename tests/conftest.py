"""pytest 공통 픽스처."""
import importlib.util
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def modules():
    """각 스크립트 모듈을 한 번만 로드."""
    return {
        "ingest":   _load("01_ingest"),
        "link":     _load("02_link"),
        "publish":  _load("03_publish"),
        "validate": _load("04_validate"),
    }


@pytest.fixture()
def temp_env(tmp_path, modules):
    """테스트별 독립 환경: 임시 DB + 임시 datasets 출력 디렉터리.

    패치 전략:
    - 모든 모듈의 DB_PATH → tmp DB (데이터 오염 방지)
    - publish/validate.ROOT → tmp_path (datasets 출력 리디렉션)
    - ingest.ROOT 는 유지 → load_derived 의 relative_to(ROOT) 가 올바르게 동작
    - publish.PIPELINE_VERSION_FOR에 테스트용 버전 추가
    """
    db_path = tmp_path / "db" / "registry.db"
    db_path.parent.mkdir()
    (tmp_path / "datasets").mkdir()

    # DB_PATH 패치 (04_validate 는 DB_PATH 없음)
    orig_db = {}
    for name, mod in modules.items():
        if hasattr(mod, "DB_PATH"):
            orig_db[name] = mod.DB_PATH
            mod.DB_PATH = db_path

    # ROOT 패치 (publish, validate — 출력/탐색 경로 리디렉션)
    orig_pub_root = modules["publish"].ROOT
    orig_val_root = modules["validate"].ROOT
    modules["publish"].ROOT = tmp_path
    modules["validate"].ROOT = tmp_path

    # 테스트 버전 등록
    orig_pv = modules["publish"].PIPELINE_VERSION_FOR.copy()
    modules["publish"].PIPELINE_VERSION_FOR = {**orig_pv, "tv1": "1.0", "tv2": "1.1"}

    yield {"db_path": db_path, "tmp_path": tmp_path}

    # 복원
    for name, mod in modules.items():
        if name in orig_db:
            mod.DB_PATH = orig_db[name]
    modules["publish"].ROOT = orig_pub_root
    modules["validate"].ROOT = orig_val_root
    modules["publish"].PIPELINE_VERSION_FOR = orig_pv
