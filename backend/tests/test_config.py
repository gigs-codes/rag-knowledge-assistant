"""Regression test for a real bug found while building the single-container
HF Spaces deploy: Settings.upload_dir/chroma_dir/users_db_path/etc. are
declared as `base_dir / "data" / "..."` in the class body, which Pydantic
evaluates once at class-definition time using the bare default `base_dir`
expression — NOT the actual runtime value after an override. Without the
_rederive_paths_from_base_dir model_validator (see config.py), overriding
base_dir would silently fail to affect any of the derived paths, which is
exactly what crashed the Docker image (it tried to write
/app/data/documents.json into a /app/data that was never created, because
upload_dir/chroma_dir had resolved to /data/... instead).

Constructs a fresh Settings(base_dir=...) directly rather than reloading
the shared app.core.config module — the real `settings` singleton is
already imported by api/deps.py at test-session start (via app.main), and
reloading it mid-session would risk leaving other tests' already-built
singletons pointed at stale state.
"""
from app.core.config import Settings


def test_derived_paths_follow_base_dir_override(tmp_path):
    settings = Settings(base_dir=tmp_path)

    assert settings.upload_dir == tmp_path / "data" / "uploads"
    assert settings.chroma_dir == tmp_path / "data" / "chroma"
    assert settings.users_db_path == tmp_path / "data" / "users.db"
    assert settings.feedback_db_path == tmp_path / "data" / "feedback.db"
    assert settings.agent_checkpoints_db_path == tmp_path / "data" / "agent_checkpoints.db"
