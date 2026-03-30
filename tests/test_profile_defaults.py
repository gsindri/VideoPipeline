from pathlib import Path

from videopipeline.profile import load_profile, resolve_default_profile_path
from videopipeline.studio.dag_config import profile_default_llm_mode, resolve_llm_mode


def test_resolve_default_profile_path_prefers_assembly_when_key_present(tmp_path, monkeypatch):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "gaming.yaml").write_text("analysis: {}\n", encoding="utf-8")
    assembly = profiles_dir / "gaming_assemblyai.yaml"
    assembly.write_text("studio:\n  default_llm_mode: external_strict\n", encoding="utf-8")

    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key")
    monkeypatch.delenv("VP_PROFILE", raising=False)
    monkeypatch.delenv("VP_STUDIO_PROFILE", raising=False)

    assert resolve_default_profile_path(search_roots=[tmp_path]) == assembly


def test_resolve_default_profile_path_prefers_gaming_without_assembly_key(tmp_path, monkeypatch):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    gaming = profiles_dir / "gaming.yaml"
    gaming.write_text("analysis: {}\n", encoding="utf-8")
    (profiles_dir / "gaming_assemblyai.yaml").write_text("analysis: {}\n", encoding="utf-8")

    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.delenv("AAI_API_KEY", raising=False)
    monkeypatch.delenv("VP_PROFILE", raising=False)
    monkeypatch.delenv("VP_STUDIO_PROFILE", raising=False)

    assert resolve_default_profile_path(search_roots=[tmp_path]) == gaming


def test_load_profile_none_uses_env_profile(tmp_path, monkeypatch):
    profile_path = tmp_path / "custom.yaml"
    profile_path.write_text("studio:\n  default_llm_mode: external_strict\n", encoding="utf-8")

    monkeypatch.setenv("VP_STUDIO_PROFILE", str(profile_path))

    loaded = load_profile(None)
    assert loaded["studio"]["default_llm_mode"] == "external_strict"


def test_profile_default_llm_mode_defaults_to_external_strict():
    assert profile_default_llm_mode({}) == "external_strict"
    assert resolve_llm_mode(None, profile={}) == "external_strict"


def test_gaming_profiles_use_tighter_context_top_n():
    repo_root = Path(__file__).resolve().parents[1]

    for profile_name in ("gaming.yaml", "gaming_assemblyai.yaml", "gaming_nvidia.yaml"):
        profile = load_profile(repo_root / "profiles" / profile_name)
        assert profile["context"]["top_n"] == 12
