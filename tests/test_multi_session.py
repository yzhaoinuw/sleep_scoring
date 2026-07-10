"""Tests for multi-session (multiple app windows) support: env-driven config,
per-slot dirs, the peer current-file endpoint, and the same-file load refusal."""

import importlib
import json
from unittest.mock import MagicMock
from urllib.error import URLError

import dash


def _reload_config():
    import app_src.config

    return importlib.reload(app_src.config)


class TestConfigEnvContract:
    """app_src.config reads the slot and peer ports exported by the launcher."""

    def test_defaults_to_slot_zero_with_no_peers(self, monkeypatch):
        monkeypatch.delenv("SLEEP_SCORING_INSTANCE_SLOT", raising=False)
        monkeypatch.delenv("SLEEP_SCORING_PEER_PORTS", raising=False)
        try:
            config = _reload_config()
            assert config.INSTANCE_SLOT == 0
            assert config.PEER_PORTS == []
        finally:
            monkeypatch.undo()
            _reload_config()

    def test_reads_slot_and_peer_ports(self, monkeypatch):
        monkeypatch.setenv("SLEEP_SCORING_INSTANCE_SLOT", "1")
        monkeypatch.setenv("SLEEP_SCORING_PEER_PORTS", "8050,8052")
        try:
            config = _reload_config()
            assert config.INSTANCE_SLOT == 1
            assert config.PEER_PORTS == [8050, 8052]
        finally:
            monkeypatch.undo()
            _reload_config()

    def test_ignores_malformed_env_values(self, monkeypatch):
        monkeypatch.setenv("SLEEP_SCORING_INSTANCE_SLOT", "not-a-slot")
        monkeypatch.setenv("SLEEP_SCORING_PEER_PORTS", "abc, 8051 ,,")
        try:
            config = _reload_config()
            assert config.INSTANCE_SLOT == 0
            assert config.PEER_PORTS == [8051]
        finally:
            monkeypatch.undo()
            _reload_config()

    def test_later_window_forces_perf_logging_off(self, monkeypatch):
        monkeypatch.setenv("SLEEP_SCORING_INSTANCE_SLOT", "2")
        try:
            config = _reload_config()
            assert config.RESAMPLER_PERF_LOG is False
            assert config.BROWSER_NAVIGATION_PERF_LOG is False
            assert config.PROFILE_RESAMPLER_UPDATES is False
        finally:
            monkeypatch.undo()
            _reload_config()


class TestAdoptLegacyTempFiles:
    """Pre-multi-session cache files move into slot 0 on first upgraded launch."""

    def test_moves_loose_files_into_fresh_slot_dir(self, tmp_path):
        from app_src.server import adopt_legacy_temp_files

        (tmp_path / "cache_entry").write_text("cached")
        (tmp_path / "recording.mat").write_text("mat")
        slot_dir = tmp_path / "slot_0"

        adopt_legacy_temp_files(tmp_path, slot_dir)

        assert (slot_dir / "cache_entry").read_text() == "cached"
        assert (slot_dir / "recording.mat").exists()
        assert not (tmp_path / "cache_entry").exists()

    def test_leaves_subdirectories_in_place(self, tmp_path):
        from app_src.server import adopt_legacy_temp_files

        other_slot = tmp_path / "slot_1"
        other_slot.mkdir()
        (other_slot / "cache_entry").write_text("slot 1 cache")

        adopt_legacy_temp_files(tmp_path, tmp_path / "slot_0")

        assert (other_slot / "cache_entry").read_text() == "slot 1 cache"

    def test_noop_when_slot_dir_already_exists(self, tmp_path):
        from app_src.server import adopt_legacy_temp_files

        slot_dir = tmp_path / "slot_0"
        slot_dir.mkdir()
        (tmp_path / "cache_entry").write_text("cached")

        adopt_legacy_temp_files(tmp_path, slot_dir)

        assert (tmp_path / "cache_entry").exists()
        assert not (slot_dir / "cache_entry").exists()


class TestCurrentFileEndpoint:
    def _client(self, monkeypatch, filepath):
        from app_src import routes  # noqa: F401 -- importing registers the endpoint
        from app_src import session
        from app_src.server import app

        monkeypatch.setattr(session, "_current_filepath", filepath)
        return app.server.test_client()

    def test_reports_open_file(self, monkeypatch):
        client = self._client(monkeypatch, "/data/recording.mat")

        response = client.get("/_sleep_scoring/current-file")

        assert response.get_json() == {
            "app": "sleep_scoring",
            "filepath": "/data/recording.mat",
        }

    def test_reports_empty_string_when_no_file_open(self, monkeypatch):
        client = self._client(monkeypatch, None)

        response = client.get("/_sleep_scoring/current-file")

        assert response.get_json() == {"app": "sleep_scoring", "filepath": ""}

    def test_fresh_process_ignores_stale_cached_filepath(self, monkeypatch):
        """A restarted slot reuses its cache dir, where filepath persists for
        days; the endpoint must not report it before a file is opened here."""
        from app_src import server

        fake_cache = MagicMock()
        fake_cache.get.return_value = "/data/stale.mat"
        monkeypatch.setattr(server, "cache", fake_cache)
        client = self._client(monkeypatch, None)

        response = client.get("/_sleep_scoring/current-file")

        assert response.get_json() == {"app": "sleep_scoring", "filepath": ""}

    def test_initialize_cache_updates_process_state(self, monkeypatch):
        from app_src import session

        monkeypatch.setattr(session, "_current_filepath", None)
        monkeypatch.setattr(session, "clear_temp_dir", lambda filename: None)
        monkeypatch.setattr(session, "clear_fig_resamplers", lambda: None)

        session.initialize_cache(MagicMock(), "/data/recording.mat")

        assert session.get_current_filepath() == "/data/recording.mat"


class _FakeResponse:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode()

    def read(self, *args):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestFindPeerSessionWithFile:
    def test_returns_port_when_peer_has_same_file(self, monkeypatch):
        from app_src import session

        monkeypatch.setattr(session, "PEER_PORTS", [8051])
        monkeypatch.setattr(
            session,
            "urlopen",
            lambda url, timeout: _FakeResponse(
                {"app": "sleep_scoring", "filepath": "/data/./recording.mat"}
            ),
        )

        assert session.find_peer_session_with_file("/data/recording.mat") == 8051

    def test_returns_none_when_peer_has_different_file(self, monkeypatch):
        from app_src import session

        monkeypatch.setattr(session, "PEER_PORTS", [8051])
        monkeypatch.setattr(
            session,
            "urlopen",
            lambda url, timeout: _FakeResponse(
                {"app": "sleep_scoring", "filepath": "/data/other.mat"}
            ),
        )

        assert session.find_peer_session_with_file("/data/recording.mat") is None

    def test_skips_dead_peer_and_checks_next(self, monkeypatch):
        from app_src import session

        def fake_urlopen(url, timeout):
            if ":8050/" in url:
                raise URLError("connection refused")
            return _FakeResponse({"app": "sleep_scoring", "filepath": "/data/recording.mat"})

        monkeypatch.setattr(session, "PEER_PORTS", [8050, 8052])
        monkeypatch.setattr(session, "urlopen", fake_urlopen)

        assert session.find_peer_session_with_file("/data/recording.mat") == 8052

    def test_ignores_non_app_listener_on_peer_port(self, monkeypatch):
        from app_src import session

        monkeypatch.setattr(session, "PEER_PORTS", [8051])
        monkeypatch.setattr(
            session,
            "urlopen",
            lambda url, timeout: _FakeResponse({"filepath": "/data/recording.mat"}),
        )

        assert session.find_peer_session_with_file("/data/recording.mat") is None

    def test_no_peers_configured_never_queries(self, monkeypatch):
        from app_src import session

        def fail_urlopen(url, timeout):
            raise AssertionError("urlopen should not be called without peers")

        monkeypatch.setattr(session, "PEER_PORTS", [])
        monkeypatch.setattr(session, "urlopen", fail_urlopen)

        assert session.find_peer_session_with_file("/data/recording.mat") is None


class TestChooseMatPeerRefusal:
    def test_refuses_file_open_in_another_window(self, monkeypatch):
        from app_src.callbacks import loading

        monkeypatch.setattr(loading, "open_file_dialog", lambda file_type: "/data/recording.mat")
        monkeypatch.setattr(loading, "find_peer_session_with_file", lambda filepath: 8051)
        initialize_cache = MagicMock()
        monkeypatch.setattr(loading, "initialize_cache", initialize_cache)

        message, ready = loading.choose_mat(1)

        assert '"recording.mat" is already open in another' in message
        assert ready is dash.no_update
        initialize_cache.assert_not_called()

    def test_loads_file_when_no_peer_has_it(self, monkeypatch):
        from app_src.callbacks import loading

        monkeypatch.setattr(loading, "open_file_dialog", lambda file_type: "/data/recording.mat")
        monkeypatch.setattr(loading, "find_peer_session_with_file", lambda filepath: None)
        initialize_cache = MagicMock()
        monkeypatch.setattr(loading, "initialize_cache", initialize_cache)

        message, ready = loading.choose_mat(1)

        assert ready == "vis"
        initialize_cache.assert_called_once()


class TestClipUrlIncludesSlotDir:
    def test_show_clip_serves_from_slot_subdir(self, monkeypatch, tmp_path):
        from app_src.callbacks import video
        from pathlib import Path

        slot_video_dir = tmp_path / "slot_2"
        slot_video_dir.mkdir()
        (slot_video_dir / "clip.mp4").write_bytes(b"")
        monkeypatch.setattr(video, "VIDEO_DIR", slot_video_dir)

        _, player, _ = video.show_clip("clip.mp4")

        assert player.url == str(Path("/assets/videos") / "slot_2" / "clip.mp4")
