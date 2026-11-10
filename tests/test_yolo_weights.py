"""YOLO weights resolve under the torch hub cache, never into the cwd.

Ultralytics downloads a bare weights name it cannot find into the current
directory; viz2psy resolves bare names under ``<torch hub dir>/ultralytics``
instead. Offline: the download is faked.
"""

from pathlib import Path

import pytest

pytest.importorskip("ultralytics")
torch = pytest.importorskip("torch")

from ultralytics.utils import downloads  # noqa: E402

from viz2psy.models import yolo  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    hub, cwd = tmp_path / "hub", tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(hub))
    monkeypatch.chdir(cwd)
    calls = []

    def fake_download(file, *a, **k):
        calls.append(Path(file))
        Path(file).write_bytes(b"weights")
        return str(file)

    monkeypatch.setattr(downloads, "attempt_download_asset", fake_download)
    return {"hub": hub, "cwd": cwd, "calls": calls}


def test_bare_name_downloads_once_into_the_cache(env):
    got = yolo.resolve_weights("yolov8n.pt")
    assert Path(got) == env["hub"] / "ultralytics" / "yolov8n.pt"
    assert Path(got).read_bytes() == b"weights"
    assert list(env["cwd"].iterdir()) == []
    assert yolo.resolve_weights("yolov8n.pt") == got
    assert len(env["calls"]) == 1
    # the download went to a private temp dir, now cleaned up
    assert env["calls"][0].parent != Path(got).parent
    assert sorted(p.name for p in Path(got).parent.iterdir()) == ["yolov8n.pt"]


def test_explicit_paths_pass_through(env, tmp_path):
    mine = tmp_path / "custom" / "my.pt"
    assert yolo.resolve_weights(str(mine)) == str(mine)
    assert yolo.resolve_weights("sub/dir.pt") == "sub/dir.pt"
    assert env["calls"] == []


def test_a_bare_name_in_the_cwd_is_still_honoured(env):
    (env["cwd"] / "yolov8n.pt").write_bytes(b"local")
    assert yolo.resolve_weights("yolov8n.pt") == "yolov8n.pt"
    assert env["calls"] == []


def test_a_failed_download_raises_and_leaves_no_target(env, monkeypatch):
    monkeypatch.setattr(downloads, "attempt_download_asset", lambda file, *a, **k: str(file))
    with pytest.raises(FileNotFoundError, match="could not be downloaded"):
        yolo.resolve_weights("yolov8n.pt")
    assert not (env["hub"] / "ultralytics" / "yolov8n.pt").exists()
