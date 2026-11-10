"""The optional sidecar ``device_name``: written when a CUDA GPU's name can be read, left out otherwise."""

import pytest

torch = pytest.importorskip("torch")

from viz2psy import metadata  # noqa: E402
from viz2psy.metadata import MetadataBuilder  # noqa: E402


def _fake_cuda(monkeypatch, names=("NVIDIA H100 80GB HBM3",), current=0, fail=False):
    def get_device_name(i):
        if fail:
            raise RuntimeError("no driver")
        return names[i]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    monkeypatch.setattr(torch.cuda, "get_device_name", get_device_name)


def _built(device):
    b = MetadataBuilder()
    b.set_device(device)
    return b.build()


def test_cuda_device_records_the_gpu_name(monkeypatch):
    _fake_cuda(monkeypatch)
    meta = _built("cuda")
    assert meta["device"] == "cuda" and meta["device_name"] == "NVIDIA H100 80GB HBM3"


def test_indexed_device_names_that_gpu(monkeypatch):
    _fake_cuda(monkeypatch, names=("NVIDIA A100 80GB PCIe", "NVIDIA H100 80GB HBM3"))
    assert metadata.device_name("cuda:1") == "NVIDIA H100 80GB HBM3"
    assert metadata.device_name("cuda") == "NVIDIA A100 80GB PCIe"  # the current device


@pytest.mark.parametrize("device", ["cpu", "mps", None])
def test_non_cuda_devices_leave_the_key_out(device):
    assert metadata.device_name(device) is None
    if device:
        assert "device_name" not in _built(device)


def test_an_unreadable_name_never_fails_the_run(monkeypatch):
    _fake_cuda(monkeypatch, fail=True)
    meta = _built("cuda:0")
    assert meta["device"] == "cuda:0" and "device_name" not in meta


def test_no_device_set_leaves_the_key_out():
    assert "device_name" not in MetadataBuilder().build()
