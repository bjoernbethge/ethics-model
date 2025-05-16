import pytest
import torch
import numpy as np
from src.torch_template import utils


def test_get_device():
    device = utils.get_device()
    assert isinstance(device, torch.device)
    assert str(device) in ["cuda", "cpu", "mps"]


def test_seed_everything_reproducibility():
    utils.seed_everything(123)
    a = torch.rand(1).item()
    utils.seed_everything(123)
    b = torch.rand(1).item()
    assert a == b


def test_count_parameters():
    model = torch.nn.Linear(10, 2)
    total, trainable = utils.count_parameters(model)
    assert total == trainable
    assert total == sum(p.numel() for p in model.parameters())


def test_to_device_tensor():
    x = torch.tensor([1.0, 2.0])
    device = utils.get_device()
    y = utils.to_device(x, device)
    assert isinstance(y, torch.Tensor)
    assert y.device == device


def test_to_device_list_and_dict():
    device = utils.get_device()
    data = [torch.tensor([1]), torch.tensor([2])]
    moved = utils.to_device(data, device)
    assert all(t.device == device for t in moved)
    data_dict = {"a": torch.tensor([1]), "b": torch.tensor([2])}
    moved_dict = utils.to_device(data_dict, device)
    assert all(t.device == device for t in moved_dict.values())


def test_average_meter():
    meter = utils.AverageMeter()
    meter.update(2, n=2)
    meter.update(4, n=1)
    assert meter.count == 3
    assert meter.avg == pytest.approx((2*2 + 4*1)/3)
    meter.reset()
    assert meter.count == 0


def test_freeze_and_unfreeze():
    model = torch.nn.Linear(2, 2)
    utils.freeze_layers(model)
    assert all(not p.requires_grad for p in model.parameters())
    utils.unfreeze_all(model)
    assert all(p.requires_grad for p in model.parameters())


def test_model_summary_prints(capsys):
    model = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.ReLU())
    utils.model_summary(model, input_size=(4,))
    captured = capsys.readouterr()
    assert "Layer (type)" in captured.out


def test_temp_seed_context():
    a = torch.rand(1).item()
    with utils.temp_seed(42):
        b = torch.rand(1).item()
        c = torch.rand(1).item()
    with utils.temp_seed(42):
        d = torch.rand(1).item()
        e = torch.rand(1).item()
    assert b == d and c == e


def test_early_stopping(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    es = utils.EarlyStopping(patience=2, verbose=False, path=str(tmp_path / "chk.pt"))
    losses = [1.0, 0.9, 0.8, 0.8, 0.8, 0.7]
    for epoch, loss in enumerate(losses):
        es(loss, model, epoch, optimizer)
        if es.early_stop:
            break
    assert es.early_stop or es.counter > 0
