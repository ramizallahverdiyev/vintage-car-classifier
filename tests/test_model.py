import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import build_model


def test_model_output_shape():
    model = build_model(num_classes=30, pretrained=False)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)

    assert output.shape == (4, 30), f"Expected (4, 30), got {output.shape}"


def test_model_class_count():
    model = build_model(num_classes=10, pretrained=False)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    assert output.shape[1] == 10


def test_model_train_eval_modes():
    model = build_model(num_classes=30, pretrained=False)
    model.train()
    assert model.training

    model.eval()
    assert not model.training


def test_model_dropout_config():
    model_no_dropout = build_model(num_classes=30, pretrained=False, dropout=0.0)
    model_dropout = build_model(num_classes=30, pretrained=False, dropout=0.5)

    assert isinstance(model_no_dropout.classifier[0], torch.nn.Dropout)
    assert model_no_dropout.classifier[0].p == 0.0
    assert model_dropout.classifier[0].p == 0.5


def test_forward_pass_different_batch_sizes():
    model = build_model(num_classes=30, pretrained=False)
    model.eval()

    for batch_size in [1, 2, 8, 16]:
        dummy = torch.randn(batch_size, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (batch_size, 30), f"Failed for batch size {batch_size}"
