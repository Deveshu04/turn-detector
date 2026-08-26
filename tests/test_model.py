import pytest
import torch

from turn_detector.model import (
    TinyMelNet, WhisperTinyTurn, count_params, truncate_whisper_encoder,
)


def test_tinymelnet_forward_and_size():
    model = TinyMelNet()
    out = model(torch.randn(2, 80, 800))
    assert out.shape == (2,)
    assert count_params(model) < 2_000_000


def test_whisper_turn_random_init():
    """Truncated encoder accepts 800-frame mel without pretrained download."""
    from transformers import WhisperConfig, WhisperModel
    config = WhisperConfig()  # whisper-tiny dimensions by default
    encoder = WhisperModel(config).encoder
    model = WhisperTinyTurn(truncate_whisper_encoder(encoder))
    out = model(torch.randn(2, 80, 800))
    assert out.shape == (2,)
    assert model.encoder.embed_positions.weight.shape[0] == 400


@pytest.mark.slow
def test_whisper_turn_pretrained():
    model = WhisperTinyTurn.from_pretrained()
    out = model(torch.randn(1, 80, 800))
    assert out.shape == (1,)
    assert 7_000_000 < count_params(model) < 10_000_000
