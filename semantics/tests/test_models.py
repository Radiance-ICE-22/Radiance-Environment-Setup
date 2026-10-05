"""models.smoke against open_clip 2.24's real API (randomly initialised ViT-B-16, so no
download) and a stand-in DINOv2 with the hub model's forward_features contract."""

import pytest

torch = pytest.importorskip("torch")
open_clip = pytest.importorskip("open_clip")

from radiance_semantics import CLIP_DIM, DINO_DIM, models  # noqa: E402


class FakeDino(torch.nn.Module):
    def forward_features(self, x):
        n = (x.shape[-2] // 14) * (x.shape[-1] // 14)
        return {"x_norm_patchtokens": torch.zeros(x.shape[0], n, DINO_DIM),
                "x_norm_clstoken": torch.zeros(x.shape[0], DINO_DIM)}


@pytest.fixture
def offline_models(monkeypatch):
    def clip(device="cpu"):
        m, _, pre = open_clip.create_model_and_transforms(models.CLIP_MODEL, pretrained=None, device=device)
        return m.eval(), pre, open_clip.get_tokenizer(models.CLIP_MODEL)
    monkeypatch.setattr(models, "load_clip", clip)
    monkeypatch.setattr(models, "load_dino", lambda device="cpu": FakeDino())


def test_smoke_shapes_with_open_clip_2_24(offline_models):
    r = models.smoke("cpu")
    assert r["clip_dim"] == CLIP_DIM
    assert len(r["clip_sim"]) == 2 and len(r["clip_sim"][0]) == 2
    assert r["dino_tokens"] == [1, 256, DINO_DIM]
    # random weights cannot tell red from blue, so `ok` reflects only the sanity rule, not a crash
    assert r["ok"] in (True, False)


def test_cli_requires_an_action(capsys):
    with pytest.raises(SystemExit):
        models.main([])
