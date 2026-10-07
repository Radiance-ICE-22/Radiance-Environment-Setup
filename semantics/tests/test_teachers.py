import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from radiance_semantics import teachers as T  # noqa: E402


def test_crop_starts_cover_the_image():
    assert T.crop_starts(100, 30, 15) == [0, 15, 30, 45, 60, 70]
    assert T.crop_starts(20, 30, 15) == [0]
    for size, tile, stride in [(1080, 54, 27), (1920, 540, 270), (777, 100, 50)]:
        s = T.crop_starts(size, tile, stride)
        assert s[0] == 0 and s[-1] + tile == size and all(b - a <= stride for a, b in zip(s, s[1:]))


def test_common_grid_and_estimates_for_1080p():
    assert T.common_grid(1080, 1920, 0.025) == (40, 71)
    s = T.TeacherSettings()
    assert len(s.scales) == 7 and s.scales[0] == 0.05 and s.scales[-1] == 0.5
    assert 3500 < T.estimate_crops(1080, 1920, s) < 5500            # dominated by the 0.05 scale
    assert T.grid_bytes(1080, 1920, s) == 40 * 71 * 512 * 2 + 36 * 64 * 384 * 2


def test_resample_is_exact_for_linear_fields():
    """A field linear in the crop-centre coordinates must come back exact at the common grid's
    cell centres (inside the centre range; outside it clamps)."""
    H, W = 100, 160
    cy = np.array([10.0, 40.0, 70.0, 90.0])
    cx = np.array([10.0, 50.0, 90.0, 130.0, 150.0])
    yy, xx = np.meshgrid(cy, cx, indexing="ij")
    grid = torch.tensor(np.stack([yy, xx], -1), dtype=torch.float32)
    out = T.resample(grid, cy, cx, H, W, 10, 16).numpy()
    qy = (np.arange(10) + 0.5) * H / 10
    qx = (np.arange(16) + 0.5) * W / 16
    exp_y = np.clip(qy, cy[0], cy[-1])[:, None].repeat(16, 1)
    exp_x = np.clip(qx, cx[0], cx[-1])[None, :].repeat(10, 0)
    assert np.allclose(out[..., 0], exp_y, atol=1e-4) and np.allclose(out[..., 1], exp_x, atol=1e-4)


def colour_encoder(x):
    """Stand-in for CLIP: the crop's mean colour (de-normalised) as a 3-d 'embedding' + 1."""
    mean = torch.tensor(T.CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(T.CLIP_STD).view(1, 3, 1, 1)
    rgb = (x * std + mean).mean(dim=(2, 3))
    return torch.cat([rgb, torch.ones(len(rgb), 1)], 1)


def test_pyramid_keeps_left_right_layout():
    img = torch.zeros(3, 60, 120)
    img[0, :, :60] = 1.0            # left half red
    img[2, :, 60:] = 1.0            # right half blue
    s = T.TeacherSettings(scales=[0.25, 0.5], cell_frac=0.1, min_tile=4, batch=64)
    g, n = T.clip_pyramid(colour_encoder, img, s)
    assert tuple(g.shape) == (*T.common_grid(60, 120, 0.1), 4) and n > 0
    left, right = g[:, 0].mean(0), g[:, -1].mean(0)
    assert left[0] > left[2] and right[2] > right[0]          # red stays left, blue right


def test_dino_dense_grid_shape():
    s = T.TeacherSettings(dino_width=56)
    tok = lambda x: torch.zeros(1, (x.shape[2] // 14) * (x.shape[3] // 14), 384)   # noqa: E731
    d = T.dino_dense(tok, torch.rand(3, 60, 120), s)
    assert tuple(d.shape) == (2, 4, 384)


def make_scene(tmp_path, n=3, H=40, W=64):
    from PIL import Image
    sd = tmp_path / "scene"
    (sd / "images").mkdir(parents=True)
    frames = []
    for i in range(n):
        Image.fromarray((np.random.default_rng(i).random((H, W, 3)) * 255).astype(np.uint8)).save(
            sd / "images" / f"frame_{i:05d}.png")
        frames.append({"file_path": f"images/frame_{i:05d}.png", "transform_matrix": np.eye(4).tolist()})
    (sd / "transforms.json").write_text(json.dumps({"frames": frames[::-1]}))   # order must not matter
    return sd


def test_extract_writes_resumes_and_tags(tmp_path):
    sd = make_scene(tmp_path)
    s = T.TeacherSettings(scales=[0.25, 0.5], cell_frac=0.1, min_tile=4, dino_width=28, batch=64)
    tok = lambda x: torch.zeros(1, (x.shape[2] // 14) * (x.shape[3] // 14), 384)   # noqa: E731
    enc = {"clip": colour_encoder, "dino": tok}
    meta = T.extract(sd, tmp_path / "teachers", s, device="cpu", encoders=enc, log=lambda m: None)
    out = tmp_path / "teachers" / s.tag()
    assert meta["stats"]["done"] == 3 and sorted(meta["images"]) == ["frame_00000", "frame_00001", "frame_00002"]
    assert np.load(out / "clip" / "frame_00001.npy").dtype == np.float16
    maps = T.load_maps(out, "frame_00002")
    assert maps["clip"].shape[-1] == 4 and maps["dino"].shape == (1, 2, 384)
    again = T.extract(sd, tmp_path / "teachers", s, device="cpu", encoders=enc, log=lambda m: None)
    assert again["stats"]["done"] == 0 and again["stats"]["skipped"] == 3
    assert T.TeacherSettings(scales=[0.25]).tag() != s.tag()
    assert T.TeacherSettings(batch=8).tag() == T.TeacherSettings(batch=512).tag()   # batch is not a result


def test_default_tag_is_unchanged_by_variant_fields():
    """Teacher caches already on the hosts (tag d7a30d7264 for the defaults) must stay valid after the
    variant plumbing: clip_mode 'pyramid' is not part of the hash; any other mode or model is."""
    from radiance_semantics.teachers import TeacherSettings
    base = TeacherSettings()
    assert base.tag() == "d7a30d7264"
    assert TeacherSettings(clip_mode="scales").tag() != base.tag()
    assert TeacherSettings(clip_model="ViT-L-14", clip_pretrained="laion2b_s32b_b82k").tag() != base.tag()


def test_clip_regions_paints_each_segment_with_its_own_embedding():
    """Variant B: two segments (left / right halves) get two different embeddings; a cell straddling both
    is their coverage-weighted mean; uncovered cells stay 0."""
    s = T.TeacherSettings(cell_frac=0.25, batch=8)
    img = torch.zeros(3, 40, 80)
    img[0, :, :40] = 1.0                                   # left half red, right half green
    img[1, :, 40:] = 1.0

    def segment(im):
        m = torch.zeros(2, 40, 80, dtype=torch.bool)
        m[0, :, :40] = True
        m[1, :30, 40:] = True                              # the bottom of the right half: no segment
        return m

    enc = lambda x: x.mean(dim=(2, 3))                     # noqa: E731  (mean colour as the embedding)
    grid, n = T.clip_regions(enc, segment, img, s)
    assert n == 2 and grid.shape[:2] == T.common_grid(40, 80, 0.25)
    left, right = grid[0, 0].numpy(), grid[0, -1].numpy()
    assert np.allclose(np.linalg.norm(left), 1, atol=1e-5) and np.allclose(grid[1, 0].numpy(), left, atol=1e-6)
    assert float(left @ right) < 0.99                      # the two segments keep different embeddings
    assert float(grid[-1, -1].abs().sum()) == 0.0          # bottom-right: nothing covers it
