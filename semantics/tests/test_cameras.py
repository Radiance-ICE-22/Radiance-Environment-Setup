import math

import numpy as np
import pytest

from radiance_semantics.cameras import camera_from_transforms, spread


def tf_two_frames():
    m0 = np.eye(4).tolist()
    m1 = np.eye(4)
    m1[:3, 3] = [1.0, 2.0, 3.0]
    return {"fl_x": 1000.0, "fl_y": 1010.0, "cx": 960.0, "cy": 540.0, "w": 1920, "h": 1080,
            "frames": [{"file_path": "images/frame_00002.png", "transform_matrix": m1.tolist()},
                       {"file_path": "images/frame_00001.png", "transform_matrix": m0}]}


def test_frames_sorted_and_intrinsics_rescaled():
    c2w, K = camera_from_transforms(tf_two_frames(), {}, 1, 960, 540)
    assert np.allclose(c2w[:3, 3], [1, 2, 3])                     # frame_00002 is second in file_path order
    assert np.allclose(K, [[500, 0, 480], [0, 505, 270], [0, 0, 1]])


def test_dataparser_transform_then_scale():
    R = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    T = np.hstack([R, np.array([[0.5], [0.0], [0.0]])])
    c2w, _ = camera_from_transforms(tf_two_frames(), {"transform": T.tolist(), "scale": 2.0}, 1, 1920, 1080)
    assert np.allclose(c2w[:3, :3], R)
    assert np.allclose(c2w[:3, 3], 2.0 * (R @ [1, 2, 3] + [0.5, 0, 0]))


def test_per_frame_intrinsics_override():
    tf = tf_two_frames()
    tf["frames"][1]["fl_x"] = 2000.0
    _, K = camera_from_transforms(tf, {}, 0, 1920, 1080)
    assert K[0, 0] == 2000.0


def test_spread():
    assert spread(100, 5) == [0, 25, 50, 74, 99]
    assert spread(3, 10) == [0, 1, 2]
    assert spread(1, 4) == [0]


torch = pytest.importorskip("torch")


def test_psnr_and_correction_stats():
    from radiance_semantics.cameras import correction_stats, psnr
    a = torch.zeros(4, 4, 3)
    assert math.isclose(psnr(a, a + 0.1), 20.0, rel_tol=1e-5)
    th = math.radians(2.0)
    corr = torch.zeros(2, 3, 4)
    corr[:, :3, :3] = torch.eye(3)
    corr[1, :3, :3] = torch.tensor([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1.0]])
    corr[1, :3, 3] = torch.tensor([0.003, 0.004, 0.0])
    s = correction_stats(corr)
    assert math.isclose(s["translation_max_m"], 0.005, rel_tol=1e-5)
    assert math.isclose(s["rotation_max_deg"], 2.0, rel_tol=1e-3)
    assert math.isclose(s["rotation_mean_deg"], 1.0, rel_tol=1e-2)


def test_optimized_c2w_applies_nerfstudio_correction():
    """With real nerfstudio objects: our helper must give exactly splatfacto's training-time pose."""
    pytest.importorskip("nerfstudio.cameras.camera_optimizers")
    from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
    from nerfstudio.cameras.cameras import Cameras
    from radiance_semantics.cameras import optimized_c2w

    c2w = torch.eye(4)[None, :3, :].repeat(3, 1, 1)
    c2w[:, :3, 3] = torch.tensor([[0.0, 0, 0], [1, 2, 3], [-1, 0, 2]])
    cams = Cameras(camera_to_worlds=c2w, fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    opt = CameraOptimizerConfig(mode="SO3xR3").setup(num_cameras=3, device="cpu")
    with torch.no_grad():
        opt.pose_adjustment[1] = torch.tensor([0.01, -0.02, 0.03, 0.0, 0.0, 0.05])

    class M:                                   # the two attributes optimized_c2w uses
        device = torch.device("cpu")
        camera_optimizer = opt

    got = optimized_c2w(M, cams, 1)
    adj = opt(torch.tensor([1]))[0]
    want_R = adj[:3, :3] @ c2w[1, :3, :3]
    want_t = c2w[1, :3, 3] + adj[:3, 3]
    assert torch.allclose(got[:3, :3], want_R, atol=1e-6) and torch.allclose(got[:3, 3], want_t, atol=1e-6)
    assert torch.allclose(optimized_c2w(M, cams, 0), c2w[0], atol=1e-6)        # zero adjustment → raw pose
    assert not torch.allclose(got, c2w[1])

    one = cams[1:2]
    one.camera_to_worlds = got[None]           # render_rgb swaps the pose like this
    assert torch.allclose(one.camera_to_worlds[0], got)
