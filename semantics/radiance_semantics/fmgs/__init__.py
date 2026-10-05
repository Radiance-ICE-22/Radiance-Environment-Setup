"""FMGS backend (Phase 4): a multi-resolution hash-grid feature field trained against the
frozen Gaussians of a splatfacto run, then baked into the same per-Gaussian table the lift writes.

    field.py    FeatureField: hash grid on normalised centres → CLIP (512) and DINO (384) heads
                (tiny-cuda-nn on the GPU; a pure-PyTorch twin for CPU tests and as a fallback)
    losses.py   0.2 · CLIP Huber (δ 1.25) + 0.8 · DINO L2 + 0.01 · pixel alignment
    train.py    the trainer: refined training cameras, Phase 1 teacher maps, gsplat feature
                rendering in 32-channel chunks; checkpoints, TensorBoard, an OOM fallback ladder,
                and a checksum proving the Gaussians were never modified
    bake.py     evaluate the field at every Gaussian in .splat order → unit rows for the table

Standalone rather than an `ns-train` plugin (decided 5 Oct): the splat is frozen, so training
needs no RGB images — only the cameras and the teacher maps Phase 1 already caches — and it
shares the lift's camera and rasterisation path exactly, which keeps the Phase 5 comparison fair.

    python -m radiance_semantics.fmgs.train --project-root <root> --scene backroom --steps 200 --out /tmp/smoke
(semantic_pipeline.py --backend fmgs runs it as the `fmgs` step, then `bake`.)
"""
