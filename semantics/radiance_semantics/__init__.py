"""radiance_semantics — semantic feature embeddings attached to trained FiGS Gaussian splats.

Phase 0 (environment and pose check) ships:

    python -m radiance_semantics.env_check   versions of the pinned stack; snapshot / compare
    python -m radiance_semantics.models      fetch and smoke-test OpenCLIP + DINOv2
    python -m radiance_semantics.probe       gsplat N-channel render-with-gradient probe
    python -m radiance_semantics.cameras     optimized-pose render check against training images

See docs/SEMANTICS_PLAN.md (phases) and docs/SEMANTICS.md (status and measured numbers).
"""

__version__ = "0.1.0"

# Foundation models, pinned here so every phase uses the same ones. Changing any of these
# changes the teacher-cache tag (Phase 1), so old features are never mixed with new ones.
CLIP_MODEL = "ViT-B-16"
CLIP_PRETRAINED = "laion2b_s34b_b88k"     # the LERF configuration; 512-d embeddings
CLIP_DIM = 512
DINO_REPO = "facebookresearch/dinov2"
DINO_MODEL = "dinov2_vits14"               # 384-d patch tokens, 14 px patches
DINO_DIM = 384
