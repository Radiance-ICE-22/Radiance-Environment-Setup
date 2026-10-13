# FMGS: Foundation Model Embedded 3D Gaussian Splatting for Holistic 3D Scene Understanding

**Authors:** Xingxing Zuo, Pouya Samangouei, Yunwen Zhou, Yan Di, Mingyang Li (Google)
**Venue:** IJCV 2024. **arXiv:** [2401.01970](https://arxiv.org/abs/2401.01970) (v2, 3 May 2024)
**Read for:** our second backend (`--backend fmgs`, `semantics/radiance_semantics/fmgs/`), the scale-averaged CLIP
pyramid ("hybrid CLIP") and the pixel-alignment ("dotsim") loss.

---

## 1. In one paragraph

FMGS adds CLIP and DINO features to a trained 3D Gaussian Splatting scene **without storing a feature per Gaussian**.
Instead, each Gaussian's mean queries a **multi-resolution hash encoding (MHE, iNGP-style)**, and small MLP heads decode
CLIP (512-d) and DINO features. The features are alpha-blended into 2D maps like colour and **distilled** against teacher
maps from the foundation models. The CLIP target is a **"hybrid" map: CLIP over a 7-scale crop pyramid, averaged into
one map.** A DINO head on the same grid plus a **pixel-alignment loss** pushes the CLIP feature's local similarity
pattern to follow DINO's sharp object boundaries. On LERF's localisation set it gets **93.2 %** vs LERF's 83.0 %
(+10.2 points), and renders 480×270 feature maps at 103 FPS vs LERF's 0.12 FPS (851× faster).

## 2. Motivation

- A 512-d feature per Gaussian multiplies the optimised parameters by **9.83×** (standard 3DGS, 58 params per
  Gaussian) or **65×** (a simplified parametrisation). Room-scale splats have millions of Gaussians.
- A hash grid has a **fixed parameter count** regardless of the number of Gaussians. Each Gaussian volume gets a single
  embedding, which encourages local semantic consistency.
- CLIP embeddings are image-level and not pixel-aligned. They are also not multi-view consistent. Distilling through
  rendering from many views enforces consistency.

## 3. Method

### 3.1 Feature field

For a Gaussian with mean `x`:

```
q = MHE_θ(x)                      # L levels × D features, concatenated (trilinear interpolation per level)
f̂ = MLP_φ^CLIP(q),  normalised to unit length
d̂ = MLP_ψ^DINO(q)                # shares the hash grid θ, separate head
```

Level resolution `N_l = N_min · b^l`.

**Hash grid used:** 24 levels, resolution 16 → 512, table size 2^20, 8 features per level. MLP heads follow LERF's.

### 3.2 Rendering features

The same α-blending as colour: `f̂_{i,j} = Σ_k f̂_k α_k Π_{l<k} (1 − α_l)`, implemented in the CUDA rasteriser. Training and
query feature maps are **480 × 270**.

### 3.3 Targets

- **Hybrid CLIP map:** OpenCLIP **ViT-B/16 (LAION-2B)** on an image pyramid of **7 crop scales from 0.05 to 0.5** of the
  image size (LERF's pyramid). The smaller scales are bilinearly upsampled to the largest map and **averaged**.
  Rationale: one map that carries every scale's context, so no scale search at query time (LERF searches 30 scales).
- **DINO map:** a single pass of the image through DINO, no pyramid.

### 3.4 Losses

```
L_CLIP  = Huber(F̂, F),  δ = 1.25           (0.5|·|² below δ, linear above)
L_DINO  = |D̂ − D|²
L_pixel = 1/(K²−1) · Σ_i Σ_{j∈N(i), j≠i} | d̂_iᵀ d̂_j − f̂_iᵀ f̂_j |     (unit-normalised features, K×K kernel)
L_total = λ L_CLIP + (1 − λ) L_DINO + γ L_pixel,      λ = 0.2, γ = 0.01
```

- `L_pixel` ("dotsim") makes the rendered CLIP map's centre-vs-neighbour similarities match DINO's. **Gradients are
  stopped through the DINO branch**, so it only shapes CLIP.
- All terms are averaged over pixels. `L_pixel` is **switched on after 2.5k steps** ("adding it later works best").

### 3.5 Training recipe

- Train vanilla 3DGS for 30k iterations (~10 min), then **freeze** its geometry and SH.
- Use **~40 % of Gaussians**: high opacity and a projected 2D radius > 2 px in at least one training view. Only these
  are rendered while training the semantic field.
- **RAdam**, weight decay 1e-9, exponential LR **5e-3 → 4e-3** over **4.2k steps**. About **1.4 h** on an RTX A5000 (24 GB).
- LERF scenes: poses re-estimated with COLMAP (the Polycam poses were noisy), test poses aligned by Sim(3).

### 3.6 Query

Render the CLIP map for a view and compute LERF's relevancy per pixel:
`r = min_n exp(f̂ᵀ f_q) / (exp(f̂ᵀ f_q) + exp(f̂ᵀ f_can^n))` with canonical phrases "object", "stuff", "things", "texture"
(no tuning needed). For segmentation they use the softmax over class-label cosine similarities instead (no canonicals).
The authors also note that features live on Gaussians, so queries can be localised in 3D directly.

## 4. Results

**LERF object localisation (accuracy; success = the argmax-relevancy pixel falls in the box)**

| Scene | FFD-LSeg | OWL-ViT | LERF | **FMGS** |
| --- | --- | --- | --- | --- |
| bouquet | 50.0 | 66.7 | 83.3 | 100.0 |
| figurines | 8.9 | 38.5 | 87.2 | 89.7 |
| ramen | 15.0 | 92.5 | 62.5 | 90.0 |
| teatime | 28.1 | 75.0 | 96.9 | 93.8 |
| kitchen | 13.0 | 42.6 | 85.2 | 92.6 |
| **Average** | 18.0 | 54.8 | 83.0 | **93.2** |
| Inference FPS | – | – | 0.1214 | 103.4 |

(This is the original 5-scene LERF set. LUDVIG and Occam's report on LangSplat's extended set, so the numbers are not
directly comparable.)

**Ablation (localisation average)**

| Variant | Avg |
| --- | --- |
| Full FMGS | **93.2** |
| without dotsim loss | 90.4 |
| without hybrid CLIP (finest single scale only) | **30.8** |
| with LERF's per-scale CLIP (scale as an input, 30-scale search at query) | 78.8 |
| without MHE (per-Gaussian embedding of the same size, same 40 % subset) | 84.4 |

**Scale averaging is by far the most important design choice in their setting.** Per-scale CLIP was worse *and* 30×
slower. They concede it might catch up with much longer training.

**3D-OVS segmentation (mIoU)**: FMGS bed 38.0, sofa 56.6, lawn 64.9, room 57.0, bench 62.1, table 63.6, beating LERF
everywhere (33.5 / 28.1 / 49.8 / 26.3 / 55.2 / 31.1). With SAM majority-vote refinement ("Ours-Refined"): 80.6–92.6.
3D-OVS itself (weakly supervised, knows the classes) remains higher on most scenes.

## 5. Limitations (the authors')

- Needs good, calibrated images (as NeRF methods do).
- Bounded by the foundation model. A better localising model would help.
- For segmentation, distilling SAM itself into 3D would be better than SAM post-refinement.

## 6. How we use it (Radiance)

**Our implementation:** `semantics/radiance_semantics/fmgs/` (`train.py`, `field.py`, `losses.py`, `bake.py`).
SEMANTICS.md §3e and §11 have the detail.

| Paper | Ours | Why |
| --- | --- | --- |
| A second stage after 30k-step 3DGS, geometry and SH frozen | Same idea, as a **standalone trainer** (`fmgs/train.py`) on the frozen FiGS splat; the Gaussians' checksum before and after must match | The planned `ns-train` plugin was dropped for this (Suhan, 5 Oct); a nerfstudio branch is planned separately |
| Hash grid 24 levels, 16→512, 2^20, F = 8 | Same ladder, but on sm_75 any 192-dim tcnn grid fails to launch → **two consecutive 12-level grids** (16→84, 98→512, each within 1 % of the paper's ladder) | RTX 2080 / tcnn limitation. A reported deviation |
| 40 % of Gaussians by opacity + radius > 2 px | Most opaque 40 %, then per view by **frustum** (centre projects within the image ± 10 %), not projected radius | Simpler with gsplat 1.0.0 |
| RAdam 5e-3 → 4e-3 | Adam 1e-2 → 1e-3 | Our choice (SEMANTICS.md §3e) |
| λ = 0.2, γ = 0.01, Huber δ = 1.25, dotsim after 2.5k | Same weights; fp16 with dynamic loss scaling and a 1e-3 norm floor in the pixel loss (a NaN fix) | 8 GB budget |
| 4.2k steps, 480×270 maps | Same; checkpoint every 1,000 steps (resumable, which saved us in the 6 Oct power cuts) | |
| Query = render a 2D map + argmax pixel | **Bake** the field at the Gaussian means into a per-Gaussian table, then the same 3D query as the lift (relevancy → voxel clusters → approach point) | We need a 3D goal for the drone, not a pixel |
| DINO feature dim (paper: DINO ViT) | DINOv2, 384-d head | Our shared teacher cache |

**What we measured (SEMANTICS.md §13–16):**
- FMGS **localises worse than the training-free lift** on our rooms: F-CD 7/13 backroom, 4/16 flightroom vs lift 10 and 10
  (amended sets). F-C (CLIP only, w_dino = w_pa = 0) 7 + 8.
- **Diagnosis: the loss balance.** With unit-norm 512-d CLIP targets, 0.2 × CLIP Huber is ~1e-4, against
  0.8 × DINO L2 ≈ 2. The shared grid is shaped almost entirely by DINO, so the CLIP head is spatially fragmented
  ("red cup" on F-CD: 61 clusters of ≤ 3 Gaussians vs one of 288 on F-C). The paper's weights presumably assume
  differently scaled targets. Rebalancing is the obvious untested experiment; it needs its own dev set.
- FMGS "no candidate" misses are **not a floor problem**: the high-relevancy Gaussians are scattered over the room
  (top-100 spread 5.8–7.5 m, ~⅓ outside the camera volume) because the field is queried at Gaussians it never
  supervised. This motivates the "bake only supervised Gaussians + query-time smoothing" proposal (HANDOVER §2 item 2.4).
- **Their key ablation does not transfer cleanly.** FMGS found that scale averaging beats per-scale CLIP by 14 points.
  Our §14 diagnostic found the scale-averaged pyramid costs **~0.07–0.09 relevancy** on small or partly visible objects.
  Variant A (three scale groups, chosen per query) improved the lift (20 → 22/29) but **hurt FMGS** (F-CD 11 → 8).
  So the hybrid map suits a learned field, while per-query scale choice suits the lift. A's advantage on the lift is
  itself partly a floor effect (§16).
- Cost on our host: F-CD 25–31 min, peak ~7.5–7.95 GB device memory on the 8 GB 2080. Lift 3–4 min.

**Open threads tied to this paper**
1. Rebalance λ (or normalise the CLIP and DINO losses by their initial magnitudes) and re-test F-CD on the dev set.
2. Bake only supervised (trainable-subset, in-frustum) Gaussians; smooth relevancy at query time.
3. B-variant teachers left uncovered cells at 0, which FMGS learned as targets. B2's pyramid fallback fixed that.
   A loss mask for uncovered pixels is the cleaner fix.
4. The "without MHE" ablation (84.4) suggests per-Gaussian features are not much worse. With a training-free lift
   that is our default.
