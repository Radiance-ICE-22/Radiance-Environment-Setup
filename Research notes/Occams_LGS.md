# Occam's LGS: An Efficient Approach for Language Gaussian Splatting

**Authors:** Jiahuan Cheng, Jan-Nico Zaech, Luc Van Gool, Danda Pani Paudel
(Johns Hopkins University; INSAIT, Sofia University)
**arXiv:** [2412.01807](https://arxiv.org/abs/2412.01807) (v2, 8 Mar 2025). **Project page:** https://insait-institute.github.io/OccamLGS/
**Read for:** the probabilistic justification of the training-free lift (why a weighted average is the right
estimator), plus its speed and dimension numbers.

---

## 1. In one paragraph

Language Gaussian Splatting (LGS) methods such as LangSplat, LEGaussians, GOI, FMGS and Feature-3DGS train a language
feature per Gaussian (or a field) by rendering features and back-propagating a loss. To make that tractable they
compress the features. Occam's LGS applies Occam's razor: treat uplifting as **maximum-likelihood estimation** under the
Gaussian Splatting forward model. Under mild assumptions the ML solution is closed form: **each Gaussian's feature is the
alpha-weight-weighted average of the 2D features at the pixels it projects to.** Computing it takes one forward
rendering pass over the training views. 512-d CLIP features are lifted with no compression in **16.4 s** (31.9 s with
checkpoint I/O), about two orders of magnitude faster than LangSplat/GOI/3DVLM (60–67 min), with state-of-the-art
mIoU on LERF and 3D-OVS.

## 2. Method

### 2.1 Feature rendering (the forward model)

Each Gaussian gets a feature `f_i ∈ R^d`. A pixel's feature is rendered like colour:

```
F(p) = Σ_{i=1..R} α_i f_i Π_{j<i} (1 − α_j) = Σ_i w_i f_i,     α_i = o_i · exp(−½ (p − x'_i)ᵀ Σ'⁻¹ (p − x'_i))
```

Rendering explicit d ≫ 3 features is expensive: the data no longer fits the shader L1 cache. That is why LangSplat
compresses to 3 dims.

### 2.2 MAP → ML → closed form (supplementary §6)

Assumptions:
1. **Uniform prior** p(f) ∝ const: CLIP space is used "efficiently" by a model trained on huge data → MAP = ML.
2. **Pixels are conditionally independent** given the splat and the features.
3. **Gaussian feature noise:** `f^{2D}_{p,v} = Σ_k w_{k,p,v} f_k + ε_{p,v}`, `ε ~ N(0, σ² I)`. This models CLIP's
   view-dependent noise.

The log-likelihood is a least-squares problem, `−1/(2σ²) Σ_{v,p} ‖f^{2D}_{p,v} − Σ_k w_{k,p,v} f_k‖²`. Setting
`∂/∂f_i = 0` gives the normal equations:

```
Σ_{v,p} w_{i,p,v} f^{2D}_{p,v}  =  Σ_k Σ_{v,p} w_{i,p,v} w_{k,p,v} f_k
```

Two simplifications that hold for a trained splat, where usually one dominant Gaussian covers a pixel:
- **(a) negligible cross-talk:** `w_i w_k ≈ 0` for i ≠ k, so the system decouples per Gaussian.
- **(b) linearisation:** weights are nearly binary, so `Σ w² ≈ Σ w`.

Result, the estimator they use:

```
f_i = Σ_v Σ_{p∈P_v} w_{i,p,v} f^{2D}_{p,v}  /  Σ_v Σ_{p∈P_v} w_{i,p,v}          (Eq. 19, supplementary)
```

### 2.3 What they actually implement (Algorithm 1, Eq. 7)

They approximate further: **only the pixel under the Gaussian's projected centre** is used per view.

```
f_i = Σ_{s∈S_i} w_i^s f_i^s / Σ_{s∈S_i} w_i^s
```

- `S_i` = views where Gaussian i is in the frustum.
- `p_i^s = ⌊x'_i⌋` = the centre's pixel. `w_i^s` = its alpha-blending weight there. `f_i^s` = the 2D feature at that pixel.
- The weights and projections are recorded during a normal tile-based forward render (gsplat). Frustum culling and
  depth sorting handle occlusion.
- Occluded Gaussians with negligible influence (`w_i^s < 10⁻⁴`) are thresholded out.

### 2.4 Filtering

After the pass, **Gaussians with no contribution in any view are removed** (M ≪ N semantic Gaussians). Reason: the
centre-pixel update is sparse. An occluded Gaussian can still affect neighbouring pixels at render time without ever
receiving a feature update, which creates semantic noise.

## 3. Setup

- gsplat; 30k-iteration 3DGS, ~2.5M Gaussians per scene; single A6000.
- 2D features: **LangSplat's pipeline**: SAM ViT-H masks at three levels (whole / part / subpart), OpenCLIP ViT-B/16
  embedding per mask. Each level is uplifted separately.
- LERF evaluation: relevancy on rendered maps, mask threshold 0.5. 3D-OVS: per-query **dynamic threshold**, the middle
  of the "stable region" where the mean relevancy inside the mask changes least with the threshold (|gradient| below a
  stability threshold of 0.4). Masks with area < 0.005 % or > 0.90 % are rejected.

## 4. Results

**LERF open-vocabulary segmentation (mIoU %)**

| | Ramen | Figurines | Teatime | Waldo Kitchen | Overall |
| --- | --- | --- | --- | --- | --- |
| LERF | 28.2 | 38.6 | 45.0 | 37.9 | 37.4 |
| GOI | 52.6 | 63.7 | 44.5 | 41.4 | 50.6 |
| LangSplat | 51.2 | 44.7 | 65.1 | 44.5 | 51.4 |
| **Occam's LGS** | 51.0 | 58.6 | **70.2** | **65.3** | **61.3** |

The gain is largest on **Waldo Kitchen**, a whole room with sparse view coverage. That is the closest LERF scene to our
backroom and flightroom.

**LERF localisation accuracy (%)**: Occam's 82.5 (ramen 74.7, figurines 80.4, teatime 93.2, waldo 81.8) vs LangSplat 84.3,
GOI 84.4. It is competitive, not best: argmax-pixel localisation is sensitive to noise. With 32-d autoencoder
compression it reaches 84.1.

**3D-OVS (mIoU %)**: Occam's 95.0 overall vs LangSplat 93.4, GOI 90.6.

**Feature dimension** (autoencoder → lift → render → decode; LERF mIoU): 16-d 60.5, 32-d 64.1, **64-d 64.5**,
128-d 62.9, 256-d 62.7, 512-d 61.3. Compression filters noise by restricting features to what is in the scene, at the
cost of scene-specific, non-transferable features.

**Ablation (LERF mIoU overall):** full 61.3; without weighted aggregation 59.9; without filtering 58.8; without both
(plain back-projection) 57.1. **Weighting and filtering together are worth +4.2.**

**Cost:** under 14 GB GPU memory for 512-d on up to 4M Gaussians. Runtime scales with Gaussians and frames. Render FPS
and storage scale with the feature dimension.

**Application: object insertion.** Copy an object's Gaussians and features from one scene into another, and it stays
queryable ("pink ice-cream"). With scene-specific autoencoders (LangSplat) this would need retraining.

## 5. Their conclusions

- Is feature compression or a spatial map (hash grid) necessary for LGS? **No.**
- Should low-dimensional LGS be abandoned? **No.** It saves memory and render time and filters noise, but it is
  scene-specific.
- The method is complementary to compression and can be combined with it.

## 6. How it relates to Radiance

**Our lift is the full-pixel form (Eq. 19), not their centre-pixel implementation (Eq. 7).**
`semantics/radiance_semantics/lift.py` sums `w_i(v,p) F_v(p)` over **every pixel** each Gaussian touches, computed as the
gradient of the rendered inner product (as in LUDVIG). So:
- Occam's §6 derivation is the **theoretical justification** we can cite for our lift: the ML estimate under Gaussian
  noise, given negligible cross-talk and near-binary weights.
- We don't need their filtering step for correctness. A Gaussian that affects any pixel receives that pixel's feature,
  so the "update/render mismatch" they filter against does not arise. Never-seen Gaussians get weight 0 and a zero row.
  We keep them because table rows mirror `.splat` records.
- The centre-pixel shortcut would be faster (no backward pass, any channel count in one go), which matters on the 8 GB
  2080. But the full-pixel sum is more faithful to the ML derivation, and our lift is already only 3–4 min (Old) or
  17–23 min (A, four 512-ch maps). The idea to remember if the lift time ever matters (e.g. on the 5060 Ti with
  bigger scenes): centre-pixel sampling during a forward pass, no 32-channel backward chunking.

**Where their assumptions are weak for our scenes**
- **(a)/(b) assume one dominant Gaussian per pixel.** On thin or semi-transparent structures (tripods, floor lamp,
  keyboard: our persistent misses, SEMANTICS.md §14), and on floaters, many Gaussians share a pixel. There the
  closed form is an approximation, and LUDVIG's remark (more optimisation steps did not help) is the empirical answer.
- **Uniform-prior assumption:** our floor sweep shows peak relevancy depends on the teacher and scoring rule (A's max
  over scale groups lifts all peaks by ~0.045). This is a calibration issue, not a lift issue, but it is the same
  "CLIP-space statistics" question.

**Ideas worth taking**
- **The dynamic "stable region" threshold** (3D-OVS appendix) is a per-query threshold chosen without ground truth.
  It could feed into our open floor-calibration decision (HANDOVER §2 item 1) as an alternative to one global floor.
  Caveat: it would have to be validated on the dev set, not the frozen evaluation set.
- **Per-level SAM/CLIP maps (whole / part / subpart)** are what our B/B2 variant does (`clip_s0..2`, SAM split by area).
  Occam's uses LangSplat's hierarchy, which comes from SAM's own multimask granularity, not from area.
- **Low-dimensional compression helped them (+3.2 mIoU at 64-d).** That is an unexplored axis for us: a scene autoencoder
  on lift rows. It would also cut A's 1.8–2.5 GB tables.
