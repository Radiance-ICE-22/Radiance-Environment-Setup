# LUDVIG: Learning-free Uplifting of 2D Visual Features to Gaussian Splatting Scenes

**Authors:** Juliette Marrie, Romain Menegaux, Michael Arbel, Diane Larlus, Julien Mairal
(Univ. Grenoble Alpes / Inria / CNRS / Grenoble INP / LJK; NAVER LABS Europe)
**arXiv:** [2410.14462](https://arxiv.org/abs/2410.14462) (v5, 27 Jul 2025). **Project page:** https://juliettemarrie.github.io/ludvig
**Read for:** the training-free lift (our default `--backend lift`) and the DINOv2 graph diffusion behind our L-CD variant.

---

## 1. In one paragraph

LUDVIG takes a Gaussian Splatting scene that is already trained and gives every Gaussian a feature: a weighted average
of 2D features from all training views, weighted by the Gaussian's alpha-blending weight at each pixel. This is the
transpose of the rendering operator, normalised. It works for any 2D feature (DINOv2, SAM masks, CLIP), needs no
optimisation, and takes about 2 ms per image per feature dimension inside the CUDA rasteriser. A second contribution is
a **graph diffusion** over the Gaussians (k-NN in 3D, edges weighted by DINOv2 similarity). It turns coarse inputs such as
scribbles or CLIP relevancy into clean 3D segmentations. Results match or beat optimisation-based methods (SA3D,
SAGA, OmniSeg3D, LangSplat, OpenGaussian) at 5–10× less time.

## 2. Problem and motivation

- Earlier semantic splat and NeRF methods (LERF, FMGS, Feature-3DGS, LangSplat, …) **optimise** a 3D feature field by
  minimising a reprojection loss over all views. With ≥1M Gaussians and 512-d features this is slow and memory-hungry,
  so they compress the features: FMGS uses a hash grid, LangSplat a 512→3 autoencoder, Feature-3DGS a 1×1 conv decoder.
- LUDVIG asks whether a direct, parameter-free "inverse rendering" is enough. It is, and it is much faster.

## 3. Method

### 3.1 Rendering as a linear operator

For view (camera) `d` and pixel `p`, the ordered set of Gaussians `S_{d,p}` gives

```
Î_d(p) = Σ_{i∈S_{d,p}} c_i(d) · w_i(d,p),      w_i(d,p) = α_i(d,p) · Π_{j<i} (1 − α_j(d,p))
```

with `α_i` = opacity × the projected 2D Gaussian density at `p`.

Rendering view-independent per-Gaussian features `f = (f_i)` into all `m` training frames is linear:
**`F̂ = W f`**, where `W` holds every `w_i(d,p)`: row (d,p), column i.

### 3.2 Uplifting (the core equation)

```
f_i = Σ_{(d,p)∈S_i} w̄_i(d,p) · F_{d,p},     w̄_i(d,p) = w_i(d,p) / Σ_{(d,p)∈S_i} w_i(d,p)
```

In matrix form, **`f = D⁻¹ Wᵀ F`**, with `D` diagonal (row sums of `Wᵀ`, i.e. each Gaussian's total weight).

- `S_i` = every (view, pixel) pair the Gaussian contributes to. **Every pixel counts, not just the centre pixel**
  (contrast with Occam's LGS, which samples the centre pixel).
- So each feature is a **convex combination** of 2D pixel features. Appendix D.4 shows why this matters: GaussianEditor's
  mask uplifting normalises by the *count* of contributing pixels instead of the weight sum. That gives large, opaque
  Gaussians inflated values and wrong semantics, which is especially visible on specular scenes.
- `W` and `D` are never formed explicitly: the uplift reuses the forward rasteriser with colours replaced by features.

**Link to optimisation.** Minimising `L(f) = ‖F − W f‖²` is what distillation methods do. LUDVIG's `f = D⁻¹ Wᵀ F` equals
**one preconditioned gradient step from f = 0** (`f = −D⁻¹ ∇L(0)`). More iterations of `L(f)` did **not** improve the
features in their experiments.

**Gaussian filtering.** `β_i = Σ w_i(d,p)` (the normaliser) measures how much a Gaussian matters. They drop the half
of the Gaussians with the lowest β, with no qualitative or quantitative loss.

### 3.3 Graph diffusion with DINOv2

- Nodes: the `n` Gaussians. Edges: the k nearest neighbours by 3D centre distance (symmetrised: i∈N(j) or j∈N(i)).
- Edge weight: **`A_ij = S_f(f_i, f_j) · P(f_i)^{1/2} · P(f_j)^{1/2}`**
  - `S_f(f_i, f_j) = exp(−‖f_i − f_j‖² / (b · s_f²))` on ℓ2-normalised DINOv2 features. `s_f` = median pairwise ℓ2
    distance, `b` = bandwidth.
  - `P(f_i)` = a **unary term**: how similar node i is to the object of interest. It stops diffusion leaking into the
    background, so the number of steps can be arbitrary.
- Diffusion (a few steps of the power method, projecting `g₀` onto A's dominant eigenspace):
  ```
  g_{t+1} = A · g̃_t,     g̃_t = g_t / ‖g_t‖₂
  ```
- `g₀` = a coarse initial signal: an uplifted scribble or mask, or the CLIP relevancy of the top Gaussians.

### 3.4 How each 2D model is used

| Input | 2D maps | 3D use |
| --- | --- | --- |
| **SAM / SAM2** (multi-view segmentation) | Uplift the reference mask, reproject it to every view, take pixels above τ (0.4 SPIn-NeRF, 1 NVOS) as the prompt pool, sample 3 random point prompts 10× and average the SAM masks | Uplift the averaged SAM masks, render to the target view, threshold |
| **DINOv2 ViT-g with registers** | Sliding windows over overlapping crops → PCA reduction (~40 dims in the runtime table), **features re-weighted by the PCA eigenvalues** (the first components pick out the foreground) → upsampled to pixels | Uplift; segment by similarity to the reference foreground (cosine to the mean, or logistic regression on the reference frame); or graph diffusion |
| **CLIP OpenCLIP ViT-B/16** | **Multi-scale crop grid, scales average-pooled into one map (as in FMGS)** | Uplift; per-Gaussian LERF relevancy; optionally DINOv2 graph diffusion; optionally SAM prompts from the top-relevancy pixels |

### 3.5 Open-vocabulary details (appendix A.4)

- **Relevancy per Gaussian** (LERF's): `r_{i,q} = min_k exp(T f_i·φ_q) / (exp(T f_i·φ_q) + exp(T f_i·φ_cano^k))`,
  `T = 10`, canonical phrases "object", "things", "stuff", "texture". They compute it **on 3D Gaussians**, not pixels,
  so it can be used in 3D. *(This is exactly our `query.relevancy`.)*
- **Localisation:** render the relevancy, apply a 5×5 mean filter, take the argmax pixel.
- **Segmentation from relevancy:** rescale by the max, threshold at τ = 0.8, then Otsu.
- **Diffusion for CLIP:** `S_f` bandwidth b = 0.5. The unary `P_q` is a logistic regression (C = 0.001, balanced classes)
  on DINOv2 features. Positives = Gaussians whose rescaled relevancy is above 0.5 and passes Otsu, then `P_q = L_q^{1/b}`
  (b = 2, or 0.025 with SAM). `g₀` = an even tighter seed set (two more Otsu passes) so diffusion stays inside the
  object. All queries diffuse in parallel.
- **Cost:** k-NN for 600k Gaussians ≈ 30 s once per scene. Diffusion ≈ 1 s per query (18 s for the 14–21 LERF queries
  of a scene). Uplift of 100 images × 40 dims ≈ 9 s on an RTX 6000 Ada.

## 4. Results

**Multi-view segmentation (IoU)**

| | NVOS | SPIn-NeRF |
| --- | --- | --- |
| SA3D-TRF | 90.3 | 93.7 |
| SA3D-GS | 92.2 | 93.2 |
| SAGA | 92.6 | 93.4 |
| OmniSeg3D | 91.7 | 94.3 |
| **LUDVIG DINOv2 + diffusion** | **92.4** | 93.8 |
| LUDVIG SAM / SAM2 | 91.3 / 91.3 | 93.8 / 93.8 |

SPIn-NeRF ablation (Table 2): geometry-only reprojection 73.1 → single-view DINOv2 88.5 / SAM2 88.6 → uplift DINOv2
91.6 / SAM2 93.8 → uplift + diffusion (DINOv2) 93.8. **Uplifting is the big step; diffusion helps most on 360° scenes.**

**LERF localisation** (LangSplat's extended set, accuracy %): LERF 73.6, LangSplat 84.3, **LUDVIG 86.3**
(ramen 78.9, figurines 80.4, teatime 94.9, waldo 90.9).

**LERF segmentation (IoU)**

| Protocol | LERF | LangSplat | OpenGaussian | Dr. Splat | **LUDVIG** |
| --- | --- | --- | --- | --- | --- |
| 2D segmentation | 37.4 (45 min) | 51.4 (105 min) | – | – | **64.3 (10 min)** |
| 3D object selection | – | – | 43.1 (50 min) | 43.6 | **50.4 (10 min)** |

Ablation (Table 5, 2D protocol): neither SAM nor diffusion 33.6; diffusion only 50.4; SAM only 57.3; both 64.3.
**Diffusion alone adds +16.8 IoU over raw CLIP relevancy.**

**ScanNet semantic segmentation**, with OpenGaussian's training replaced by LUDVIG's uplift (≈3 min instead of ≈40 min),
same 2D maps: mIoU 19 / 15 / 10 classes = **33.9 / 37.4 / 46.4** vs OpenGaussian 24.7 / 30.1 / 38.3.

## 5. Limitations (the authors')

- Quality depends on the splat. Specular surfaces (Fork) leave semi-transparent floaters, and motion blur hurts.
  Learning features jointly with the splat might regularise the geometry.
- Diffusion costs inference time (~1 s per query), while LangSplat-style SAM-everything maps cost preprocessing
  (~24 s per image, ~80 min per scene).
- Diffusion on multi-part objects can leave uneven weights, so Otsu keeps one part. Close-together objects can merge.

## 6. How we use it (Radiance)

**What we took**
- **The lift equation, exactly** (`semantics/radiance_semantics/lift.py`):
  `f_i = Σ_{v,p} w_i(v,p) F_v(p) / Σ_{v,p} w_i(v,p)` over **all pixels**, as LUDVIG's `D⁻¹ Wᵀ F`. We get the numerator as
  the gradient of `Σ_p ⟨render(f), F_v⟩` at f = 0 and the denominator as the gradient of `Σ_p render(1)`: two backward
  passes through gsplat 1.0.0, chunked to 32 channels (its backward limit). Verified by `probe.py` on 5 Oct.
- **Never-seen Gaussians:** weight exactly 0 → row 0 and flagged unseen. LUDVIG prunes the bottom 50 % by β. We keep all
  Gaussians, because table rows must stay in `.splat` record order.
- **Relevancy:** the same LERF formula on 3D Gaussians (T = 10, the same four canonical phrases), `query.relevancy`.
- **CLIP teachers** started as LUDVIG/FMGS's scale-averaged pyramid ("Old", `lift_pyr`). Since 8 Oct the default is
  multi-scale A, which keeps three scale groups and picks one per query (SEMANTICS.md §15).

**What differs: our L-CD is a simplified LUDVIG diffusion** (`query.dino_graph` / `query.diffuse`)

| | LUDVIG | Ours (L-CD) |
| --- | --- | --- |
| Graph | k-NN in 3D, RBF kernel on DINOv2 distance, scaled by the median distance | 16-NN in 3D, edge = clipped DINO cosine |
| Leakage control | Unary term P from a logistic regression per query | None. Only nodes with relevancy > 0 update |
| Update | Power iteration `g ← A g / ‖g‖`, many steps | Convex mix `r ← (1−α) r + α · (weighted neighbour mean)`, α = 0.5, 2 iterations |
| Seed | A tight Otsu set of top-relevancy Gaussians | All relevancies |
| DINO maps | ViT-g/reg, sliding windows, PCA, eigenvalue-weighted | Our DINOv2 teacher grid (64 × 36 cells), 384-d |

Our finding (SEMANTICS.md §12–13): **L-CD changed no hit** (+50–70 ms per query). It removed one false positive and
one ambiguous result. Our lifted DINO rows are very smooth (mean neighbour cosine 0.945), so the graph has little
contrast. LUDVIG's version differs in three ways that could matter: an RBF kernel normalised by the median distance
(sharper than a raw cosine), a per-query unary term, and many more steps from a tight seed. **A faithful LUDVIG diffusion
is an untested option**, a candidate for the "query-time fusion" item in HANDOVER §2.

**Other ideas from the paper not yet tried**
- **Eigenvalue-weighted PCA of DINOv2** to sharpen DINO contrast before building the graph.
- **Otsu / automatic thresholds instead of a fixed floor.** This bears on our open floor-calibration question
  (HANDOVER §2 item 1).
- **SAM prompts from top-relevancy pixels** for crisper 2D masks. This is less relevant to us: we need a 3D goal, not a
  2D mask.
- **Pruning by β** could shrink tables (A's tables are 1.8–2.5 GB), but it breaks the row ↔ `.splat` record identity
  unless pruned rows are kept as zeros.
