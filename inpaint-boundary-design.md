# Inpaint boundary artifact — design & tracking notes

Status: **RESET to accepted baseline `790ff1e`** (working tree clean; 29 Rust + 30
Python tests pass). Approach A discarded as "demostrably worse". This file is a
living design/tracking note for the next attempt. **Nothing here is committed.**

---

## 1. Objective

Inpainting over `boundary_fill` must give a masked region that is (a) seamless at
the mask boundary (no rim / dipole / frozen grain), and (b) retains the real
structure/texture of the image (not a smooth flat disk).

The accepted baseline satisfies the *interior* well but leaves a **frozen grainy
rim** in the ~1-6 px boundary band. Every attempt so far has traded one failure
for another. This note records what we learned so the next attempt targets the
root cause.

## 2. Ground-truth laws established by experiments (not guesses)

- **The default `inpaint_mask` operator is not a smoothing operator.** With
  `first=second=third=0, fourth=1.0`, the *only* term that evolves masked pixels
  is the 4th-order edge-*enhancer* (term 3) plus the `hf * strength` source.
  There is no active low-frequency diffusion term in the mask at all.
- **The 4th-order enhancer amplifies a sharp transition into a growing
  dipole/ring.** On a clean step with the real parameters, overshoot grows
  `0.045 -> 0.376` over iterations (measured, no mask, no fill).
- **The 2b taper suppresses that enhancer near the boundary** (hard zero for
  d<=4, smooth ramp over `taper_dist`) and thereby fixes the dipole — but it
  simultaneously *pins* the fill's grain there (frozen rim). Dipole and rim are
  two halves of the same coin: both come from the enhancer + hf re-inject at the
  boundary band.
- **The B-spline low-pass excludes masked pixels** → `lf == 0` deep inside the
  mask, so masked values ride entirely through `hf` (`hf = in - blur = in`
  there). Any "reconstruct from lf" idea must first make `lf` span the mask.
  **[SUPERSEDED by v1.1 below** — the low-pass now includes masked pixels, so
  `lf` is a smooth field spanning the mask and is no longer 0 deep inside.]
- **The interior problem is structure, not additive noise.** Bolting texture on
  at the end (post-hoc noise injection) is rejected; the masked region must
  *carry* the real multi-scale structure.

## 3. Attempt log (each reversed / rejected)

| attempt | change | result | verdict |
|---|---|---|---|
| 1. baseline | windowed-mean fill + 2b derivative taper | interior good; frozen grainy rim 1-6px (b1-4 ~0.0028, b4-6 ~0.0035 frozen) | **accepted** (checkpoint `790ff1e`) |
| 2. source suppression | zero `hf*strength` source in d<=4 (hard 0→1) | dipole on inner edge of 4px ring; noise init wrong even at iter 0 | worse — rejected |
| 3. Laplace/harmonic init (B) | replace windowed mean with SOR harmonic extension | boundary-neutral (same b1-4 0.0028); interior over-smoothed; **ring returns (0.018) if taper removed** | init is not the lever — rejected |
| 4. approach A | masked = `lf` + mask-inclusive low-pass | **flat disk** (interior std 0.002 vs bg 0.0042; deep hi 0.00005 vs 0.0033); broke noise-init semantics; required core decomposition change | worse — rejected |

### What #2-4 each taught us
- #2: a **hard 0→1 discontinuity in the update operator** re-seeds a dipole at
  that inner edge — any near-boundary modification must be smooth (C1), never
  a step.
- #3: the artifact is **diffusion-side** (the enhancer acting on the fill at the
  boundary), not an initialization low-frequency-step problem.
- #4: dropping the hf carrier entirely yields an **instantaneous full low-pass**
  = flat disk. A real fix must *diffuse* (dissipative, structure-preserving),
  not low-pass in one shot.

## 4. New lead: the zeroed terms (this is the main open idea)

`output[masked] = hf*strength + (Σ_k derivatives[k]*abcd[k])/variance + lf`

Where:
- `derivatives[0]` = kernel on **lf**, gradient-aligned   (coeff `first`)
- `derivatives[1]` = kernel on **lf**, isophote/edge-aligned (coeff `second`) ← Laplacian of lf
- `derivatives[2]` = kernel on **hf**, gradient-aligned   (coeff `third`)
- `derivatives[3]` = kernel on **hf**, isophote-aligned   (coeff `fourth`)
- denominators `variance = Σ hf²` (normalized)

`inpaint_mask` defaults zero `first/second/third`, leaving only term 3 (the
enhancer) + source. `diffuse_gray_image` (same code, real-image sharpening) uses
a **balanced full set** (`first=0.0065, second=-0.25, third=-0.25, fourth=-0.2774`,
all anisotropies 1.0) and behaves as a genuine multi-scale diffusion.

**Hypothesis:** much of the boundary pathology is because the masked-update is a
degenerate single enhancer instead of a balanced diffusion. In particular the
**lf-Laplacian term (`second`)** is the missing *stable, structure-preserving,
dissipative* smoother that could denoise the fill (→ kill the frozen rim) and
stabilize the boundary (→ no dipole), while preserving mid-scale structure (→ not
a flat disk).

### Known interaction to respect
Term 0/1 act on `lf`; `lf` is 0 deep in the mask (see law above). So for an
lf-based smoother to work *inside* the mask, `lf` must span the mask (mask-
inclusive low-pass) — the same decomposition change #4 made. On its own that
flat-disks (because #4 removed the hf carrier), but combined with a **kept,
structure-carrying hf term + a genuine lf smoother** it may give both.

### Experiment: naive enabling of the other terms does NOT help (measured)
Ran `inpaint_mask` with several coefficient sets on `data.npy` (iter 8 -> 32;
b1-4 shown as `8>32` high-freq):

| set | b1-4 | b4-6 | deep(>13) | fill std |
|---|---|---|---|---|
| baseline (4th only) | 0.0024>0.0024 (frozen) | 0.0036>0.0034 | 0.0008 | 0.010 |
| +second (keep 4th) | 0.068>0.196 | 0.004-0.005 | 0.0008 | 0.098 |
| second only | 0.068>0.196 | 0.0042>0.0056 | 0.004 | 0.098 |
| iso-second only | 0.053>0.138 | 0.004-0.005 | 0.004 | 0.077 |
| diffuse-style balanced | 0.021>0.068 | 0.004>0.012 | **3.7 (diverged)** | 1.84 |

**Conclusion:** enabling `first/second/third` as-is does NOT fix the boundary —
it *worsens* b1-4 (overshoot from the lf-Laplacian acting on the windowed-mean
lf transition at the boundary) and the balanced/diffuse-style set diverges (fill
std 1.8). The "missing smoother" is **not** a turnkey parameter change. The
lf-terms would need (a) `lf` to be a genuinely smooth boundary-continuous field
(mask-inclusive low-pass) and (b) coefficients re-tuned to this algorithm's
`/variance` normalization and per-scale `abcd`/`norm` weighting. So the other
terms are relevant in principle but not a drop-in fix.

### Safety principles (from the attempts)
1. Near-boundary operator changes must be smooth (C1) ramps — never hard 0→1.
2. A real fix *diffuses* gradually; it must not be an instantaneous low-pass.
3. The masked region must carry the fill's structure (not noise-only).
4. Do not break the `diffuse_gray_image` (no-mask) path or the exact
   reconstruction (`output = hf + lf`) of unmasked pixels.

## 5. Prototype design — refined v1 (being built)

Goal recap: preserve the accepted interior (baseline already gives a good, textured
interior that decays), fix only the frozen grainy rim (1-6 px) without a dipole
and without flattening the interior.

Two coordinated changes:

**v1.1 — mask-inclusive low-pass.** `decompose_2d_bspline` /
`_bspline_vertical_pass` / `_bspline_horizontal` stop excluding masked pixels
from the blur: every pixel (masked & unmasked) contributes to `lf`. So `lf` is a
smooth, boundary-continuous field spanning the mask (fill values included; never
0 deep inside). Unmasked exact reconstruction `output = hf + lf = in` is
unchanged. This is the same decomposition change approach A used, kept here.

**v2 — root-cause fix (where v1.2 led): the fill clamp.** While validating the
v1.2 blend in the v1.1 regime we found the *real* defect: the masked
reconstruction line clamped with `.max(0)` (`output = ... .max(0)`), and this
**pre-existing** clamp was flooring every negative pixel to 0. The surrounding
(unmasked) pixels are never clamped (they take `output = hf + lf`), so they
legitimately span values like p1 = −0.026 … p99 = 0.024 around ~0; the fill,
clamped at 0, lost its entire negative half → a homogeneous constant ring.
Robust scale said it all: fill MAD ≈ 0 while surrounding MAD ≈ 0.011.

The final solution is simple and removes *both* v1.2 (the blend) and approach 2b
(the taper, and with it `distance_to_unmasked`):

```
output[masked] = lf + acc            // acc = hf*strength + (Σ derivatives[k]*abcd[k])/variance
output[unmasked] = hf + lf           // unchanged
output[maskless general diffusion] = (lf + acc).max(0)   // legacy clamp preserved
```

Measured on `data.npy` (iter 16, but stable to it=64), `mask.npy`:
- fill rim std ≈ 0.011 vs surrounding ≈ 0.012, MAD ≈ 0.0118 vs 0.0107 — **near-
  exact structure/statistics match** (baseline was ~0.005 / MAD≈0).
- No dipole/ring: FLAT masked maxdev ~7e-7; ZERO masked ~6e-7.
- No runaway: fill range −0.031 … 0.023 at it=64.
- Seam mean jump ≈ −0.002 (boundary-continuous).
- `boundary_fill` single-pixel fill is seed-stable at the background level.

So: keep **v1.1** (mask-inclusive `lf`), **drop** the v1.2 blend, **drop** the 2b
taper, and **drop the clamp for the masked fill only** (preserve it for the
maskless path). The v1.2 blend is removed because it flattens structure
(blend → std 0.0013 even without the clamp); the taper is removed because it was
only a dipole guard for the old (pre-v1.1) regime, and v1.1 + no-clamp has no
dipole.

Physics: with `lf` spanning the mask (v1.1), the operator's existing
multi-scale hf/lf reconstruction — no longer clipped — reproduces the nearby
structure. No special boundary band is needed.

### Validation experiment (do this FIRST, cheap)
Before committing to a rewrite, test the hypothesis directly: call
`inpaint_mask(..., first=..., second=..., third=..., fourth=..., *_anisotropy=...)`
with a balanced/positive-diffusion set on `data.npy`/`mask.npy` and measure:
- frozen band: does b1-6 now decay with iterations?
- interior structure: is fill std / deep high-freq close to background (not
  flat)?
- boundary: seam vs bg-ring, and no dipole.

This isolates whether enabling the other terms (not a code rewrite) already
confers most of the benefit.

## 6. Metrics (reference values on `data.npy`, iter 32, seed 42)

High-frequency = mean |adjacent difference| over the masked-distance band
(`dist` = EDT from unmasked).

| band | baseline (frozen) | approach A (flat) | target |
|---|---|---|---|
| b1-4  | 0.0028 (frozen) | 0.00023 | decays, ≈ bg-ring |
| b4-6  | 0.0035 (frozen) | 0.00021 | decays |
| deep (>13) | ~0.0008 | 0.00005 | keep structure |
| bg (>8 unmasked) | 0.0033 | 0.0033 | reference |
| fill std / bg std | — | 0.002 / 0.0042 | match |

## 7. Current state
- `HEAD` = `790ff1e` (accepted baseline); working tree has the uncommitted v2 fix
  (v1.1 + no taper + no blend + no fill clamp). `data.npy`/`mask.npy` are local
  test data; this note itself is an untracked file.
- **Resolved:** the "frozen grainy rim" and the "flat homogeneous ring" were the
  *same* defect, and its root cause was the fill's `.max(0)` clamp flooring
  negatives (plus, secondarily, `lf == 0` in the mask before v1.1). With v1.1 +
  no clamp, the existing operator reproduces the surrounding structure.
- Next: full suite clean (28 Rust + 30 Python), final review, then commit.
