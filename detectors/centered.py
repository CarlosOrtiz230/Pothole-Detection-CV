import numpy as np
from scipy.ndimage import sobel, gaussian_filter
from utils.preprocess import to_gray

CFG = dict(
    center_frac        = 0.30,   # slightly larger ROI
    base_dark_ratio    = 0.60,   # old rule (kept)
    min_dark_area      = 0.03,   # 3 % pixels dark
    allow_low_texture  = 3,      # σ≤3 allowed IF rim cues fire
    rim_width_px       = 10,     # pixels for rim band
    rim_grad_thr       = 28,     # Sobel mean on rim band
    center_rim_delta   = 15,     # center brighter by ≥15 gray
    outer_inner_ratio  = 1.10,   # outer_mean / inner_mean ≥1.1
    circular_ok_if_rim = 1.05,   # circularity allowed if rim cues pass
    bonus_rim_detect   = 0.25,   # extra confidence when rim cues hit
)

def detect(img, cfg=None, debug=True):
    p = {**CFG, **(cfg or {})}
    g = to_gray(img)
    h, w = g.shape
    side = int(min(h, w) * p['center_frac'])
    x0, y0 = (w - side) // 2, (h - side) // 2
    roi = g[y0:y0+side, x0:x0+side]

    if debug:
        print(f"[INFO] ROI shape: {roi.shape} at (x={x0}, y={y0}), size={side}")

    # --- Darkness ---
    thr = roi.mean() * p['base_dark_ratio']
    dark = roi < thr
    dark_frac = dark.mean()

    if debug:
        print(f"[DARKNESS] ROI mean: {roi.mean():.2f}")
        print(f"[DARKNESS] Threshold: {thr:.2f}")
        print(f"[DARKNESS] Dark fraction: {dark_frac:.3f} (min required: {p['min_dark_area']})")

    # --- Texture ---
    sigma = roi.std()
    if debug:
        print(f"[TEXTURE] Std Dev (σ): {sigma:.2f} (min allowed: {p['allow_low_texture']})")

    # --- Rim and Edge Energy ---
    rw = p['rim_width_px']
    rim = roi.copy()
    rim[rw:-rw, rw:-rw] = 0
    rim_grad = sobel(rim.astype(float)).mean()

    if debug:
        print(f"[RIM] Rim gradient mean: {rim_grad:.2f} (threshold: {p['rim_grad_thr']})")

    # --- Center vs Rim Brightness Delta ---
    inner = roi[rw:-rw, rw:-rw]
    rim_px = rim[rim > 0] if rim[rim > 0].size else np.array([roi.mean()])
    delta_cr = inner.mean() - rim_px.mean()

    if debug:
        print(f"[RIM CONTRAST] Inner mean: {inner.mean():.2f}")
        print(f"[RIM CONTRAST] Rim mean: {rim_px.mean():.2f}")
        print(f"[RIM CONTRAST] ΔCR (inner - rim): {delta_cr:.2f} (required: ≥ {p['center_rim_delta']})")

    # --- Outer vs Inner Brightness Ratio ---
    blur = gaussian_filter(g.astype(float), 5)
    outer = blur.mean()
    ratio_outer_inner = outer / (inner.mean() + 1e-6)

    if debug:
        print(f"[OUTER RATIO] Blur outer mean: {outer:.2f}")
        print(f"[OUTER RATIO] Inner mean: {inner.mean():.2f}")
        print(f"[OUTER RATIO] Ratio outer/inner: {ratio_outer_inner:.2f} (required: ≥ {p['outer_inner_ratio']})")

    # --- Aspect Ratio / Circularity ---
    ys, xs = np.where(dark)
    circular_ok = False
    asp = 0.0
    if xs.size:
        a = max(xs.max()-xs.min()+1, ys.max()-ys.min()+1)
        b = min(xs.max()-xs.min()+1, ys.max()-ys.min()+1)
        asp = a / (b + 1e-6)
        circular_ok = asp < p['circular_ok_if_rim']

    if debug:
        print(f"[CIRCULARITY] Aspect ratio: {asp:.2f} (circular_ok={circular_ok})")

    # --- Wet Rim Cue Hit? ---
    rim_hit = (
        (rim_grad > p['rim_grad_thr'] and delta_cr >= p['center_rim_delta']) or
        (ratio_outer_inner >= p['outer_inner_ratio'])
    )

    if debug:
        print(f"[RIM HIT] Rim hit: {rim_hit}")

    # --- Final Confidence ---
    conf = dark_frac
    if rim_hit:
        conf += p['bonus_rim_detect']
    conf = float(np.clip(conf, 0.0, 1.0))

    if debug:
        print(f"[CONFIDENCE] Final confidence: {conf:.2f}")

    # --- Rejection Logic ---
    if sigma < p['allow_low_texture'] and not rim_hit:
        if debug:
            print("[REJECTED] Texture too flat and no rim cues.")
        return 0.0, (x0, y0, side, side)

    if not circular_ok and not rim_hit:
        if debug:
            print("[REJECTED] Not circular enough and no rim cues.")
        return 0.0, (x0, y0, side, side)

    return conf, (x0, y0, side, side)
