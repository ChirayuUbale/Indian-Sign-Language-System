"""
Preprocessing experiments for two-handed sign recognition (noisy backgrounds).

Usage:
    pip install opencv-python mediapipe numpy
    python preprocess_experiments.py --input dataset/ --output out/ --pipelines all --preview

Input layout:   dataset/<CLASS>/image.jpg   (or a flat folder -> label "unlabeled")
Output layout:  out/<pipeline>/<CLASS>/image.png, out/overlay/..., out/preview/...,
                out/failures.csv, out/landmarks.csv (P9)

Pipelines
    P0  baseline: grayscale + resize (no segmentation)
    P1  author's: HSV threshold -> blur -> largest contour -> median/morph (silhouette)
    P2  author's + Canny
    P3  CLAHE -> MediaPipe crop -> grayscale                 (cheap background removal)
    P4  hull+GrabCut+adaptive-skin mask -> CLAHE -> masked RGB
    P5  binary silhouette from the same mask
    P6  Canny edges restricted to the mask (no boundary artifact)
    P7  3-channel stack: [masked gray, edges, mask]
    P8  landmark skeleton on a black canvas
    P9  normalised landmark coordinates (CSV, for an MLP/SVM; no image)

Note: uses the legacy mediapipe `mp.solutions.hands` API (works on mediapipe 0.10.x).
"""
import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HAND_CONNECTIONS = list(mp.solutions.hands.HAND_CONNECTIONS)


# ----------------------------------------------------------------------------
# Lighting / enhancement
# ----------------------------------------------------------------------------
def clahe_lab(img, clip=2.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def gamma_correct(img, g):
    table = np.array([(i / 255.0) ** (1.0 / g) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(img, table)


def for_detection(img):
    """Enhanced COPY used only to help MediaPipe find hands in dark/blurry frames."""
    out = clahe_lab(img)
    if out.mean() < 80:
        out = gamma_correct(out, 1.6)
    return out


def limit_side(img, max_side):
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    f = max_side / s
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------
# Hand detection
# ----------------------------------------------------------------------------
def make_detector():
    return mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3
    )


def detect_hands(det, img):
    """Returns list of (21,2) pixel arrays sorted left->right, or None."""
    h, w = img.shape[:2]
    enhanced = for_detection(img)
    candidates = [img, enhanced, cv2.resize(enhanced, None, fx=1.5, fy=1.5)]
    for cand in candidates:
        res = det.process(cv2.cvtColor(cand, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            hands = [
                np.array([[lm.x * w, lm.y * h] for lm in hl.landmark], np.float32)
                for hl in res.multi_hand_landmarks
            ]
            hands.sort(key=lambda p: p[:, 0].mean())  # consistent order by x
            return hands
    return None


# ----------------------------------------------------------------------------
# Mask building: hull + GrabCut + adaptive skin
# ----------------------------------------------------------------------------
def ellipse(k):
    k = max(3, int(k)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def hull_masks(shape, hands, dilate_frac=0.12, erode_frac=0.04):
    m = np.zeros(shape[:2], np.uint8)
    for p in hands:
        cv2.fillConvexPoly(m, cv2.convexHull(p.astype(np.int32)), 255)
    _, _, bw, bh = cv2.boundingRect(np.vstack(hands).astype(np.int32))
    size = max(bw, bh)
    dil = cv2.dilate(m, ellipse(size * dilate_frac))
    core = cv2.erode(m, ellipse(size * erode_frac))
    if core.sum() == 0:
        core = m
    return core, dil


def adaptive_skin(img, core):
    """Skin colour model sampled INSIDE the hull (no fixed thresholds)."""
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    px = ycc[core > 0]
    if len(px) < 50:
        return np.full(core.shape, 255, np.uint8)
    mu = px[:, 1:].mean(0)
    sd = px[:, 1:].std(0) + 1e-3
    k = 2.5
    lo = np.array([0, mu[0] - k * sd[0] - 4, mu[1] - k * sd[1] - 4]).clip(0, 255)
    hi = np.array([255, mu[0] + k * sd[0] + 4, mu[1] + k * sd[1] + 4]).clip(0, 255)
    return cv2.inRange(ycc, lo.astype(np.uint8), hi.astype(np.uint8))


def run_grabcut(img, core, dil):
    gc = np.full(img.shape[:2], cv2.GC_BGD, np.uint8)
    gc[dil > 0] = cv2.GC_PR_FGD
    gc[core > 0] = cv2.GC_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return dil
    return np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def keep_largest(mask, n=2):
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:n] + 1
    out = np.zeros_like(mask)
    for i in order:
        out[lab == i] = 255
    return out


def build_mask(img, hands):
    core, dil = hull_masks(img.shape, hands)
    gc = run_grabcut(img, core, dil)
    skin = adaptive_skin(img, core)
    refined = cv2.bitwise_and(cv2.bitwise_and(gc, skin), dil)
    mask = cv2.bitwise_or(core, refined)  # hull core guarantees the hand anchor
    mask = keep_largest(mask, n=2)
    k = ellipse(max(5, 0.01 * max(img.shape[:2])))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def get_box(mask, shape, margin=0.15):
    x, y, w, h = cv2.boundingRect(mask)
    mx, my = int(w * margin), int(h * margin)
    return (max(0, x - mx), max(0, y - my),
            min(shape[1], x + w + mx), min(shape[0], y + h + my))


def square_resize(img, box, size, interp=cv2.INTER_AREA):
    x1, y1, x2, y2 = box
    c = img[y1:y2, x1:x2]
    h, w = c.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s) + c.shape[2:], c.dtype)
    oy, ox = (s - h) // 2, (s - w) // 2
    canvas[oy:oy + h, ox:ox + w] = c
    return cv2.resize(canvas, (size, size), interpolation=interp)


def build_ctx(img, hands, size):
    mask = build_mask(img, hands)
    if mask.sum() == 0:
        return None
    box = get_box(mask, img.shape)
    x1, y1, x2, y2 = box

    # enhance only inside the hand crop so shirt/windows don't skew the lighting fix
    enh = np.zeros_like(img)
    enh[y1:y2, x1:x2] = clahe_lab(img[y1:y2, x1:x2])
    soft = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255.0
    masked = (enh * soft[..., None]).astype(np.uint8)  # background -> black
    return SimpleNamespace(img=img, hands=hands, mask=mask, box=box,
                           enh=enh, masked=masked, size=size)


# ----------------------------------------------------------------------------
# Pipelines
# ----------------------------------------------------------------------------
def p0_baseline(img, size):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)


def author_hsv(img, size, canny=False):
    """Approximation of the paper's plain-background path. Thresholds are fixed
    ('experimental') values, so expect them to fail on your noisy images."""
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 30, 60), (20, 170, 255))
    m = cv2.medianBlur(m, 5)
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if cnts:
        c = max(cnts, key=cv2.contourArea)  # author assumes largest contour = hands
        cv2.drawContours(out, [c], -1, 255, -1)
    out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    if canny:
        out = cv2.Canny(out, 100, 200)
    return out


def p3_crop_gray(c):
    x1, y1, x2, y2 = c.box
    # box from the hand mask, but NO masking: keeps the original background
    g = cv2.cvtColor(clahe_lab(c.img[y1:y2, x1:x2]), cv2.COLOR_BGR2GRAY)
    full = np.zeros(c.img.shape[:2], np.uint8)
    full[y1:y2, x1:x2] = g
    return square_resize(full, c.box, c.size)


def p4_masked_rgb(c):
    return square_resize(c.masked, c.box, c.size)


def p5_silhouette(c):
    s = square_resize(c.mask, c.box, c.size)
    return (s > 127).astype(np.uint8) * 255


def edges_in_mask(c):
    gray = cv2.cvtColor(c.enh, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
    inner = cv2.erode(c.mask, ellipse(7))  # drop the mask-boundary edge / outside clutter
    return cv2.bitwise_and(edges, inner)


def p6_edges(c):
    return square_resize(edges_in_mask(c), c.box, c.size, cv2.INTER_NEAREST)


def p7_stack(c):
    gray = cv2.cvtColor(c.masked, cv2.COLOR_BGR2GRAY)
    ch = [square_resize(gray, c.box, c.size),
          square_resize(edges_in_mask(c), c.box, c.size, cv2.INTER_NEAREST),
          p5_silhouette(c)]
    return cv2.merge(ch)


def p8_skeleton(c):
    canvas = np.zeros(c.img.shape[:2], np.uint8)
    for p in c.hands:
        pts = p.astype(np.int32)
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), 255, 2)
        for pt in pts:
            cv2.circle(canvas, tuple(pt), 3, 255, -1)
    return square_resize(canvas, c.box, c.size)


def landmark_features(hands):
    """Shared origin + shared scale keep the geometry BETWEEN the two hands.
    Hands are sorted left->right; a missing hand is zero-padded."""
    allp = np.vstack(hands)
    scale = float(np.ptp(allp, axis=0).max()) + 1e-6
    origin = allp.mean(0)
    slots = [np.zeros(42, np.float32), np.zeros(42, np.float32)]
    for i, p in enumerate(hands[:2]):
        slots[i] = ((p - origin) / scale).ravel()
    return np.concatenate(slots + [np.array([len(hands)], np.float32)])


# name -> (function, needs_detection, folder name)
PIPELINES = {
    "P0": (lambda img, c, s: p0_baseline(img, s), False, "P0_baseline_gray"),
    "P1": (lambda img, c, s: author_hsv(img, s), False, "P1_author_hsv"),
    "P2": (lambda img, c, s: author_hsv(img, s, canny=True), False, "P2_author_canny"),
    "P3": (lambda img, c, s: p3_crop_gray(c), True, "P3_crop_gray"),
    "P4": (lambda img, c, s: p4_masked_rgb(c), True, "P4_masked_rgb"),
    "P5": (lambda img, c, s: p5_silhouette(c), True, "P5_silhouette"),
    "P6": (lambda img, c, s: p6_edges(c), True, "P6_edges_masked"),
    "P7": (lambda img, c, s: p7_stack(c), True, "P7_stack3"),
    "P8": (lambda img, c, s: p8_skeleton(c), True, "P8_skeleton"),
    "P9": (lambda img, c, s: landmark_features(c.hands), True, "P9_landmarks"),
}


# ----------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------
def to_bgr(x):
    return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if x.ndim == 2 else x


def save_overlay(path, ctx):
    vis = ctx.img.copy()
    cnts, _ = cv2.findContours(ctx.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    x1, y1, x2, y2 = ctx.box
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for p in ctx.hands:
        for pt in p.astype(np.int32):
            cv2.circle(vis, tuple(pt), 3, (0, 0, 255), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


def save_preview(path, outputs, size):
    tiles = []
    for name, out in outputs.items():
        if out.ndim == 1:
            continue
        t = to_bgr(cv2.resize(out, (size, size), interpolation=cv2.INTER_NEAREST)).copy()
        cv2.putText(t, name, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(t)
    if tiles:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.hstack(tiles))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pipelines", default="all", help="e.g. all or P0,P4,P7")
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--max-side", type=int, default=800, help="downscale big photos (speed)")
    ap.add_argument("--preview", action="store_true", help="save side-by-side strips + overlays")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    names = list(PIPELINES) if args.pipelines == "all" else [n.strip() for n in args.pipelines.split(",")]
    files = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    print(f"{len(files)} images, pipelines: {names}")

    det = make_detector()
    failures, lm_rows = [], []
    ok_count = {n: 0 for n in names}

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            failures.append((str(f), "unreadable"))
            continue
        img = limit_side(img, args.max_side)
        label = f.parent.name if f.parent != in_dir else "unlabeled"
        outputs = {}

        ctx = None
        if any(PIPELINES[n][1] for n in names):
            hands = detect_hands(det, img)
            if hands is None:
                failures.append((str(f), "no hands detected"))
            else:
                ctx = build_ctx(img, hands, args.size)
                if ctx is None:
                    failures.append((str(f), "empty mask"))

        for n in names:
            fn, needs_det, folder = PIPELINES[n]
            if needs_det and ctx is None:
                continue
            out = fn(img, ctx, args.size)
            outputs[n] = out
            ok_count[n] += 1
            if n == "P9":
                lm_rows.append([label, f.name] + out.tolist())
            else:
                dst = out_dir / folder / label / (f.stem + ".png")
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), out)

        if args.preview:
            if ctx is not None:
                save_overlay(out_dir / "overlay" / label / (f.stem + ".jpg"), ctx)
            save_preview(out_dir / "preview" / label / (f.stem + ".png"), outputs, args.size)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "failures.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "reason"])
        w.writerows(failures)
    if lm_rows:
        with open(out_dir / "landmarks.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["label", "file"] + [f"f{i}" for i in range(len(lm_rows[0]) - 2)])
            w.writerows(lm_rows)

    total = max(1, len(files))
    print(f"\nDetection/segmentation failures: {len(failures)}/{len(files)} "
          f"({100 * len(failures) / total:.1f}%)  -> see failures.csv")
    for n in names:
        print(f"  {n}: {ok_count[n]} images written")


if __name__ == "__main__":
    main()