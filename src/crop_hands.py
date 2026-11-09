"""Crop sign images around detected hands while preserving the class folders.

Example:
    python src/crop_hands.py --input data --output data_cropped

The input tree is left untouched. Each detected hand's MediaPipe landmarks are
combined into one square crop, with a little margin for the fingertips.
"""
import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def hand_crop_box(hand_landmarks, image_shape, padding):
    """Get a square box around all detected hand landmarks, clipped to image."""
    height, width = image_shape[:2]
    xs = [point.x * width for point in hand_landmarks]
    ys = [point.y * height for point in hand_landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Square crops keep hand proportions stable across the dataset. Padding
    # includes a small amount of wrist/background without letting it dominate.
    side = max(x_max - x_min, y_max - y_min) * (1 + 2 * padding)
    side = max(2, int(round(side)))
    center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    x1 = int(round(center_x - side / 2))
    y1 = int(round(center_y - side / 2))
    x2, y2 = x1 + side, y1 + side

    # Shift the square into the frame where possible instead of shrinking it.
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > width:
        x1 -= x2 - width
        x2 = width
    if y2 > height:
        y1 -= y2 - height
        y2 = height
    x1, y1 = max(0, x1), max(0, y1)
    return x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data", help="Root containing signer/class folders")
    parser.add_argument("--output", default="data_cropped", help="Destination root (keeps input unchanged)")
    parser.add_argument("--padding", type=float, default=0.25,
                        help="Extra margin around each hand, as a fraction of crop size (default: 0.25)")
    parser.add_argument("--size", type=int, default=0,
                        help="Optional square output size; 0 keeps the crop's original resolution")
    parser.add_argument("--min-confidence", type=float, default=0.35)
    args = parser.parse_args()

    source = Path(args.input)
    destination = Path(args.output)
    if not source.is_dir():
        parser.error(f"Input folder does not exist: {source}")
    if source.resolve() == destination.resolve() or source.resolve() in destination.resolve().parents:
        parser.error("Output must be separate from the input folder")
    if args.padding < 0 or args.size < 0:
        parser.error("padding and size must be non-negative")

    files = sorted(path for path in source.rglob("*")
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    failures = []
    saved = 0
    detector = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=args.min_confidence,
        model_complexity=1,
    )
    try:
        for path in files:
            image = cv2.imread(str(path))
            if image is None:
                failures.append((str(path), "unreadable image"))
                continue

            result = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not result.multi_hand_landmarks:
                failures.append((str(path), "no hand detected"))
                continue

            # Enclose both hands together when a sign uses two hands.
            boxes = [hand_crop_box(hand.landmark, image.shape, args.padding)
                     for hand in result.multi_hand_landmarks]
            x1 = min(box[0] for box in boxes)
            y1 = min(box[1] for box in boxes)
            x2 = max(box[2] for box in boxes)
            y2 = max(box[3] for box in boxes)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                failures.append((str(path), "empty crop"))
                continue
            if args.size:
                crop = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_AREA)

            relative = path.relative_to(source)
            output_path = (destination / relative).with_suffix(".png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), crop):
                failures.append((str(path), "could not write crop"))
                continue
            saved += 1
    finally:
        detector.close()

    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "crop_failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "reason"])
        writer.writerows(failures)
    print(f"Saved {saved}/{len(files)} hand crops to {destination}")
    print(f"Skipped {len(failures)} images; details: {destination / 'crop_failures.csv'}")


if __name__ == "__main__":
    main()
