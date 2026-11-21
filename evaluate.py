import argparse
import csv
import json
from math import hypot
from pathlib import Path

import cv2
import numpy as np

from main import brightness_histogram, read_image, save_image
from scanner import Corners, Image, ScanSettings, order_corners, scan


GROUPS = ("front", "shadow", "tilted", "cluttered")
PRESETS = {
    "default": ScanSettings(),
    "planned": ScanSettings(5, 50, 150),
    "strict": ScanSettings(7, 80, 200),
}
MIN_IOU = 0.90
MAX_CORNER_ERROR = 0.05


def compare_outline(
    predicted: Corners, reference: Corners, shape: tuple[int, int]
) -> tuple[float, float]:
    predicted = order_corners(predicted)
    reference = order_corners(reference)
    intersection, _ = cv2.intersectConvexConvex(predicted, reference)
    union = cv2.contourArea(predicted) + cv2.contourArea(reference) - intersection
    corner_error = min(
        float(
            np.linalg.norm(np.roll(predicted, offset, axis=0) - reference, axis=1).max()
        )
        for offset in range(4)
    )
    return float(np.clip(intersection / union, 0, 1)), corner_error / hypot(*shape)


def tile(image: Image | None, title: str) -> Image:
    canvas = np.full((196, 288, 3), 245, dtype=np.uint8)
    cv2.putText(
        canvas,
        title,
        (9, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    if image is not None:
        height, width = image.shape[:2]
        ratio = min(278 / width, 168 / height)
        view = cv2.resize(
            image,
            (max(1, round(width * ratio)), max(1, round(height * ratio))),
            interpolation=cv2.INTER_AREA,
        )
        if view.ndim == 2:
            view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
        y = 22 + (168 - view.shape[0]) // 2
        x = (288 - view.shape[1]) // 2
        canvas[y : y + view.shape[0], x : x + view.shape[1]] = view
    return canvas


def evaluate(manifest: Path, destination: Path) -> None:
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 20 or len({row.get("file") for row in rows}) != 20:
        raise ValueError("서로 다른 평가 이미지 20장이 필요합니다.")
    if any(
        row.get("group") not in GROUPS or not row.get("ground_truth") for row in rows
    ):
        raise ValueError("각 이미지의 평가 분류와 ground_truth 네 점을 기록하세요.")

    destination.mkdir(parents=True, exist_ok=True)
    strips: dict[str, list[Image]] = {group: [] for group in GROUPS}
    metrics = []
    for row in rows:
        path = manifest.parent / row["file"]
        image = read_image(path)
        reference = order_corners(
            np.array(json.loads(row["ground_truth"]), dtype=np.float32)
        )
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for preset, settings in PRESETS.items():
            result = scan(image, settings)
            work_height, work_width = result.original.shape[:2]
            scale = np.array(
                [work_width / width, work_height / height], dtype=np.float32
            )
            expected = reference * scale
            iou = error = None
            if result.corners is not None:
                iou, error = compare_outline(
                    result.corners, expected, (work_height, work_width)
                )
            passed = (
                iou is not None
                and error is not None
                and iou >= MIN_IOU
                and error <= MAX_CORNER_ERROR
            )
            verdict = (
                "pass" if passed else "no_candidate" if iou is None else "wrong_outline"
            )
            metrics.append(
                {
                    "file": row["file"],
                    "group": row["group"],
                    "preset": preset,
                    "verdict": verdict,
                    "iou": "" if iou is None else f"{iou:.4f}",
                    "corner_error": "" if error is None else f"{error:.4f}",
                    "mean_brightness": f"{gray.mean():.1f}",
                    "edge_percent": f"{100 * np.count_nonzero(result.edges) / result.edges.size:.2f}",
                }
            )
            if preset != "default":
                continue
            overlay = result.overlay.copy()
            cv2.polylines(
                overlay, [np.rint(expected).astype(np.int32)], True, (0, 170, 0), 2
            )
            chart = brightness_histogram(image)
            save_image(destination / "histograms" / f"{path.stem}.png", chart)
            strips[row["group"]].append(
                np.hstack(
                    [
                        tile(result.original, path.stem),
                        tile(result.edges, "Canny"),
                        tile(overlay, "Green = reference, red = predicted"),
                        tile(result.scanned, verdict),
                        tile(chart, "Brightness distribution"),
                    ]
                )
            )
    with (destination / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    for group, images in strips.items():
        if images:
            save_image(destination / f"{group}.jpg", np.vstack(images))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="기준 모서리가 있는 평가 이미지 20장을 비교합니다."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    options = parser.parse_args()
    try:
        evaluate(options.manifest, options.output_dir)
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(1, f"평가를 완료할 수 없습니다. {error}\n")


if __name__ == "__main__":
    main()
