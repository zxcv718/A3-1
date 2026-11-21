import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

from scanner import MAX_BLUR, Image, ScanResult, ScanSettings, scan


STAGE_WINDOWS = (
    "1. Original",
    "2. Preprocessed",
    "3. Edges",
    "4. Contour",
    "5. Result",
)
CONTROL_WINDOW = STAGE_WINDOWS[1]
BLUR_LABEL = "Blur (2n+1)"
LOW_LABEL = "Canny low"
HIGH_LABEL = "Canny high"


def read_image(path: Path) -> Image:
    try:
        data = path.read_bytes()
        if data:
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                return image
    except (OSError, cv2.error) as error:
        raise ValueError(f"이미지를 읽을 수 없습니다. {path} ({error})") from error
    raise ValueError(f"이미지를 읽을 수 없습니다. {path}의 파일 형식을 확인하세요.")


def save_image(path: Path, image: Image) -> None:
    extension = path.suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("저장 파일의 확장자는 .jpg, .jpeg 또는 .png여야 합니다.")
    try:
        success, encoded = cv2.imencode(extension, image)
        if not success:
            raise OSError("이미지를 인코딩하지 못했습니다.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded.tobytes())
    except (OSError, cv2.error) as error:
        raise ValueError(f"이미지를 저장할 수 없습니다. {path} ({error})") from error


def brightness_histogram(image: Image) -> Image:
    """Draw the input image's relative brightness distribution, in 32 bins."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    counts = (
        np.bincount(gray.astype(np.uint8, copy=False).ravel(), minlength=256)
        .reshape(32, 8)
        .sum(axis=1)
    )
    percentages = counts * (100.0 / gray.size)
    ceiling = max(5, 5 * int(np.ceil(percentages.max() / 5)))
    chart = np.full((320, 600, 3), 255, dtype=np.uint8)
    left, bottom, plot_width, plot_height = 55, 275, 512, 213
    cv2.line(chart, (left, bottom), (left + plot_width, bottom), (65, 65, 65), 1)
    cv2.line(chart, (left, bottom), (left, bottom - plot_height), (65, 65, 65), 1)
    for index, percent in enumerate(percentages):
        x = left + index * 16
        y = bottom - round(float(percent) / ceiling * plot_height)
        if y < bottom:
            cv2.rectangle(chart, (x + 1, y), (x + 14, bottom - 1), (160, 100, 25), -1)

    mean = float(gray.mean())
    title = f"Brightness histogram  |  mean: {mean:.1f}"
    labels = [
        (title, (left, 36)),
        (f"{ceiling}%", (5, 67)),
        ("0", (left, 298)),
        ("128", (left + 242, 298)),
        ("255", (left + 482, 298)),
    ]
    for label, origin in labels:
        cv2.putText(
            chart,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
    return chart


def fit_preview(image: Image) -> Image:
    height, width = image.shape[:2]
    scale = min(1.0, 400 / width, 310 / height)
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def empty_preview(message: str) -> Image:
    canvas = np.full((140, 400, 3), 38, dtype=np.uint8)
    cv2.putText(
        canvas,
        message,
        (14, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return canvas


def show_scan(image: Image, settings: ScanSettings, output: Path) -> int:
    try:
        for index, name in enumerate(STAGE_WINDOWS):
            cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(name, 30 + (index % 3) * 410, 60 + (index // 3) * 455)
        cv2.createTrackbar(
            BLUR_LABEL,
            CONTROL_WINDOW,
            (settings.blur - 1) // 2,
            (MAX_BLUR - 1) // 2,
            lambda _: None,
        )
        cv2.createTrackbar(
            LOW_LABEL, CONTROL_WINDOW, settings.canny_low, 255, lambda _: None
        )
        cv2.createTrackbar(
            HIGH_LABEL, CONTROL_WINDOW, settings.canny_high, 255, lambda _: None
        )
        cv2.imshow(STAGE_WINDOWS[0], fit_preview(image))
        print("s: 현재 스캔본 저장 / q 또는 Esc: 종료", flush=True)

        previous_values = None
        result: ScanResult | None = None
        while True:
            blur_position = cv2.getTrackbarPos(BLUR_LABEL, CONTROL_WINDOW)
            low = cv2.getTrackbarPos(LOW_LABEL, CONTROL_WINDOW)
            high = cv2.getTrackbarPos(HIGH_LABEL, CONTROL_WINDOW)
            if min(blur_position, low, high) < 0:
                return 0
            values = (2 * blur_position + 1, low, high)
            if values != previous_values:
                previous_values = values
                if low >= high:
                    result = None
                    for window in STAGE_WINDOWS[1:]:
                        cv2.imshow(
                            window, empty_preview("Canny low must be below high")
                        )
                else:
                    result = scan(image, ScanSettings(*values))
                    final = result.scanned
                    if final is None:
                        final = empty_preview("No document detected")
                    for window, stage in zip(
                        STAGE_WINDOWS[1:],
                        (result.preprocessed, result.edges, result.overlay, final),
                    ):
                        cv2.imshow(window, fit_preview(stage))

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord("s"):
                if result is None or result.scanned is None:
                    print("저장할 스캔 결과가 없습니다.", file=sys.stderr, flush=True)
                    continue
                try:
                    save_image(output, result.scanned)
                except ValueError as error:
                    print(error, file=sys.stderr, flush=True)
                else:
                    print(f"저장했습니다. {output}", flush=True)
    finally:
        cv2.destroyAllWindows()


def main() -> int:
    defaults = ScanSettings()
    parser = argparse.ArgumentParser(
        description="사진 속 문서를 찾아 스캔본으로 저장합니다."
    )
    parser.add_argument("image", type=Path, help="문서 사진 경로")
    parser.add_argument("--output", type=Path, help="스캔본의 JPG 또는 PNG 경로")
    parser.add_argument(
        "--no-gui", action="store_true", help="창을 열지 않고 즉시 저장"
    )
    parser.add_argument(
        "--blur", type=int, default=defaults.blur, help="홀수 블러 커널 크기"
    )
    parser.add_argument(
        "--canny-low", type=int, default=defaults.canny_low, help="낮은 Canny 임계값"
    )
    parser.add_argument(
        "--canny-high", type=int, default=defaults.canny_high, help="높은 Canny 임계값"
    )
    parser.add_argument(
        "--histogram", type=Path, help="입력 사진의 밝기 히스토그램 저장 경로"
    )
    args = parser.parse_args()
    output = args.output or args.image.with_name(f"{args.image.stem}_scanned.png")

    try:
        settings = ScanSettings(args.blur, args.canny_low, args.canny_high)
        paths = (
            [args.image, output]
            if args.histogram is None
            else [args.image, output, args.histogram]
        )
        if len({path.resolve() for path in paths}) != len(paths):
            raise ValueError(
                "입력 파일, 스캔본, 히스토그램은 서로 다른 경로여야 합니다."
            )
        image = read_image(args.image)
        if args.histogram is not None:
            save_image(args.histogram, brightness_histogram(image))

        if not args.no_gui:
            return show_scan(image, settings, output)
        result = scan(image, settings)
        if result.scanned is None:
            print("문서 후보를 찾지 못했습니다.", file=sys.stderr)
            return 2
        save_image(output, result.scanned)
    except (ValueError, OSError, cv2.error) as error:
        print(error, file=sys.stderr)
        return 1

    print(f"저장했습니다. {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
