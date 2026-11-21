from dataclasses import dataclass

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray


Image = MatLike
Corners = NDArray[np.float32]

MIN_AREA_RATIO = 0.05
CONTOUR_TOLERANCE = 0.02
ADAPTIVE_BLOCK_SIZE = 21
ADAPTIVE_OFFSET = 10
DETECTION_LONG_EDGE = 640
MAX_BLUR = 31


@dataclass(frozen=True)
class ScanSettings:
    blur: int = 3
    canny_low: int = 25
    canny_high: int = 80

    def __post_init__(self) -> None:
        if not 1 <= self.blur <= MAX_BLUR or self.blur % 2 == 0:
            raise ValueError(f"블러 커널은 1~{MAX_BLUR} 사이의 홀수여야 합니다.")
        if not 0 <= self.canny_low < self.canny_high <= 255:
            raise ValueError("Canny 임계값은 0 ≤ 낮은 값 < 높은 값 ≤ 255여야 합니다.")


@dataclass(frozen=True)
class ScanResult:
    original: Image
    preprocessed: Image
    edges: Image
    overlay: Image
    corners: Corners | None
    scanned: Image | None


def order_corners(points: NDArray[np.number]) -> Corners:
    """Order a convex quadrilateral clockwise, starting at its upper-left corner."""
    corners = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(corners).all() or len(np.unique(corners, axis=0)) != 4:
        raise ValueError("서로 다른 네 모서리가 필요합니다.")

    # 이미지에서는 y가 아래로 증가하므로 각도가 증가하는 순서가 시계 방향이다.
    offsets = corners - corners.mean(axis=0)
    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    corners = corners[np.argsort(angles)]
    edges = np.roll(corners, -1, axis=0) - corners
    next_edges = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    if np.any(turns <= 0):
        raise ValueError("네 모서리가 볼록 사각형을 이루어야 합니다.")

    # 합이 같은 점은 더 위에 있는 점을 시작점으로 사용한다.
    start = min(range(4), key=lambda i: (corners[i].sum(), corners[i, 1]))
    return np.roll(corners, -start, axis=0)


def preprocess(image: Image, blur: int) -> tuple[Image, Image]:
    """BGR 배열을 회색조로 바꾸어 원본과 가우시안 블러 결과를 반환한다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray, cv2.GaussianBlur(gray, (blur, blur), 0)


def detect_edges(blurred: Image, low: int, high: int) -> Image:
    """블러를 적용한 회색조 배열에서 0과 255로 구성된 엣지 영상을 찾는다."""
    return cv2.Canny(blurred, low, high)


def find_document(edges: Image) -> Corners | None:
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = edges.size * MIN_AREA_RATIO
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            break
        tolerance = cv2.arcLength(contour, True) * CONTOUR_TOLERANCE
        polygon = cv2.approxPolyDP(contour, tolerance, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            return order_corners(polygon)
    return None


def warp_document(gray: Image, ordered: NDArray[np.number]) -> Image:
    source = np.asarray(ordered, dtype=np.float32)
    top_left, top_right, bottom_right, bottom_left = source
    width = round(
        max(
            np.linalg.norm(top_right - top_left),
            np.linalg.norm(bottom_right - bottom_left),
        )
    )
    height = round(
        max(
            np.linalg.norm(bottom_left - top_left),
            np.linalg.norm(bottom_right - top_right),
        )
    )
    if width < 2 or height < 2:
        raise ValueError("문서 영역이 투시 변환을 하기에는 너무 작습니다.")

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(gray, matrix, (width, height))


def scan(image: Image, settings: ScanSettings = ScanSettings()) -> ScanResult:
    height, width = image.shape[:2]
    scale = min(1.0, DETECTION_LONG_EDGE / max(height, width))
    preview = image
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        preview = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    gray, blurred = preprocess(preview, settings.blur)
    edges = detect_edges(blurred, settings.canny_low, settings.canny_high)
    overlay = preview.copy()
    corners = find_document(edges)
    if corners is None:
        return ScanResult(preview, blurred, edges, overlay, None, None)

    outline = np.rint(corners).astype(np.int32)
    cv2.polylines(overlay, [outline], isClosed=True, color=(0, 0, 255), thickness=3)
    for number, (x, y) in enumerate(outline, start=1):
        cv2.circle(overlay, (int(x), int(y)), 5, (0, 255, 255), -1)
        cv2.putText(
            overlay,
            str(number),
            (int(x) + 7, int(y) - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if scale < 1.0:
        # 검출은 축소본에서 하고, 최종 스캔은 원본 해상도로 만든다.
        ratio = np.array(
            [width / preview.shape[1], height / preview.shape[0]], dtype=np.float32
        )
        warped = warp_document(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), corners * ratio)
    else:
        warped = warp_document(gray, corners)
    scanned = cv2.adaptiveThreshold(
        warped,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_OFFSET,
    )
    return ScanResult(preview, blurred, edges, overlay, corners, scanned)
