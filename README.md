# 문서 스캐너

사진 속 종이의 네 모서리를 찾아 투시 변환하고, 스캔본을 JPG 또는 PNG로 저장합니다. Python 3.12에서 검증했습니다.

## 실행

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py '사진.jpg'
```

다섯 창에 원본, 전처리, 엣지, 윤곽선, 스캔 결과가 표시됩니다. `2. Preprocessed` 창에 있는 세 슬라이더로 블러와 Canny 임계값을 조절합니다. 블러의 슬라이더 위치 `n`은 커널 크기 `2n+1`에 해당합니다. 시작값은 블러 `3`, Canny `25/80`입니다. `s`로 저장하고 `q` 또는 `Esc`로 종료합니다. 기본 저장 경로는 입력 사진 옆의 `사진_scanned.png`입니다.

창을 열 수 없는 환경에서는 명령줄에서 스캔본과 히스토그램을 저장할 수 있습니다.

```sh
python main.py '사진.jpg' --no-gui --output scan.png --histogram brightness.png
python main.py '사진.jpg' --no-gui --blur 5 --canny-low 50 --canny-high 150
```

## 처리 과정

- `scanner.py`에서 회색조 변환, 블러, Canny, 윤곽선 검출, 투시 변환, 적응형 이진화를 순서대로 실행합니다.
- 윤곽선이 네 점으로 근사되면 중심 기준 각도를 사용해 시계 방향으로 정렬합니다. 좌표 합이 가장 작은 점부터 출력 직사각형의 모서리와 대응시킵니다.
- `main.py`는 스캔 과정의 이미지를 OpenCV 창에 표시하고 입력·저장 오류를 사용자에게 알립니다.

## 검증 및 평가 재현

```sh
python -m unittest discover -s tests -v
python evaluate.py evaluation/sources.csv
```

평가 결과와 사용한 사진의 출처는 [분석 보고서](report.md)에서 확인할 수 있습니다.
