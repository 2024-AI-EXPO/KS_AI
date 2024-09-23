FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    build-essential \
    pkg-config \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*


# 프로젝트 파일 복사
COPY . /app

# 폰트 파일을 적절한 경로에 복사
COPY gulim.ttc /usr/share/fonts/truetype/

# 모델 파일 복사
COPY models /app/models

# 필요한 Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 포트 설정
EXPOSE 8000

# FastAPI 앱 실행
CMD ["uvicorn", "websocket:app", "--host", "0.0.0.0", "--port", "8000"]
