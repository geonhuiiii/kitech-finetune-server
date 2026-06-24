# KITECH 파인튜닝 백엔드 서버 (FastAPI + Uvicorn)
FROM python:3.11-slim

# 파이썬 런타임 권장 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FINETUNE_DATA_DIR=/data

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시 활용)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY app ./app

# 업로드/모델 산출물 저장 경로 (볼륨 마운트 권장)
RUN mkdir -p /data \
    && addgroup --system kitech \
    && adduser --system --ingroup kitech kitech \
    && chown -R kitech:kitech /app /data
USER kitech

EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
