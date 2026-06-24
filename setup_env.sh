#!/usr/bin/env bash
# finetune-server micromamba 환경 설치 + 서버 실행 스크립트
# 사용법: bash setup_env.sh

set -e

ENV_NAME="finetune"
PORT="${FINETUNE_PORT:-8199}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/4] micromamba 환경 확인/생성 (Python 3.11) ==="
if micromamba env list | grep -q "^${ENV_NAME} "; then
  echo "환경 '${ENV_NAME}' 이미 존재 — 스킵"
else
  micromamba create -n "${ENV_NAME}" python=3.11 -c conda-forge -y
fi

echo "=== [2/4] 기본 의존성 설치 ==="
micromamba run -n "${ENV_NAME}" pip install \
  fastapi==0.115.5 \
  "uvicorn[standard]==0.32.1" \
  python-multipart==0.0.18 \
  "pydantic==2.10.3" \
  "python-dotenv>=1.0.0"

echo "=== [3/4] torch CPU 설치 (CUDA 충돌 회피) ==="
micromamba run -n "${ENV_NAME}" pip install \
  torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "=== [4/4] pillow 설치 ==="
micromamba run -n "${ENV_NAME}" pip install pillow

echo ""
echo "=== 설치 검증 ==="
micromamba run -n "${ENV_NAME}" python -c "
import torch, torchvision
from PIL import Image
print(f'✅ torch {torch.__version__}  |  torchvision {torchvision.__version__}  |  CUDA={torch.cuda.is_available()}')
"

echo ""
echo "=== 서버 시작 (포트 ${PORT}) ==="
cd "${SCRIPT_DIR}"
if [ -f ".env.example" ] && [ ! -f ".env" ]; then
  cp .env.example .env
  echo ".env.example → .env 복사 완료"
fi

nohup micromamba run -n "${ENV_NAME}" \
  uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" \
  > ~/finetune-server.log 2>&1 &

SERVER_PID=$!
echo "서버 PID: ${SERVER_PID}"
sleep 3

if curl -sf "http://localhost:${PORT}/health" > /dev/null; then
  echo "✅ 서버 정상 기동: http://localhost:${PORT}"
  echo ""
  echo "서버 IP 확인:"
  hostname -I | awk '{print $1}'
  echo ""
  echo "프론트엔드 .env.development 에 아래 줄 추가:"
  echo "FINETUNE_API_SERVER_URL=http://$(hostname -I | awk '{print $1}'):${PORT}"
else
  echo "❌ 서버 응답 없음 — 로그 확인:"
  tail -20 ~/finetune-server.log
fi
