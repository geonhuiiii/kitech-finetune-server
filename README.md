# KITECH 파인튜닝 백엔드 서버

`RepositoryFinetuneChatbot`(파인튜닝 위자드)이 연동하는 **외부 파인튜닝 서버**입니다.
프론트엔드의 `FINETUNE_API_SERVER_URL` 이 이 서버를 가리키면, Next.js API 라우트가
요청을 그대로 프록시하고, 미설정 시에는 프론트엔드 내장 규칙 기반 폴백으로 동작합니다.

구현 스펙은 [`src/app/api/repositories/finetune-recommend/API_SPEC.md`](../src/app/api/repositories/finetune-recommend/API_SPEC.md) 를 따릅니다.

## 기술 스택

- Python 3.11 / FastAPI / Uvicorn
- 비동기 SSE 로그 스트리밍, 멀티파트 청크 업로드
- 의존성 최소화 (torch 등 ML 패키지 불포함 — 학습 단계는 시뮬레이션)

## 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET`  | `/health` | 헬스체크 |
| `POST` | `/recommend` | 백본 모델 추천 (`{dataType, taskType}` → `{models}`) |
| `POST` | `/upload` | 데이터셋 청크 업로드 (multipart + `Content-Range`) |
| `POST` | `/jobs` | 파인튜닝 잡 제출 (201) |
| `GET`  | `/jobs/{jobId}` | 잡 상태 조회 |
| `GET`  | `/jobs/{jobId}/logs` | SSE 학습 로그 스트리밍 |
| `GET`  | `/jobs/{jobId}/model` | 파인튜닝된 모델 가중치 다운로드 |

API 문서(Swagger UI): 서버 기동 후 `http://localhost:8000/docs`

---

## Docker Engine으로 실행 (Docker Desktop 불필요)

> 아래 명령은 Docker Desktop 없이 **Docker Engine + Compose 플러그인**만으로 동작합니다.
> Windows에서는 WSL2 배포판(Ubuntu 등) 안의 Docker Engine 셸에서 실행하세요.

### 0) (Windows/WSL) WSL Ubuntu에 Docker Engine 설치 — 최초 1회

> Docker Desktop을 쓰지 않으므로, WSL2 배포판(예: `Ubuntu-22.04`) 안에 Docker Engine을 직접 설치합니다.
> PowerShell에서 `wsl -d Ubuntu-22.04` 로 진입한 뒤 아래를 실행하세요. (sudo 비밀번호 필요)

```bash
# 1. 공식 저장소 등록 후 docker-ce 설치
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 2. sudo 없이 docker 사용 (그룹 적용 위해 WSL 재시작 권장)
sudo usermod -aG docker $USER

# 3. Docker 데몬 시작
#   (a) WSL에 systemd가 켜져 있으면:
sudo systemctl enable --now docker
#   (b) systemd 미사용 환경이면 (또는 위가 안 되면):
sudo service docker start
```

> systemd가 꺼져 있다면 `/etc/wsl.conf` 에 `[boot]\nsystemd=true` 를 추가하고
> PowerShell에서 `wsl --shutdown` 후 다시 진입하면 systemd가 활성화됩니다.

설치 확인:

```bash
docker version
docker compose version
```

### 1) 프로젝트로 이동 (WSL에서 Windows 경로 접근)

```bash
cd "/mnt/c/Users/geonhui/Desktop/PROJECT/레포지터리/repoServer_login_backup1 현재작업본/kitech-repository-FE/finetune-server"
```

### 2) 빌드 & 기동 (Compose 권장)

```bash
docker compose up -d --build
```

기동 확인:

```bash
curl http://localhost:8000/health
# {"status":"ok","service":"kitech-finetune-server"}
```

로그 확인 / 중지:

```bash
docker compose logs -f finetune-server
docker compose down            # 컨테이너 제거 (볼륨 유지)
docker compose down -v         # 볼륨까지 제거
```

### 3) Compose 없이 순수 docker 명령으로 실행

```bash
cd finetune-server
docker build -t kitech-finetune-server:latest .

docker volume create finetune-data
docker run -d --name kitech-finetune-server \
  -p 8000:8000 \
  -v finetune-data:/data \
  --restart unless-stopped \
  kitech-finetune-server:latest
```

---

## 동작 빠른 검증

```bash
# 1. 모델 추천
curl -s -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"dataType":"image","taskType":"classification"}'

# 2. 파일 업로드 (단일 파일)
echo "filename,label" > sample.csv && echo "a.jpg,crack" >> sample.csv
curl -s -X POST http://localhost:8000/upload \
  -F uploadId=test-123 -F role=label_csv -F file=@sample.csv

# 3. 잡 제출
curl -s -X POST http://localhost:8000/jobs \
  -H 'Content-Type: application/json' \
  -d '{"dataType":"image","taskType":"classification","subTaskType":"binary_multiclass","backboneModelId":"convnext_v2_b","uploadId":"test-123","classNames":["crack","normal"]}'

# 4. 상태 조회 / 로그 스트리밍 (위 응답의 jobId 사용)
curl -s http://localhost:8000/jobs/<jobId>
curl -N http://localhost:8000/jobs/<jobId>/logs
```

---

## 프론트엔드 연동

프론트엔드(`kitech-repository-FE`)의 `.env.development` / `.env.production` 에 추가:

```
FINETUNE_API_SERVER_URL=http://localhost:8000
```

- 같은 머신이면 `http://localhost:8000`
- 다른 서버면 해당 호스트:포트 (예: `http://10.241.120.217:8000`)

설정 후 Next.js의 다음 라우트가 이 서버로 프록시됩니다.

- `POST /api/repositories/finetune-recommend` → `POST {URL}/recommend`
- `POST /api/repositories/finetune-upload` → `POST {URL}/upload`

> `FINETUNE_API_SERVER_URL` 이 비어 있으면 프론트엔드는 내장 규칙 기반 추천(폴백)으로 동작하고
> 업로드는 Next.js 임시 디렉터리에 저장됩니다.

---

## 환경 변수

[`.env.example`](.env.example) 참고. 주요 항목:

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `FINETUNE_DATA_DIR` | `/data` | 업로드/모델 저장 루트 |
| `FINETUNE_MAX_FILE_SIZE` | `5368709120` (5GB) | 업로드 최대 크기 |
| `FINETUNE_UPLOAD_TTL_HOURS` | `24` | 업로드 세션 자동 정리 TTL |
| `FINETUNE_SIM_EPOCHS` | `10` | 시뮬레이션 학습 에포크 수 |
| `FINETUNE_SIM_EPOCH_SECONDS` | `3` | 에포크당 소요 시간(초) |
| `FINETUNE_CORS_ORIGINS` | `*` | CORS 허용 출처 |
| `FINETUNE_PUBLIC_BASE_URL` | (빈값) | 모델 다운로드 절대 URL 베이스 |

---

## 학습 모드: 실제 학습(torch) ↔ 시뮬레이션 자동 전환

[`app/jobs_engine.py`](app/jobs_engine.py) 는 잡을 받으면 태스크에 맞는 **실제 트레이너**를 찾습니다.

- **실제 학습**: torch/torchvision/pillow 가 설치돼 있고 지원 태스크면, 업로드된 데이터로 진짜 학습합니다.
  - 현재 지원: **이미지 분류**(`image` / `classification`) — torchvision ResNet18 전이학습
    ([`app/trainers/image_classification.py`](app/trainers/image_classification.py)).
    업로드 세션의 `class_*` 폴더를 클래스로 인식, Train/Val 자동 분할, 실제 loss/val_acc 로그를 SSE 로 스트리밍,
    학습된 `state_dict` 를 `.pt` 로 저장합니다.
  - 사전학습 가중치 다운로드가 막힌 사내망에서도 무작위 초기화로 학습이 진행됩니다.
- **시뮬레이션 폴백**: torch 미설치이거나 아직 트레이너가 없는 태스크면 로스 곡선을 흉내 내고 플레이스홀더 파일을 만듭니다.

### 실제 학습 활성화 (torch 설치)

로컬:

```bash
pip install -r requirements-train.txt --index-url https://download.pytorch.org/whl/cpu
```

Docker: `Dockerfile` 에서 `requirements-train.txt` 도 설치하도록 한 줄 추가하거나,
GPU 런타임 베이스 이미지(`pytorch/pytorch:*-cuda*`)로 교체하면 됩니다.

### 다른 태스크 실제 트레이너 추가

[`app/trainers/registry.py`](app/trainers/registry.py) 에 매핑을 추가하고,
[`app/trainers/base.py`](app/trainers/base.py) 의 `Trainer` 인터페이스(`is_available`, `train`)를 구현하면
jobs_engine / 프론트엔드 변경 없이 연결됩니다.

## 로컬(비-Docker) 개발 실행

```bash
cd finetune-server
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
FINETUNE_DATA_DIR=./data uvicorn app.main:app --reload --port 8000
```
