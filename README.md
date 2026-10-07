# Laya Server on NVIDIA CUDA 13 (Blackwell sm_120)

Containerized, high-throughput deployment setup for serving the **Laya** non-autoregressive decision model (`convaiinnovations/laya` via `receptron/laya`) using GPU acceleration on CUDA 13.0 and NVIDIA Blackwell architecture GPUs (e.g., RTX 50-series). The project encapsulates PyTorch nightly wheels with `sm_120` support and exposes an HTTP System 1 decision engine on port `8090` via Docker Compose.

---

## 2. Project Architecture and Structure

### High-Level Architecture

The project follows a **Containerized Microservice / Model Serving** architecture designed for low-latency machine learning inference:

```
[ Host Machine (Ubuntu 24.04 / CUDA 13.0 / NVIDIA Driver 580+) ]
       │
       ├── Shared Memory (/dev/shm: 16 GB)
       ├── Local Cache Bind Mounts (~/.cache/huggingface, ~/.cache/receptron-laya)
       │
       └── [ Docker Engine (NVIDIA Container Toolkit) ]
                 │
                 └── [ Container: laya-server (network_mode: host) ]
                           │
                           ├── Environment: CUDA 13.0 Runtime + Python 3 venv
                           ├── Framework: PyTorch Nightly (cu130 with sm_120 SASS/PTX)
                           ├── Engine: laya[serve] (ASGI / Uvicorn)
                           │
                           └── Endpoints:
                                 ├── GET  /health
                                 └── POST /v1/systemone (Typed Decisions / Fast Path)

```

### Directory and Key Files Structure

```text
.
├── Dockerfile          # Builds the container image: Ubuntu 24.04 base, CUDA 13 runtime, PyTorch cu130 nightly, and laya[serve]
└── docker-compose.yml  # Orchestrates resource limits, GPU reservations, volume caching, and environment configurations

```

* **`Dockerfile`**: Sets up the reproducible build environment. Installs system dependencies, provisions an isolated Python virtual environment (`/opt/venv`), installs bleeding-edge PyTorch wheels with compute capability `sm_120` support, installs the `laya[serve]` package, and declares `laya-serve` as the executable entrypoint.
* **`docker-compose.yml`**: Manages the operational runtime parameters. Directs the NVIDIA runtime to allocate 1 GPU device with compute capabilities, locks shared memory to 16 GB (`shm_size`), mounts persistent Hugging Face and Receptron model caches from the host to prevent redundant cold-start downloads, binds directly to the host network stack, and configures Laya server variables (`LAYA_PORT`, `LAYA_DEVICE`, `LAYA_PRELOAD`).

### Component Interaction

1. **Docker Compose Engine** initializes the container and requests access to physical GPU `0` through the NVIDIA Container Runtime.
2. The entrypoint invokes `laya-serve`, which reads environment variables (`LAYA_HOST`, `LAYA_PORT`, `LAYA_DEVICE=cuda`, `LAYA_PRELOAD=1`).
3. Upon initialization, `laya-serve` checks the local cache mount (`/root/.cache/huggingface` and `/root/.cache/receptron-laya`). If models are missing, it pulls the weights using the optional `$HF_TOKEN`.
4. Weights are mapped directly to CUDA device memory using the custom PyTorch `cu130` kernels.
5. Inbound HTTP requests sent to `http://localhost:8090/v1/systemone` are processed by the internal encoder (ModernBERT backbone), evaluating multi-class `choice` or binary `noul` decisions in a single non-autoregressive forward pass.

---

## 3. Build and Test Commands

### Prerequisites and Dependencies

* **Host Operating System:** Linux (Ubuntu 22.04 LTS or 24.04 LTS recommended).
* **NVIDIA Driver:** Version `>= 580.x` supporting CUDA 13.0.
* **NVIDIA Container Toolkit:** Configured with Docker engine (`nvidia-ctk runtime configure`).
* **Docker Engine & Docker Compose:** Docker Compose V2 installed.
* **Hugging Face Account (Optional):** Access token configured in your shell if accessing private or gated checkpoints.

### Installation & Dependency Provisioning

No local Python toolchain is required on the host; all dependencies are encapsulated inside the container image.

Export your Hugging Face token (if applicable):

```bash
export HF_TOKEN="hf_your_actual_token_here"

```

### Build and Run for Development

Build the image from scratch and launch the service in detached mode:

```bash
# Build the container image without cache to ensure the latest cu130 wheels
docker compose build --no-cache

# Run the container
docker compose up -d

# Stream logs to monitor startup and model loading
docker compose logs -f laya-server

```

To stop the service:

```bash
docker compose down

```

### Verification and Test Suite Execution

*Note: The repository currently consists of deployment infrastructure files (`Dockerfile` and `docker-compose.yml`); no internal unit or integration test suite (e.g., `pytest`) is present in the source tree.*

#### 1. Healthcheck Verification

Verify that the service is running, CUDA is detected, and model revisions are loaded:

```bash
curl -s http://localhost:8090/health | jq .

```

Expected output should indicate `"status": "ok"` and `"device": "cuda"`.

#### 2. End-to-End Inference Smoke Test

Send a sample payload to the `/v1/systemone` endpoint:

```bash
curl -X POST http://localhost:8090/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": "O cliente relatou cobranca indevida na fatura e solicitou cancelamento imediato.",
    "questions": {
      "departamento": {
        "type": "choice",
        "instructions": "Para qual fila encaminhar?",
        "criteria": {
          "financeiro": "Cobranca indevida, estornos, faturas e boletos",
          "suporte_tecnico": "Falhas no sistema, lentidao, bugs e erros tecnicos",
          "comercial": "Planos, contratacao e negociacao de novos servicos"
        }
      },
      "risco_cancelamento": {
        "type": "noul",
        "instructions": "Existe risco explicito de cancelamento?"
      }
    }
  }'

```

---

## 4. Coding Conventions and Standards

Because this repository currently consists entirely of configuration and infrastructure code (Docker and YAML), conventions are enforced through infrastructure-as-code best practices:

* **Dockerfile Best Practices:**
* Multi-statement chaining (`&&`) with cleanup (`rm -rf /var/lib/apt/lists/*`) in `RUN` layers to minimize image footprint.
* Explicit virtual environment paths (`/opt/venv`) instead of polluting system Python packages.
* Pinned base image coordinates (`nvidia/cuda:13.0.0-runtime-ubuntu24.04`) for reproducible CUDA runtime layers.
* Standard environment naming: `DEBIAN_FRONTEND=noninteractive`, `PYTHONUNBUFFERED=1`.


* **Docker Compose Specifications:**
* Two-space indentation with standard YAML dictionary key mappings.
* Quoted memory sizes (`"16gb"`) and resource allocation declarations under the Compose V2 `deploy.resources` schema.
* Environment variable interpolation syntax with fallback defaults (`${HF_TOKEN:-}`).


* **Missing Project Code Standards:**
* *Python/Application Code Standards:* No application-level Python scripts are present in the root directory. If Python microservices, client SDKs, or pre/post-processing scripts are added, they should adhere to **PEP 8**, format with **Black/Ruff**, and type-check with **mypy**.



---

## 5. Key Design Decisions

### 1. PyTorch Nightly with CUDA 13.0 Wheels (`cu130`)

* **What:** The `Dockerfile` pulls PyTorch from the nightly index via `[https://download.pytorch.org/whl/nightly/cu130](https://download.pytorch.org/whl/nightly/cu130)` rather than the stable PyTorch releases (e.g., standard `cu124` or `cu126`).
* **Why:** The target deployment environment utilizes next-generation GPUs with the Blackwell architecture (Compute Capability `sm_120`), such as the RTX 5060 Ti, operated under NVIDIA Driver 580+ and CUDA 13.0. Stable PyTorch packages compiled for CUDA 12.x lack compiled binary kernels (SASS) for `sm_120`, causing runtime crashes (`RuntimeError: CUDA error: no kernel image is available for execution on the device`) during encoder tensor operations (such as `torch.arange`). Using the `cu130` nightly wheel ensures native SASS instructions for Blackwell SMs are present.

### 2. Host Network Mode (`network_mode: host`) and Direct Socket Binding

* **What:** The Compose service specifies `network_mode: host` and supplies server binding configuration via environment variables (`LAYA_HOST=0.0.0.0`, `LAYA_PORT=8090`).
* **Why:** In high-throughput, low-latency System 1 inference pipelines where requests typically execute within 15–30 ms, network bridge overhead, container NAT, and packet filtering induce measurable latency. Host networking bypasses the Docker bridge virtual switch, eliminating networking latency while simplifying port accessibility on internal compute hosts.

### 3. Bidirectional Volume Caching for Models and Preloading Enabled

* **What:** Host mounts are mapped for both `${HOME}/.cache/huggingface` and `${HOME}/.cache/receptron-laya`, paired with `LAYA_PRELOAD=1`.
* **Why:** Model weights for transformer encoders and decision heads are large and immutable across runtime executions. Mounting persistent host caches prevents container restarts from re-downloading checkpoints from the Hugging Face Hub, preserving network bandwidth and avoiding Hub rate limits. Enabling `LAYA_PRELOAD=1` forces immediate warm-up of CUDA contexts and model allocation into VRAM during container startup, preventing first-request latency spikes (cold starts).

---

## 6. Development Workflow

### Branching Strategy

* The repository follows a standard feature-branching workflow (e.g., GitHub Flow or GitFlow).
* The `main` branch represents the deployable, tested infrastructure state.
* Create short-lived feature or fix branches from `main`:
```bash
git checkout -b feat/add-prometheus-metrics
# or
git checkout -b fix/update-cuda-patch

```



### Contribution Guidelines

1. **Local Validation:**
* Before committing, execute `docker compose build --no-cache` to ensure all external dependencies (wheel URLs, APT packages) resolve cleanly.
* Verify container start-up with `docker compose up -d` and run the smoke-test `curl` against `/v1/systemone`.


2. **Submitting Changes:**
* Commit with clear, conventional commit messages (e.g., `feat(docker): upgrade cuda base image to 13.0.1`, `fix(compose): tune shm_size`).
* Push your branch to the remote repository and open a Pull Request (PR) targeted at `main`.


3. **Pull Request Standards:**
* **Telemetry & Logs:** Include output logs confirming successful container initialization (`Application startup complete`) and inference metrics.
* **Hardware Details:** State the target GPU, driver version, and CUDA version validated during testing.
* **Documentation:** Update the `README.md` if port bindings, mount paths, or environment variables are altered.
* **Review Approval:** At least one review approval from an infrastructure/MLOps maintainer is required prior to merging.
