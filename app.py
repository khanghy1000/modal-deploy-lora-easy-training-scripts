import modal
import tomllib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PYTHON_VERSION = "3.11"
LORA_BACKEND_REPO_URL = "https://github.com/67372a/LoRA_Easy_Training_scripts_Backend.git"
LORA_BACKEND_REPO_BRANCH = "refresh"
LORA_BACKEND_REPO_COMMIT = "refresh" # Specify a commit hash to force update the image

lora_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python=PYTHON_VERSION
    )
    .env(
        {
            "DEBIAN_FRONTEND": "noninteractive",
            "TZ": "Etc/UTC",
        }
    )
    .apt_install(
        "git",
        "wget",
        "libgl1",
        "libglib2.0-0",
        "python3-tk",
        "libjpeg-dev",
        "libpng-dev",
        "google-perftools",
        "libgl1-mesa-dri",
    )
    .run_commands(
        "set -ex",
        "pip install --upgrade pip uv",
        f"git clone -b {LORA_BACKEND_REPO_BRANCH} --recursive {LORA_BACKEND_REPO_URL} /lora_backend",
    )
    .workdir("/lora_backend")
    .run_commands(
        "uv pip install --system -U typing-extensions==4.15.0",
        "uv pip install --system -U torch~=2.7.1 torchvision~=0.22.1 numpy~=2.2.6 --index-url https://download.pytorch.org/whl/cu128",
        "uv pip install --system -U --force-reinstall --no-deps git+https://github.com/67372a/RamTorch",
        "uv pip install --system -U --force-reinstall --no-deps git+https://github.com/67372a/customized-optimizers",
        "uv pip install --system -U --no-deps xformers==0.0.31.post1 --index-url https://download.pytorch.org/whl/cu128",
        "uv pip install --system -U --no-deps torchao~=0.13.0 --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128",
        "uv pip install --system -U --force-reinstall --no-deps git+https://github.com/67372a/LyCORIS@dev",
        "uv pip install --system --no-deps https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
    )
    .workdir("/lora_backend/sd_scripts")
    .run_commands(
        "uv pip install --system -r requirements.txt",
        "uv pip install --system -e ../custom_scheduler/.",
        "uv pip install --system -r ../requirements.txt",
    )
    .run_commands(
        "mkdir -p /root/.cache/huggingface/accelerate",
        """cat > /root/.cache/huggingface/accelerate/default_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcase_fp16: 'NO'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
use_cpu: false
EOF""",
    )
    .workdir("/lora_backend")
    .run_commands(
        f"git fetch origin {LORA_BACKEND_REPO_BRANCH}",
        f"git checkout {LORA_BACKEND_REPO_COMMIT}",
    )
    .run_commands(
        "echo 'Backend installation completed.'",
    )
)

logger.info("LoRA Backend image defined.")

CONFIG_FILE = Path(__file__).parent / "config.toml"

try:
    with open(CONFIG_FILE, "rb") as f:
        config = tomllib.load(f)
    modal_settings = config.get("modal_settings", {})
    backend_settings = config.get("backend_settings", {})
    ALLOW_CONCURRENT_INPUTS = modal_settings.get("allow_concurrent_inputs", 10)
    CONTAINER_IDLE_TIMEOUT = modal_settings.get("container_idle_timeout", 60)
    TIMEOUT = modal_settings.get("timeout", 180000)
    GPU_CONFIG = modal_settings.get("gpu", "L40S")
    CPU_CONFIG = modal_settings.get("cpu", 2)
    MEMORY_CONFIG = modal_settings.get("memory", 10240)
    PORT = backend_settings.get("port", 8000)
except Exception as e:
    ALLOW_CONCURRENT_INPUTS = 10
    CONTAINER_IDLE_TIMEOUT = 60
    TIMEOUT = 180000
    GPU_CONFIG = "L40S"
    CPU_CONFIG = 2
    MEMORY_CONFIG = 10240
    PORT = 8000

app = modal.App(name="lora-backend", image=lora_image)

class Paths:
    MODELS = "/models"
    DATASET = "/dataset"
    OUTPUTS = "/outputs"
    STATES = "/states"
    LOGS = "/logs"

# Define volumes
models_vol = modal.Volume.from_name("lora-models", create_if_missing=True)
dataset_vol = modal.Volume.from_name("lora-dataset", create_if_missing=True)
outputs_vol = modal.Volume.from_name("lora-outputs", create_if_missing=True)
states_vol = modal.Volume.from_name("lora-states", create_if_missing=True)
logs_vol = modal.Volume.from_name("lora-logs", create_if_missing=True)

@app.function(
    memory=MEMORY_CONFIG,
    cpu=CPU_CONFIG,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    scaledown_window=CONTAINER_IDLE_TIMEOUT,
    volumes={
        Paths.MODELS: models_vol,
        Paths.DATASET: dataset_vol,
        Paths.OUTPUTS: outputs_vol,
        Paths.STATES: states_vol,
        Paths.LOGS: logs_vol,
    },
    max_containers=1,
)
@modal.concurrent(max_inputs=ALLOW_CONCURRENT_INPUTS)
@modal.asgi_app()
def run_lora_backend():
    import os
    import sys
    import json
    
    os.chdir("/lora_backend")
    
    if "/lora_backend" not in sys.path:
        sys.path.insert(0, "/lora_backend")

    with open("config.json", "w", encoding="utf-8") as f:
        json.dump({"remote": False, "port": PORT, "host": "0.0.0.0"}, f)
        
    import main
    backend_app = main.app

    import threading
    import httpx
    import time
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.routing import Route
    from starlette.responses import JSONResponse

    async def keep_alive_ping(request):
        return JSONResponse({"status": "alive"})

    backend_app.routes.append(Route("/keep_alive_ping", keep_alive_ping))

    class KeepAliveMiddleware(BaseHTTPMiddleware):
        def __init__(self, app):
            super().__init__(app)
            self.public_url = None
            self.thread_started = False
            self.lock = threading.Lock()

        async def dispatch(self, request, call_next):
            if not self.public_url:
                with self.lock:
                    if not self.public_url:
                        host = request.headers.get("host")
                        if host and "modal.run" in host:
                            self.public_url = f"https://{host}"
                        elif host:
                            self.public_url = f"http://{host}"
            
            with self.lock:
                if self.public_url and not self.thread_started:
                    self.thread_started = True
                    threading.Thread(target=self.keep_alive_loop, daemon=True).start()
                    
            response = await call_next(request)
            return response
            
        def keep_alive_loop(self):
            while True:
                time.sleep(20)
                try:
                    is_training = False
                    if hasattr(backend_app.state, "TRAINING_THREAD"):
                        thread = backend_app.state.TRAINING_THREAD
                        if thread is not None and getattr(thread, "poll", lambda: None)() is None:
                            is_training = True
                    
                    if is_training and self.public_url:
                        httpx.get(f"{self.public_url}/keep_alive_ping", timeout=10)
                except Exception as e:
                    print(f"Keep-alive ping error: {e}")

    backend_app.add_middleware(KeepAliveMiddleware)

    return backend_app

@app.local_entrypoint()
def main():
    print("Run 'modal serve app.py' to start the backend server.")
