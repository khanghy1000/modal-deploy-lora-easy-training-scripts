import modal
import tomllib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PYTHON_VERSION = "3.11"
LORA_BACKEND_REPO_URL = (
    "https://github.com/67372a/LoRA_Easy_Training_scripts_Backend.git"
)

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
        f"git clone -b refresh --recursive {LORA_BACKEND_REPO_URL} /lora_backend",
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
        gpu="any",
    )
    .workdir("/lora_backend/sd_scripts")
    .run_commands(
        "uv pip install --system -r requirements.txt",
        "uv pip install --system -e ../custom_scheduler/.",
        "uv pip install --system -r ../requirements.txt",
    )
    # Patch anima_train_leco.py
    .run_commands(
        'sed -i \'s/network.prepare_optimizer_params_with_multiple_te_lrs(None, unet_lr, args.learning_rate)/network.prepare_optimizer_params_with_multiple_te_lrs(None, unet_lr, args.learning_rate, getattr(args, "apply_orthograd", False), getattr(args, "orthograd_targets", []))/g\' anima_train_leco.py'
    )
    .run_commands(
        "mkdir -p /root/.cache/huggingface/accelerate",
        "printf \"compute_environment: LOCAL_MACHINE\\ndistributed_type: 'NO'\\nmachine_rank: 0\\nmain_training_function: main\\nmixed_precision: bf16\\nnum_machines: 1\\nnum_processes: 1\\nrdzv_backend: static\\nsame_network: true\\nuse_cpu: false\\n\" > /root/.cache/huggingface/accelerate/default_config.yaml",
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
    CONTAINER_IDLE_TIMEOUT = modal_settings.get("container_idle_timeout", 60)
    TIMEOUT = modal_settings.get("timeout", 180000)
    GPU_CONFIG = modal_settings.get("gpu", "L40S")
    CPU_CONFIG = modal_settings.get("cpu", 2)
    MEMORY_CONFIG = modal_settings.get("memory", 10240)
except Exception as e:
    CONTAINER_IDLE_TIMEOUT = 60
    TIMEOUT = 180000
    GPU_CONFIG = "L40S"
    CPU_CONFIG = 2
    MEMORY_CONFIG = 10240

app = modal.App(name="anima-leco-training", image=lora_image)


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
)
def train(
    ileco_prompt_pairs: str = "/dataset/anima_ileco_pairs.json",
    pretrained_model_name_or_path: str = "/models/anima-base-v10.safetensors",
    qwen3: str = "/models/qwen_3_600m.safetensors",
    output_dir: str = "/outputs",
    output_name: str = "anima_ileco_test",
    save_model_as: str = "safetensors",
    network_module: str = "networks.lora_anima",
    network_dim: int = 8,
    network_alpha: int = 8,
    learning_rate: str = "1e-4",
    optimizer_type: str = "AdamW8bit",
    lr_scheduler: str = "constant",
    max_train_steps: int = 500,
    mixed_precision: str = "bf16",
    add_reverse_pairs: bool = True,
    reverse_multiplier: float = -1.0,
    reverse_weight: float = 1.0,
    ileco_guidance_scale: float = 3.0,
    ileco_denoising_steps: int = 4,
    ileco_min_sigma: float = 0.2,
    ileco_max_sigma: float = 0.8,
    ileco_resolution: int = 512,
    ileco_batch_size: int = 1,
):
    import os
    import sys
    import subprocess

    os.chdir("/lora_backend/sd_scripts")

    cmd = [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process",
        "1",
        "anima_train_leco.py",
        f"--ileco_prompt_pairs={ileco_prompt_pairs}",
        f"--pretrained_model_name_or_path={pretrained_model_name_or_path}",
        f"--qwen3={qwen3}",
        f"--output_dir={output_dir}",
        f"--output_name={output_name}",
        f"--save_model_as={save_model_as}",
        f"--network_module={network_module}",
        f"--network_dim={network_dim}",
        f"--network_alpha={network_alpha}",
        f"--learning_rate={learning_rate}",
        f"--optimizer_type={optimizer_type}",
        f"--lr_scheduler={lr_scheduler}",
        f"--max_train_steps={max_train_steps}",
        f"--mixed_precision={mixed_precision}",
        f"--reverse_multiplier={reverse_multiplier}",
        f"--reverse_weight={reverse_weight}",
        f"--ileco_guidance_scale={ileco_guidance_scale}",
        f"--ileco_denoising_steps={ileco_denoising_steps}",
        f"--ileco_min_sigma={ileco_min_sigma}",
        f"--ileco_max_sigma={ileco_max_sigma}",
        f"--ileco_resolution={ileco_resolution}",
        f"--ileco_batch_size={ileco_batch_size}",
    ]

    if add_reverse_pairs:
        cmd.append("--add_reverse_pairs")

    print(f"Executing: {' '.join(cmd)}")
    sys.stdout.flush()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Training failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print("Training completed successfully!")


@app.local_entrypoint()
def main(
    ileco_prompt_pairs: str = "/dataset/anima_ileco_pairs.json",
    pretrained_model_name_or_path: str = "/models/anima-base-v10.safetensors",
    qwen3: str = "/models/qwen_3_600m.safetensors",
    output_dir: str = "/outputs",
    output_name: str = "anima_ileco_test",
    save_model_as: str = "safetensors",
    network_module: str = "networks.lora_anima",
    network_dim: int = 8,
    network_alpha: int = 8,
    learning_rate: str = "1e-4",
    optimizer_type: str = "AdamW8bit",
    lr_scheduler: str = "constant",
    max_train_steps: int = 500,
    mixed_precision: str = "bf16",
    add_reverse_pairs: bool = False,
    reverse_multiplier: float = -1.0,
    reverse_weight: float = 1.0,
    ileco_guidance_scale: float = 1.0,
    ileco_denoising_steps: int = 0,
    ileco_min_sigma: float = 0,
    ileco_max_sigma: float = 1,
    ileco_resolution: int = 512,
    ileco_batch_size: int = 1,
):
    print("Starting Anima iLECO training on Modal...")
    train.remote(
        ileco_prompt_pairs=ileco_prompt_pairs,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        qwen3=qwen3,
        output_dir=output_dir,
        output_name=output_name,
        save_model_as=save_model_as,
        network_module=network_module,
        network_dim=network_dim,
        network_alpha=network_alpha,
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
        lr_scheduler=lr_scheduler,
        max_train_steps=max_train_steps,
        mixed_precision=mixed_precision,
        add_reverse_pairs=add_reverse_pairs,
        reverse_multiplier=reverse_multiplier,
        reverse_weight=reverse_weight,
        ileco_guidance_scale=ileco_guidance_scale,
        ileco_denoising_steps=ileco_denoising_steps,
        ileco_min_sigma=ileco_min_sigma,
        ileco_max_sigma=ileco_max_sigma,
        ileco_resolution=ileco_resolution,
        ileco_batch_size=ileco_batch_size,
    )
