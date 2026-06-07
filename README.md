# Run [67372a/LoRA_Easy_Training_Scripts_Backend](https://github.com/67372a/LoRA_Easy_Training_scripts_Backend) on Modal
Based on [IjoiK12/modal-deploy-kohya-ss](https://github.com/IjoiK12/modal-deploy-kohya-ss) and [ihsaiasdasm's article](https://civitai.red/articles/30846/guide-free-lora-training-on-modal).
## Prerequisites

Before you begin, ensure you have the following:

1.  **A Modal Account:** Sign up at [modal.com](https://modal.com). New users often receive free credits.
2.  **Modal Client Installed and Configured:**
    ```bash
    pip install modal
    modal setup
    ```
3.  **Python:** Python 3.11 or newer installed locally.
4.  **Git:** For cloning this repository.
5.  **Training Data:** Your base models (e.g., `.safetensors` files) and image datasets prepared.
6.  **LoRA Easy Training Scripts GUI:** Set up the GUI from [67372a/LoRA_Easy_Training_Scripts](https://github.com/67372a/LoRA_Easy_Training_scripts).

## Uploading Data (Models & Datasets)

Your models, datasets, and outputs will be stored in persistent Modal Volumes. You need to upload your base models and datasets to these volumes using the `modal volume put` command from your local terminal.

The `app.py` script maps these volumes to paths inside the container:
* `lora-models` volume is mounted at `/models/`
* `lora-dataset` volume is mounted at `/dataset/` (Note: singular "dataset" in the path as per your `app.py`)
* `lora-outputs` volume is mounted at `/outputs/`
* `lora-states` volume is mounted at `/states/`
* `lora-logs` volume is mounted at `/logs/`

**Uploading Base Models:**
   * Volume Name: `lora-models`
   * Example: If your model `my_sdxl_model.safetensors` is locally at `C:\AI\Models\my_sdxl_model.safetensors`:
        ```bash
        modal volume put lora-models C:\AI\Models\my_sdxl_model.safetensors my_sdxl_model.safetensors
        ```
        This makes the model available inside the container at `/models/my_sdxl_model.safetensors`.

**Uploading Datasets:**
   * Volume Name: `lora-dataset`
   * Example: If your processed dataset folder (e.g., `mycharacter_style`) is locally at `D:\TrainingData\my_style_project\mycharacter_style`:
        ```bash
        modal volume put lora-dataset D:\TrainingData\my_style_project\mycharacter_style mycharacter_style
        ```
        This makes the dataset available inside the container at `/dataset/mycharacter_style/`. When using the lora GUI, you would set "Image folder" to `/dataset/`.

**Verifying Volume Contents:**
   * You can list the contents of your volumes:
        ```bash
        modal volume ls lora-models -r
        modal volume ls lora-dataset -r
        ```

## Running the Application

You have two primary ways to run the application:

1.  **Temporary Run (for Development/Testing):**
    ```bash
    modal serve app.py
    ```
    The application will run as long as this command is active in your terminal. Modal will provide a temporary URL to access the GUI. Press `Ctrl+C` to stop.

2.  **Persistent Deployment:**
    ```bash
    modal deploy app.py
    ```
    This deploys the application to Modal, where it will run in the background and be accessible via a persistent URL. You can close your terminal. To update the deployment after code changes, run this command again.

Modal will output the URL (e.g., `https://your_username--lora-backend-run-lora-backend-dev.modal.run`) for the backend. Put the URL in the "Backend Server URL" field of the [GUI](https://github.com/67372a/LoRA_Easy_Training_Scripts) to connect.

## Downloading Results

Your trained models (LoRA files, etc.) will be saved to the `/outputs/` directory within the `lora-outputs` volume. Use `modal volume get` to download them:

```bash
modal volume get lora-outputs /my_awesome_lora.safetensors C:\LoRAs\my_awesome_lora.safetensors
```
