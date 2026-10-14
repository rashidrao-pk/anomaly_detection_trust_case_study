## Environment

Use Python 3.10 on a platform supported by TensorFlow 2.15.
Install the original dependency set for the root-level model compatibility module:

### using `venv`

```bash
python -m venv .venv
source .venv/bin/activate
```

### using `conda`

```bash
conda create -n xai-ad python=3.10 -y
conda activate xai-ad

python -m pip install --upgrade pip setuptools wheel
```

3. Install the repository dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[test]"
python -m pytest -q
```

## 2. Install GPU support for your operating system

### Apple Silicon (M1–M4, macOS)

```bash
conda activate xai-ad

python -m pip install tensorflow==2.15.0
python -m pip install tensorflow-metal

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Linux — NVIDIA CUDA

```bash
conda activate xai-ad

nvidia-smi

python -m pip install "tensorflow[and-cuda]==2.15.0"

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Windows — NVIDIA CUDA through WSL2

- First install WSL2 with an Ubuntu distribution, then open Ubuntu and run:

```bash
conda create -n xai-ad python=3.10 -y
conda activate xai-ad

python -m pip install "tensorflow[and-cuda]==2.15.0"

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Windows native / CPU-only

```bash
conda activate xai-ad

python -m pip install tensorflow==2.15.0

python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

## Verify Setup

```bash
python scripts/check_device.py
```
