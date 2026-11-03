# Tested environments

These exact version pins were exported from environments that passed actual
SmolLM2 B-only optimization smoke tests on 2 November 2026. They are environment
snapshots, not cross-platform universal locks or wheel-hash lockfiles.

| File | Tested scope |
| --- | --- |
| [cpu-windows-py311.txt](cpu-windows-py311.txt) | Windows x64, Python 3.11.15, CPU training and Streamlit query console |
| [cuda-wsl-py311.txt](cuda-wsl-py311.txt) | Ubuntu WSL2 x86_64, Python 3.11.15, PyTorch CUDA 12.6, Flower 1.40.0 with Ray 2.55.1 |

The Windows environment intentionally does not include Flower/Ray. Use WSL2 for
the five-client simulation. The CUDA run used an NVIDIA RTX 3050 Laptop GPU with
6 GB dedicated memory. CPU and GPU arithmetic use FP32.

## Install on a new machine

From the repository root, with `uv` installed:

```powershell
uv venv --python 3.11 .venv-cpu
uv pip install --python .venv-cpu/Scripts/python.exe --index-strategy unsafe-best-match -r requirements/cpu-windows-py311.txt
uv pip install --python .venv-cpu/Scripts/python.exe --no-deps -e .
```

For Ubuntu/WSL2, run these commands inside Linux:

```bash
uv venv --python 3.11 ~/.venvs/loflwr
uv pip install --python ~/.venvs/loflwr/bin/python --index-strategy unsafe-best-match -r requirements/cuda-wsl-py311.txt
uv pip install --python ~/.venvs/loflwr/bin/python --no-deps -e .
```

The files specify the official PyTorch wheel index for their respective builds.
Other dependencies come from PyPI. On systems using `pip`, replace `uv pip
install --python PATH --index-strategy unsafe-best-match` with `PATH -m pip install`.
The explicit uv index strategy permits the exact pins to resolve across PyPI
and the PyTorch wheel index. Creating the environment and
installing wheels needs internet access; runtime model loading does not.

Prepare the model, common adapter and dataset before going offline using the
[project setup instructions](../README.md). A new machine also needs a compatible
NVIDIA driver for CUDA; installing the wheel does not install that driver.

## Existing demonstration laptop

The tested environments already exist at:

- Windows CPU/UI: `C:\Users\samee\.venvs\loflwr-cpu\Scripts\python.exe`
- WSL training/Flower: `/home/sameer/.venvs/loflwr/bin/python`

Use the [morning guide](../docs/morning-demo.md) to launch without reinstalling.
The repository itself is installed separately because a machine-specific
editable path is not portable. Package pins include all observed installed
dependencies, including supporting analysis packages. They do not include model
weights, data, credentials, or private paths.

Repeat the actual model and Flower smoke tests after changing these versions.
The dependency-free tests alone do not verify training or transport compatibility.
