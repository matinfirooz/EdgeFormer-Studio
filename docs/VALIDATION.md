# Validation record

Validated on 2026-10-25 with Python 3.12.14, PyTorch 2.5.1+cpu and torchvision 0.20.1+cpu on Linux. CPU intra-op threads were set to one for the regression and smoke workflows. Dependency ranges describe supported installation intent; they are not a claim that every version/platform combination was tested.

| Check | Observed result |
| --- | --- |
| `python -m pytest -q` | 51 tests passed. |
| Ruff lint and formatting | Passed. |
| Full offline reproduction | FP32 training, evaluation, W4 PTQ save, W32/W8/W4 benchmarks, 16-point precision/noise sweep, 10-layer sensitivity, attention report and W4 QAT fine-tuning/evaluation completed. |
| QAT backward pass | Nonzero finite weight gradients in all Conv/Linear wrappers. |
| Attention implementation parity | Manual and SDPA forward values and input gradients agree in float64 within 1e-10 in the tested CPU configuration. |
| Interrupted training | Final model tensors and histories exactly matched uninterrupted runs for both FP32 and W4 QAT, using completed-epoch checkpoints on synthetic data. |
| Checkpoint compatibility | Floating, PTQ, QAT and legacy `model`/`config` state-dict round trips passed. |
| MAC accounting | Analytical total matched actual Conv/Linear forward-hook counts plus attention matrix products; grayscale patch projection checked. |
| Dataset behavior | Deterministic synthetic loaders and original Tiny ImageNet annotation/class mapping checked locally. |
| Notebook code | Every code cell in all four notebooks executed in order with the noninteractive Matplotlib backend. Outputs in the distributed notebooks remain cleared. |
| Distribution | Python wheel built; its CLI benchmark ran from outside the source tree. |
| Documentation and figures | Local Markdown links and SVG XML checked; PNG diagrams and generated analysis plots inspected. |

The smoke configuration has 19,428 parameters and 340,224 MACs per image. Its W4 ideal packed parameters plus FP32 scales total 16,480 bytes, while actual floating parameter tensors still occupy 77,712 bytes. These are implementation-derived storage counts. Smoke accuracy is for synthetic patterns only.

CUDA runtime, Windows execution, full CIFAR/Tiny ImageNet training, deployment kernels and physical hardware PPA were not measured. GitHub Actions is configured for Python 3.10/3.12 CPU checks; remote CI execution is not claimed in this validation record.

To reproduce the main checks:

```bash
python -m pip install -e '.[dev,data]'
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python scripts/reproduce_examples.py
```
