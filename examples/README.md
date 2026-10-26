# Offline evidence

`results/` contains actual outputs of `python scripts/reproduce_examples.py` using `configs/smoke.yaml`. The data are four synthetic image-pattern classes. Every report identifies that scope. Floating runtime varies by host and load; numerical results may vary across PyTorch versions/backends.

| Output | What it demonstrates |
| --- | --- |
| `fp32_history.csv`, `fp32_run.json`, `training.png` | Training/validation, metadata and the cosine schedule. |
| `evaluation.json` | Floating reference evaluation. |
| `benchmark_w32/w8/w4.json` | Actual floating forward timings and separate analytical storage/MAC quantities. |
| `sweep.csv`, `sweep.json`, `sweep.png` | Four precisions × two noise scales × two error seeds, on identical evaluation samples. |
| `sensitivity.csv/json/png` | Each Conv/Linear layer quantized to W4 independently. |
| `attention.png` | Per-block CLS maps and residual-aware rollout. |
| `qat_w4_history.csv`, `qat_w4_run.json`, `evaluation_qat_w4.json` | QAT fine-tuning with simulated W4 inputs/weights. |
| `manifest.json` | Scope and reproduction command. |

No pretrained checkpoints are included in the distribution. The script regenerates them in `runs/example/`. Synthetic accuracy and these host timings must not be reported as CIFAR/ImageNet performance or accelerator results. Top-5 accuracy is trivially one for this four-class task.
