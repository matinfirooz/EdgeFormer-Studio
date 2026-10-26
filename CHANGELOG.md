# Changelog

## 0.2.0 — 2026-10-25

### Correctness

- Fix zero gradients through QAT rounding using an explicit identity straight-through estimator.
- Preserve dtype, device, frozen parameters and train/eval mode during layer conversion; support a root Linear and selective named layers.
- Reject invalid precision values, unsupported conversion combinations, empty evaluation sets and non-finite training losses.
- Count configured input channels in patch projection MACs.
- Make benchmark `--bits` actually apply the stated simulation, while reporting floating execution and actual runtime tensor sizes.
- Preserve module modes and RNG when timing; synchronize the selected CUDA device.
- Keep biases and normalization floating in new PTQ conversions; support per-channel scales by default.

### Workflows

- Add unified CLI, validation, clipping, cosine scheduling, best/latest checkpoints and exact epoch-boundary CPU resume tests.
- Add explicit dataset choice, download control, deterministic synthetic patterns and original/prepared Tiny ImageNet validation layouts.
- Add matched precision/noise sweeps, output/attention distortion, one-layer sensitivity, mixed precision plans and accuracy/storage nondomination.
- Add attention rollout, runnable notebooks, offline reproduction script, optional dependency groups, CI and expanded tests.
- Replace obsolete concept illustrations with editable diagrams tied to the implementation; remove empty documentation placeholders.

### Compatibility

- Original TinyViT state-dict keys and `model`/`config` checkpoints remain loadable. SDPA may introduce small floating differences from the original explicit attention path; use `attention_backend: manual` when comparing implementations.
- Original script entry points remain wrappers around the unified CLI. `compare_models.py` now evaluates data; use `--config configs/smoke.yaml` for an offline run.
- `quantize --precision int8/int4` remains a legacy alias. New checkpoints carry explicit transform metadata.
- PTQ numerical defaults changed: per-output-channel weight scales and floating biases. Re-run old experiments rather than treating the old and new PTQ paths as equivalent.
- Existing legacy `bits` checkpoints retain their stored floating values and are not requantized on load.
- `parameter_memory_mb` is retained as a legacy uniform-storage helper, measured in MiB. New reports use byte counts and include nonquantized parameters plus scale overhead.
- Attention maps now contain pre-dropout probabilities. Old code that expected dropped probabilities should apply dropout explicitly.

### Scope

CPU software behavior and synthetic end-to-end workflows are verified. This release does not provide pretrained real-dataset accuracy results, integer runtime kernels, accelerator RTL, or silicon power/area measurements.
