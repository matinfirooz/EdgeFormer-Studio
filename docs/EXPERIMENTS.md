# Running a defensible precision study

1. Train and retain a floating reference. Record the resolved configuration, checkpoint and dataset partition. Use independent validation for tuning when the final test partition must remain untouched.
2. Establish reference accuracy on the exact samples used for candidate comparison. Keep resize/normalization, batch size and class mapping identical.
3. Run weight-only W8/W6/W4 PTQ from that same reference. Do not repeatedly requantize an already rounded checkpoint.
4. Inspect per-layer sensitivity on a fixed subset. Protect sensitive layers in an explicit JSON mixed-precision plan, then validate the complete plan separately.
5. Train QAT candidates from the floating reference and report that they receive additional optimization. Compare equivalent fine-tuning budgets when attributing accuracy differences to QAT.
6. Repeat stochastic noise experiments across seeds. The built-in sweep repeats error realizations, not model training. Aggregate seeds and report uncertainty before making general claims.
7. Benchmark on a stated device and thread count with matched batch size, warmup and runs. Retain the JSON environment metadata. Compare simulator runtime separately from deployment runtime.
8. Use packed bytes as a theoretical storage target. The simulator retains floating tensors and does not emit a packed deployment artifact.
9. Replace the compute-energy coefficient and quadratic scaling with characterized target costs before interpreting energy. Memory traffic, activation storage, nonlinear units and utilization need their own model.

## Analytical MACs

For `N` tokens, width `D`, hidden MLP width `M` and `L` blocks:

| Component | MACs per image |
| --- | ---: |
| Patch projection | `(N-1) * P^2 * C * D` |
| QKV | `L * 3 * N * D^2` |
| QK and AV | `L * 2 * N^2 * D` |
| Attention output projection | `L * N * D^2` |
| MLP | `L * 2 * N * D * M` |
| Classifier | `D * classes` |

No multiply-add factor of two is applied. Bias additions, normalization, softmax, GELU and residual additions are excluded. The profiler reads each block's actual MLP size rather than assuming all MLPs have the same ratio.

## Storage interpretation

Quantized Conv/Linear weights receive their configured bit width; other parameters retain their runtime element size. Add FP32 scales, one per weight output channel or per tensor. Ideal packing has no alignment padding. Runtime parameter bytes sum the actual parameter tensors; neither quantity measures total process memory.

The attention storage comparison counts a score/probability matrix pair versus a row-sized pair. It excludes Q/K/V, parameter memory, training activations and backend workspace. It illustrates scaling and does not claim the software implements a row-streaming accelerator.

## Release evidence

The bundled outputs use synthetic patterns, CPU execution, fixed seed 42, width 32, two blocks and 16×16 images. They verify execution and serialization. They cannot establish realistic quantization accuracy, accelerator energy or conference-level hardware results.
