# Numerical contract

## Symmetric Q/DQ

For bit width `b`, `qmax = 2^(b-1)-1`, scale `s = max(abs(x))/qmax`, integer values `q = clamp(round(x/s), -qmax, qmax)`, and floating output `x_hat = q*s`. Rounding follows PyTorch's rounding semantics. A positive floating-point minimum protects zero/tiny scales. Zero tensors return zero. Supported widths are integers 2–16; 32 selects identity.

This is a narrow symmetric range: W4 uses -7…7 and W8 uses -127…127. It deliberately does not use the extra negative two's-complement code. Per-channel reduction preserves the selected channel axis, normally axis zero for Conv/Linear output channels. PTQ keeps bias, normalization weights, class and position embeddings floating.

## QAT and gradients

QAT uses floating master parameters. In forward, each supported operator fake quantizes its input tensor dynamically and its weight per output channel. Bias is floating. Its surrogate is `x + detach(QDQ(x)-x)`: the forward values equal Q/DQ while the surrogate derivative with respect to `x` is one. The scale is computed from detached data. This is an identity STE, without a learned scale or clipping-gradient mask.

QAT covers patch Conv2d, QKV/output Linear, MLP Linear and classifier Linear. It does not quantize attention matrix multiplication, softmax, LayerNorm, GELU or residual arithmetic. Dynamic input maxima are recomputed on each forward, including evaluation; there is no calibration observer. Batch composition can therefore affect activation scales and results. Record and match batch size when comparing QAT runs.

The implementation's finite-input validation and dynamic maxima add overhead. QAT runtime is the cost of this simulator, not an estimate of integer deployment speed. Checkpoint transform metadata rebuilds wrappers before loading the state dict.

## Statistical approximate arithmetic

During evaluation, an `ApproxLinear` adds independent Gaussian noise with standard deviation `noise_scale * max(mean(abs(output)), 1e-8)`. Training is exact for these wrappers. This models output sensitivity to a chosen error distribution, not the behavior of a particular arithmetic circuit. Results depend on seed, batch composition, input order and noise scale. The approximation model cannot replace a QAT/custom Linear because doing so would discard its forward semantics.

## Attention

Manual attention computes `softmax(QK^T/sqrt(d))`, applies attention dropout, multiplies by V and projects. The SDPA path passes zero dropout in evaluation. Requests for attention maps select the manual path; returned maps are pre-dropout probabilities. Block residuals and pre-normalization are unchanged from the original model.

The rollout helper averages heads, adds identity to represent residual mixing, normalizes each row and multiplies transitions in layer order. It uses the CLS-to-patch entries of the resulting matrix. Rollout is a visualization heuristic.

## Distortion

- Logit MAE: mean absolute logit difference over all selected samples and classes.
- Logit RMSE: square root of mean squared logit difference over the same values.
- Agreement: fraction of identical top-1 predictions.
- Cosine: mean cosine similarity across samples; zero vectors follow PyTorch's epsilon convention.
- Attention MAE: mean absolute probability difference, then mean across blocks.
- Attention JS: half the sum of KL divergences to the midpoint distribution, using natural logs, averaged over rows/heads and blocks. A small epsilon protects the logarithm and rows are renormalized.

Sweep logit metrics use all cached batches. Attention metrics use only the first image. Accuracy/storage dominance requires no worse values in both objectives and a strict improvement in at least one. Exact duplicate objective points remain nondominated; repeats are not aggregated automatically.

## References

- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
- [PyTorch quantization-aware training and STE](https://pytorch.org/blog/quantization-aware-training/).
- [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).
