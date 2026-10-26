"""Command-line experiment workflows. Run ``python -m edgeformer --help``."""

import argparse
import copy
import json
import sys
from pathlib import Path
import torch
from edgeformer.models import TinyViT
from edgeformer.engine import evaluate, train_epoch
from edgeformer.quantization import quantize_model_weights, prepare_qat
from edgeformer.quantization.fake_quant import validate_bits
from edgeformer.approximate import replace_linear_with_approx
from edgeformer.profiling.macs import estimate_vit_macs
from edgeformer.profiling.memory import storage_report, attention_storage_report
from edgeformer.profiling.energy import estimate_energy_mj
from edgeformer.profiling.latency import benchmark_latency
from edgeformer.analysis import output_distortion, attention_distortion, pareto_mask
from edgeformer.analysis.sensitivity import cache_batches, layer_sensitivity
from edgeformer.utils.config import load_config, validate_training_config
from edgeformer.utils.dataset import loaders_from_config
from edgeformer.utils.seed import set_seed, capture_rng, restore_rng
from edgeformer.utils.checkpoint import load_checkpoint, save_checkpoint
from edgeformer.utils.io import write_json, write_csv, environment


def _common(parser):
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset", choices=["synthetic", "cifar10", "cifar100", "tiny_imagenet"])
    parser.add_argument("--data-dir")
    parser.add_argument("--download", action="store_true", help="Allow CIFAR download")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1, help="CPU intra-op threads (default: 1)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-batches", type=int)


def parser():
    p = argparse.ArgumentParser(prog="edgeformer", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    train = sub.add_parser(
        "train", help="Train, validate and save resumable best/latest checkpoints"
    )
    _common(train)
    train.add_argument("--epochs", type=int)
    train.add_argument("--resume", help="Resume a latest checkpoint with optimizer and RNG states")
    train.add_argument("--qat-bits", type=int)
    train.add_argument("--deterministic", action="store_true")
    train.add_argument("--output", default="runs/train")
    ev = sub.add_parser("evaluate", help="Evaluate FP32/PTQ/QAT with sample-weighted metrics")
    _common(ev)
    ev.add_argument("--bits", type=int, default=32)
    ev.add_argument("--approximate", action="store_true")
    ev.add_argument("--noise-scale", type=float, default=0.01)
    ev.add_argument("--output", default="results/evaluation.json")
    q = sub.add_parser("quantize", help="Save a weight-only fake PTQ checkpoint")
    _common(q)
    q.add_argument("--bits", type=int, default=8)
    q.add_argument("--precision", choices=["int8", "int4"], help="Legacy alias for --bits")
    q.add_argument("--per-tensor", action="store_true")
    q.add_argument("--plan", help="JSON mapping of module paths to weight bit widths")
    q.add_argument("--output")
    b = sub.add_parser(
        "benchmark", help="Measure actual floating runtime and report analytical costs"
    )
    _common(b)
    b.add_argument("--bits", type=int, default=32)
    b.add_argument("--runs", type=int, default=30)
    b.add_argument("--warmup", type=int, default=5)
    b.add_argument("--pj-per-mac-int8", type=float, default=0.25)
    b.add_argument("--output", default="results/benchmark.json")
    sweep = sub.add_parser(
        "sweep",
        aliases=["compare"],
        help="Evaluate matched precision/error variants and Pareto set",
    )
    _common(sweep)
    sweep.add_argument("--bits", type=int, nargs="+", default=[32, 8, 4])
    sweep.add_argument("--noise-scales", type=float, nargs="+", default=[0.0, 0.01])
    sweep.add_argument("--runs", type=int, default=10)
    sweep.add_argument("--warmup", type=int, default=3)
    sweep.add_argument(
        "--repeats", type=int, default=1, help="Repeat seeds for statistical error models"
    )
    sweep.add_argument("--output", default="results/sweep.csv")
    sen = sub.add_parser("sensitivity", help="Quantize one layer at a time on fixed batches")
    _common(sen)
    sen.add_argument("--bits", type=int, default=4)
    sen.add_argument("--output", default="results/sensitivity.csv")
    attn = sub.add_parser(
        "attention", help="Save per-layer CLS attention and rollout on one evaluation image"
    )
    _common(attn)
    attn.add_argument("--output", default="results/attention.png")
    return p


def _setup(args):
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max-batches must be positive")
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    cfg = load_config(args.config)
    checkpoint = getattr(args, "resume", None) or args.checkpoint
    state = None
    if checkpoint:
        model, state = load_checkpoint(checkpoint, device)
        # Legacy checkpoints lacked data settings; infer the original real dataset.
        saved = state["config"]
        for key, value in saved.items():
            if isinstance(value, dict) and key in cfg:
                cfg[key].update(value)
            else:
                cfg[key] = copy.deepcopy(value)
        cfg["model"] = dict(model.config)
        if "data" not in saved:
            cfg["data"]["dataset"] = {10: "cifar10", 100: "cifar100", 200: "tiny_imagenet"}.get(
                model.head.out_features, "synthetic"
            )
    else:
        set_seed(cfg["seed"] if args.seed is None else args.seed)
        model = TinyViT(**cfg["model"]).to(device)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.download:
        cfg["data"]["download"] = True
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    set_seed(cfg["seed"])
    return model, cfg, state, device


def _transform(model, bits, noise=0.0):
    validate_bits(bits)
    kind = getattr(model, "_edgeformer_transform", {}).get("kind", "fp32")
    if bits != 32:
        if kind in {"qat", "ptq"}:
            raise ValueError(
                "This checkpoint already defines its precision; omit --bits or use the floating reference"
            )
        model = quantize_model_weights(model, bits)
    if noise:
        model = replace_linear_with_approx(model, noise)
    return model.eval()


def _effective_storage(model):
    meta = getattr(model, "_edgeformer_transform", {})
    return storage_report(
        model,
        meta.get("bits", 32),
        per_channel=meta.get("per_channel", True),
        layer_bits=meta.get("layer_bits", {}),
    )


def _sample(model, batch_size=1, device="cpu", seed=42):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(
        batch_size, model.in_channels, model.image_size, model.image_size, generator=g
    ).to(device)


def run_train(args):
    if args.checkpoint and args.resume:
        raise ValueError("Use either --checkpoint (fine tune) or --resume")
    model, cfg, state, device = _setup(args)
    if args.resume:
        required = ["optimizer", "scheduler", "rng", "loader_rng", "epoch", "transform"]
        missing = [key for key in required if key not in state]
        if missing or "data" not in state["config"]:
            raise ValueError(
                "Checkpoint is not resumable; use --checkpoint to start a fine-tuning run"
            )
    if args.epochs is not None:
        if args.resume and args.epochs != cfg["training"]["epochs"]:
            raise ValueError(
                "Resume keeps the original cosine schedule; use --checkpoint to start a new schedule"
            )
        cfg["training"]["epochs"] = args.epochs
    if args.qat_bits is not None:
        if args.resume:
            if (
                state["transform"].get("kind") != "qat"
                or state["transform"]["bits"] != args.qat_bits
            ):
                raise ValueError("Resume must preserve the checkpoint QAT precision")
        else:
            model = prepare_qat(model, args.qat_bits)
    if getattr(model, "_edgeformer_transform", {}).get("kind") == "ptq":
        raise ValueError(
            "Fine tuning a PTQ copy removes quantization constraints; use the floating checkpoint with --qat-bits"
        )
    if args.resume:
        if (
            cfg["seed"] != state["config"]["seed"]
            or cfg["data"] != state["config"]["data"]
            or cfg["training"] != state["config"]["training"]
        ):
            raise ValueError(
                "Resume preserves data, seed and training settings; use --checkpoint for a new experiment"
            )
        if args.max_batches != state.get("max_batches"):
            raise ValueError("Resume requires the original --max-batches setting")
        args.deterministic = state.get("deterministic", args.deterministic)
    validate_training_config(cfg)
    set_seed(cfg["seed"], deterministic=args.deterministic)
    train_loader, val_loader = loaders_from_config(cfg)
    t = cfg["training"]
    opt = torch.optim.AdamW(
        model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    start, best, history = 1, -1.0, []
    if args.resume:
        for key in ["optimizer", "scheduler", "rng", "loader_rng", "epoch"]:
            if key not in state:
                raise ValueError(f"Checkpoint has no {key}; use --checkpoint for fine tuning")
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        train_loader.generator.set_state(state["loader_rng"].cpu())
        start = state["epoch"] + 1
        best = state.get("best_accuracy", -1.0)
        history = state.get("history", [])
    output = Path(args.output)
    if output.suffix == ".pt":
        latest = output
        best_path = output.with_name(output.stem + ".best.pt")
        history_path = output.with_name(output.stem + "_history.csv")
        metadata = output.with_name(output.stem + "_run.json")
    else:
        latest, best_path = output / "latest.pt", output / "best.pt"
        history_path, metadata = output / "history.csv", output / "run.json"
    write_json(
        metadata,
        dict(
            config=cfg,
            environment=environment(device),
            synthetic=cfg["data"]["dataset"] == "synthetic",
            deterministic=args.deterministic,
            max_batches=args.max_batches,
        ),
    )
    for epoch in range(start, t["epochs"] + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            opt,
            device,
            grad_clip=t["grad_clip"],
            label_smoothing=t["label_smoothing"],
            max_batches=args.max_batches,
        )
        val_metrics = evaluate(model, val_loader, device, args.max_batches)
        row = dict(
            epoch=epoch,
            learning_rate=opt.param_groups[0]["lr"],
            train_loss=train_metrics["loss"],
            train_accuracy=train_metrics["accuracy"],
            val_loss=val_metrics["loss"],
            val_accuracy=val_metrics["accuracy"],
            train_samples=train_metrics["samples"],
            val_samples=val_metrics["samples"],
        )
        history.append(row)
        improved = val_metrics["accuracy"] > best
        best = max(best, val_metrics["accuracy"])
        scheduler.step()
        payload = dict(
            epoch=epoch,
            optimizer=opt.state_dict(),
            scheduler=scheduler.state_dict(),
            rng=capture_rng(),
            loader_rng=train_loader.generator.get_state(),
            best_accuracy=best,
            history=history,
            max_batches=args.max_batches,
            deterministic=args.deterministic,
        )
        save_checkpoint(model, cfg, latest, **payload)
        if improved:
            save_checkpoint(model, cfg, best_path, **payload)
        write_csv(history_path, history)
        print(json.dumps(row))
    print(f"Latest checkpoint: {latest}; best validation accuracy: {best:.4f}")


def run_evaluate(args):
    if not args.checkpoint:
        raise ValueError(
            "Evaluation requires --checkpoint; use train with configs/smoke.yaml for an offline example"
        )
    model, cfg, _, device = _setup(args)
    model = _transform(model, args.bits, args.noise_scale if args.approximate else 0.0)
    _, loader = loaders_from_config(cfg, eval_only=True)
    result = dict(
        metrics=evaluate(model, loader, device, args.max_batches),
        transform=getattr(model, "_edgeformer_transform", {"kind": "fp32"}),
        approximation_noise=args.noise_scale if args.approximate else 0.0,
        dataset=cfg["data"]["dataset"],
        seed=cfg["seed"],
        synthetic=cfg["data"]["dataset"] == "synthetic",
        environment=environment(device),
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


def run_quantize(args):
    if not args.checkpoint:
        raise ValueError("--checkpoint is required")
    model, cfg, _, _ = _setup(args)
    bits = {"int8": 8, "int4": 4}[args.precision] if args.precision else args.bits
    plan = json.loads(Path(args.plan).read_text()) if args.plan else None
    model = quantize_model_weights(model, bits, per_channel=not args.per_tensor, layer_bits=plan)
    output = args.output or f"checkpoints/vit_fake_int{bits}.pt"
    save_checkpoint(model, cfg, output)
    print(f"Saved floating weight-only PTQ simulation: {output}")


def run_benchmark(args):
    model, cfg, state, device = _setup(args)
    model = _transform(model, args.bits)
    sample = _sample(model, args.batch_size or 1, device, cfg["seed"])
    macs = estimate_vit_macs(model)
    meta = getattr(model, "_edgeformer_transform", {})
    # This uniform proxy deliberately does not claim a mixed-precision cost.
    proxy_bits = meta.get("bits", 32)
    result = dict(
        environment=environment(device),
        checkpoint_loaded=state is not None,
        execution="floating point; quantization is simulated",
        model=cfg["model"],
        transform=meta,
        macs_per_image=macs,
        storage=_effective_storage(model),
        attention_storage=attention_storage_report(model, sample.shape[0]),
        latency=benchmark_latency(model, sample, args.warmup, args.runs, return_stats=True),
        uniform_compute_proxy_mj=estimate_energy_mj(macs["total"], proxy_bits, args.pj_per_mac_int8)
        if not meta.get("layer_bits")
        else None,
        proxy_assumptions=dict(
            pj_per_mac_int8=args.pj_per_mac_int8,
            bits=proxy_bits,
            applicable=not bool(meta.get("layer_bits")),
            scaling="quadratic bit-width heuristic; uncalibrated; excludes memory and nonlinear operations",
        ),
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


def run_sweep(args):
    model, cfg, state, device = _setup(args)
    if getattr(model, "_edgeformer_transform", {}).get("kind", "fp32") != "fp32":
        raise ValueError("Sweep requires the floating reference checkpoint")
    if args.repeats < 1 or not args.bits or not args.noise_scales:
        raise ValueError("Sweep needs positive repeats and nonempty design lists")
    _, loader = loaders_from_config(cfg, eval_only=True)
    batches = cache_batches(loader, args.max_batches or len(loader))
    model.eval()
    sample = batches[0][0].to(device)
    with torch.inference_mode():
        reference = torch.cat([model(x.to(device)).cpu() for x, _ in batches])
        _, ref_maps = model(sample[:1], return_attention=True)
    rows = []
    for bits in dict.fromkeys(args.bits):
        for noise in dict.fromkeys(args.noise_scales):
            for repeat in range(args.repeats):
                seed = cfg["seed"] + repeat
                set_seed(seed)
                candidate = _transform(model, bits, noise).to(device)
                metrics = evaluate(candidate, batches, device)
                set_seed(seed)
                with torch.inference_mode():
                    logits = torch.cat([candidate(x.to(device)).cpu() for x, _ in batches])
                    _, maps = candidate(sample[:1], return_attention=True)
                timing = benchmark_latency(
                    candidate, sample, args.warmup, args.runs, return_stats=True
                )
                storage = _effective_storage(candidate)
                rows.append(
                    dict(
                        name=f"W{bits}-noise{noise:g}-seed{seed}",
                        bits=bits,
                        noise_scale=noise,
                        seed=seed,
                        accuracy=metrics["accuracy"],
                        loss=metrics["loss"],
                        samples=metrics["samples"],
                        **output_distortion(reference, logits),
                        **attention_distortion(ref_maps, maps),
                        ideal_packed_total_bytes=storage["ideal_packed_total_bytes"],
                        runtime_parameter_bytes=storage["runtime_parameter_bytes"],
                        latency_ms=timing["median_ms"],
                        latency_p95_ms=timing["p95_ms"],
                    )
                )
    # Accuracy/storage Pareto set avoids ranking noisy host timings as ASIC results.
    for row, keep in zip(rows, pareto_mask(rows)):
        row["pareto_accuracy_storage"] = keep
    write_csv(args.output, rows)
    write_json(
        str(Path(args.output).with_suffix(".json")),
        dict(
            config=cfg,
            environment=environment(device),
            checkpoint_loaded=state is not None,
            synthetic=cfg["data"]["dataset"] == "synthetic",
            scope="Fixed evaluation batches; floating runtime; accuracy is fraction; JS uses natural logarithms",
            rows=rows,
        ),
    )
    from edgeformer.visualization.tradeoff_plot import save_tradeoff

    save_tradeoff(
        rows,
        str(Path(args.output).with_suffix(".png")),
        title=f"{cfg['data']['dataset']} | floating simulation | {len(rows)} design points",
    )
    print(f"Saved {len(rows)} matched design points: {args.output}")


def run_sensitivity(args):
    model, cfg, _, device = _setup(args)
    if getattr(model, "_edgeformer_transform", {}).get("kind", "fp32") != "fp32":
        raise ValueError("Sensitivity requires a floating reference")
    _, loader = loaders_from_config(cfg, eval_only=True)
    batches = cache_batches(loader, args.max_batches or 4)
    rows = layer_sensitivity(model, batches, args.bits, device)
    write_csv(args.output, rows)
    write_json(
        str(Path(args.output).with_suffix(".json")),
        dict(
            config=cfg,
            environment=environment(device),
            samples=sum(y.numel() for _, y in batches),
            synthetic=cfg["data"]["dataset"] == "synthetic",
            rows=rows,
        ),
    )
    print(f"Saved {len(rows)} layer ablations: {args.output}")


def run_attention(args):
    model, cfg, _, device = _setup(args)
    _, loader = loaders_from_config(cfg, eval_only=True)
    image, _ = next(iter(loader))
    from edgeformer.visualization.attention_map import save_attention_report

    save_attention_report(model, image[0].to(device), args.output)
    print(f"Saved attention report: {args.output}")


def main(argv=None):
    args = parser().parse_args(argv)
    commands = dict(
        train=run_train,
        evaluate=run_evaluate,
        quantize=run_quantize,
        benchmark=run_benchmark,
        sweep=run_sweep,
        compare=run_sweep,
        sensitivity=run_sensitivity,
        attention=run_attention,
    )
    try:
        commands[args.command](args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"edgeformer: {exc}", file=sys.stderr)
        return 2
    return 0
