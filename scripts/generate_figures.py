from pathlib import Path
from html import escape

DEST = Path(__file__).resolve().parents[1] / "assets"
BLUE, TEAL, AMBER, INK, MUTED = "#2563eb", "#0f766e", "#b45309", "#17243b", "#53647b"


class SVG:
    def __init__(self, w, h, title, subtitle):
        self.w, self.h = w, h
        self.a = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{escape(title)}">',
            "<defs>"
            + "".join(
                f'<marker id="arrow-{k}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L9 4.5 L0 9 Z" fill="{c}"/></marker>'
                for k, c in [("blue", BLUE), ("teal", TEAL), ("gray", MUTED), ("amber", AMBER)]
            )
            + "</defs>",
            "<style>text{font-family:DejaVu Sans,Arial,sans-serif} .mono{font-family:DejaVu Sans Mono,monospace}</style>",
            f'<rect width="{w}" height="{h}" fill="#f7f9fd"/>',
        ]
        self.text(48, 51, "EDGEFORMER STUDIO", 13, TEAL, weight=700, spacing=2)
        self.text(48, 94, title, 31, INK, weight=700)
        self.text(48, 127, subtitle, 16, MUTED)

    def text(self, x, y, t, size=16, fill=INK, anchor=None, weight=400, spacing=None):
        self.a.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}"'
            + (f' text-anchor="{anchor}"' if anchor else "")
            + (f' letter-spacing="{spacing}"' if spacing else "")
            + f">{escape(t)}</text>"
        )

    def rect(self, x, y, w, h, fill="white", stroke="#dbe3ef", radius=12, dash=None):
        self.a.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"'
            + (f' stroke-dasharray="{dash}"' if dash else "")
            + "/>"
        )

    def panel(self, x, y, w, h, title, tag):
        self.rect(x, y, w, h)
        self.text(x + 22, y + 31, title, 17, INK, weight=700)
        self.text(x + w - 22, y + 31, tag, 12, MUTED, anchor="end")

    def box(self, x, y, w, h, title, lines=(), kind="blue", size=17):
        c = {"blue": BLUE, "teal": TEAL, "amber": AMBER, "gray": MUTED}[kind]
        f = {"blue": "#edf3ff", "teal": "#ecf8f5", "amber": "#fff7e9", "gray": "#f3f6fa"}[kind]
        self.rect(x, y, w, h, f, c, 8)
        self.text(x + w / 2, y + 29, title, size, c, anchor="middle", weight=700)
        for i, line in enumerate(lines):
            self.text(x + w / 2, y + 55 + i * 22, line, 13, INK, anchor="middle")

    def arrow(self, path, kind="blue", dashed=False):
        c = {"blue": BLUE, "teal": TEAL, "amber": AMBER, "gray": MUTED}[kind]
        self.a.append(
            f'<path d="{path}" fill="none" stroke="{c}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round" marker-end="url(#arrow-{kind})"'
            + (' stroke-dasharray="6 5"' if dashed else "")
            + "/>"
        )

    def dot(self, x, y, c=BLUE):
        self.a.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{c}"/>')

    def save(self, name):
        self.a.append("</svg>")
        (DEST / name).write_text("\n".join(self.a))


s = SVG(
    1440,
    1040,
    "From a floating model to a reproducible design study",
    "Executable workflows for compact Vision Transformers · version 0.2",
)
s.panel(
    38,
    159,
    1364,
    225,
    "01  TRAIN AND VALIDATE",
    "Checkpoints preserve configuration, optimizer, schedule and RNG",
)
s.box(
    65,
    225,
    225,
    110,
    "Data loaders",
    ["CIFAR / Tiny ImageNet", "Seeded offline smoke data"],
    "teal",
)
s.box(351, 225, 245, 110, "TinyViT", ["FP32 baseline or QAT", "Patch → attention → MLP"])
s.box(657, 225, 300, 110, "Training engine", ["AdamW · cosine schedule", "Validation every epoch"])
s.box(
    1018,
    225,
    350,
    110,
    "Best / latest checkpoints",
    ["Atomic state-dict saves", "Safe loading · epoch resume"],
    "teal",
)
for a, b in [(290, 351), (596, 657), (957, 1018)]:
    s.arrow(f"M{a} 280 H{b}")
s.panel(
    38,
    413,
    1364,
    294,
    "02  COMPARE MATCHED DESIGN POINTS",
    "Same reference weights, evaluation batches and input shapes",
)
s.box(
    65,
    491,
    240,
    130,
    "Reference model",
    ["Floating checkpoint", "or seeded untrained model", "No hidden random fallback"],
)
s.box(
    381,
    491,
    280,
    130,
    "Precision / error variants",
    ["W32 / W8 / W4 · mixed plan", "Per-channel weight PTQ", "Optional statistical noise"],
    "amber",
)
s.box(
    737,
    491,
    285,
    130,
    "Matched evaluation",
    ["Top-1 · loss · agreement", "Logit RMSE · attention JS", "Layer sensitivity"],
    "teal",
)
s.box(
    1098,
    491,
    270,
    130,
    "Design table",
    ["CSV + JSON + Pareto plot", "Accuracy / packed storage", "Runtime measured separately"],
    "teal",
)
for a, b in [(305, 381), (661, 737), (1022, 1098)]:
    s.arrow(f"M{a} 551 H{b}", "teal")
s.arrow("M1193 335 V398 H51 V464 H185 V491", "gray", True)
s.text(
    65,
    665,
    "PTQ: weights only  |  QAT: Linear/Conv input + weight Q/DQ with straight-through gradients",
    14,
    MUTED,
)
s.panel(
    38,
    735,
    1364,
    224,
    "03  INTERPRET EACH KIND OF EVIDENCE",
    "Measured execution, numerical simulation and analytical cost stay distinguishable",
)
s.box(
    65,
    805,
    393,
    112,
    "Measured",
    ["Forward latency · median / p95", "Device, thread count, batch size"],
    "blue",
)
s.box(
    490,
    805,
    408,
    112,
    "Simulated numerics",
    ["Quantization grid · output noise", "Accuracy changes on selected data"],
    "amber",
)
s.box(
    930,
    805,
    438,
    112,
    "Analytical hardware cost",
    ["MACs · ideal packed bytes + scales", "Uncalibrated compute-energy proxy"],
    "teal",
)
s.text(
    48,
    1000,
    "No integer runtime kernels, circuit PPA or trained-dataset accuracy is implied by a precision label.",
    14,
    MUTED,
)
s.save("architecture.svg")

s = SVG(
    1540,
    1080,
    "The attention block, as a dataflow schematic",
    "Logical software datapath · solid arrows carry tensors · dashed arrows carry precision controls",
)
s.panel(
    35, 157, 1470, 196, "A  TOKEN FRONT END", "Example: image 32×32, patch 4×4, embedding width D"
)
s.box(62, 225, 225, 93, "Input image", ["B × C × H × W"], "gray")
s.box(352, 225, 265, 93, "Patch projection", ["Conv2d → flatten → transpose"])
s.box(682, 225, 345, 93, "Token assembly", ["Prepend CLS · add position embedding"])
s.box(1092, 225, 380, 93, "Token stream X", ["B × N × D ; N = (H / P)² + 1"], "teal")
for a, b in [(287, 352), (617, 682), (1027, 1092)]:
    s.arrow(f"M{a} 272 H{b}")
s.panel(
    35,
    382,
    1470,
    402,
    "B  PRE-NORM TRANSFORMER BLOCK",
    "Repeat × depth · full-precision LayerNorm, softmax and residual adds",
)
s.a.append('<g transform="translate(0 23)">')
# common data bus starts left; fork for residual
s.text(62, 448, "X", 17, BLUE, weight=700)
s.arrow("M88 443 H154")
s.dot(116, 443)
s.box(154, 415, 170, 78, "LayerNorm", ["B × N × D"], "gray")
s.box(382, 415, 225, 78, "QKV projection", ["3 × (B × N × D)"])
s.box(665, 415, 245, 78, "Split into heads", ["Q, K, V: B × h × N × d"])
s.box(968, 415, 280, 78, "Attention core", ["softmax(QKᵀ / √d) V"], "teal")
s.box(1306, 415, 170, 78, "Output proj.", ["B × N × D"])
for a, b in [(324, 382), (607, 665), (910, 968), (1248, 1306)]:
    s.arrow(f"M{a} 454 H{b}")
s.arrow("M1391 493 V540 H1310")
s.a.append(f'<circle cx="1286" cy="540" r="23" fill="white" stroke="{BLUE}" stroke-width="2"/>')
s.text(1286, 547, "+", 24, BLUE, anchor="middle")
s.arrow("M116 443 V516 H1286 V517")
s.text(521, 511, "residual X", 12, MUTED)
s.arrow("M1263 540 H1108 V591")
s.box(1005, 591, 206, 82, "LayerNorm", ["B × N × D"], "gray")
s.box(700, 591, 245, 82, "Feed-forward MLP", ["D → hidden → D · GELU"])
s.arrow("M1005 632 H945")
s.arrow("M700 632 H562")
s.a.append(f'<circle cx="538" cy="632" r="23" fill="white" stroke="{BLUE}" stroke-width="2"/>')
s.text(538, 639, "+", 24, BLUE, anchor="middle")
s.dot(1193, 540)
s.arrow("M1193 540 V565 H538 V609")
s.text(775, 560, "residual after attention", 12, MUTED)
s.arrow("M515 632 H225")
s.text(76, 639, "Block output", 16, TEAL, weight=700)
s.text(
    62,
    727,
    "SDPA path: attention tensors are internal to PyTorch. Inspection path: explicit QKᵀ, softmax and AV.",
    14,
    MUTED,
)
s.a.append("</g>")
s.panel(
    35,
    816,
    1470,
    204,
    "C  NUMERICAL CONTROLS AND MODEL READOUT",
    "Quantization and profiling do not change this logical graph",
)
s.box(
    62,
    886,
    450,
    96,
    "Precision controls",
    ["PTQ: Conv / Linear weights only", "QAT: Conv / Linear inputs + weights"],
    "amber",
)
s.box(
    570, 886, 385, 96, "Final normalization + CLS", ["Normalize sequence → select token 0"], "gray"
)
s.box(
    1013,
    886,
    459,
    96,
    "Classification head",
    ["Linear D → number of classes", "Floating logits"],
    "teal",
)
s.arrow("M955 934 H1013")
s.arrow("M287 886 V873 H1391 V784", "amber", True)
s.text(
    524,
    867,
    "Precision applies at Conv / Linear boundaries; nonlinear operations remain floating point.",
    13,
    AMBER,
)
s.text(
    48,
    1057,
    "This is a logical model diagram, not a claim of an implemented RTL accelerator or a particular physical memory hierarchy.",
    14,
    MUTED,
)
s.save("attention_datapath.svg")

s = SVG(
    1440,
    860,
    "Quantization with a working training gradient",
    "Two execution paths · both retain floating tensors and floating operator kernels",
)
s.panel(
    38,
    165,
    1364,
    270,
    "01  WEIGHT-ONLY POST-TRAINING SIMULATION",
    "The original floating model remains unchanged",
)
s.box(68, 239, 235, 103, "Floating weight W", ["Conv / Linear tensor"])
s.box(
    368,
    239,
    275,
    103,
    "Per-channel scale",
    ["s = max |W| / (2ᵇ⁻¹ − 1)", "One scale per output channel"],
    "amber",
)
s.box(
    708,
    239,
    310,
    103,
    "Round and clamp",
    ["q = clamp(round(W / s))", "Symmetric narrow integer range"],
    "amber",
)
s.box(
    1083, 239, 280, 103, "Floating weight Ŵ", ["Ŵ = q × s", "Save Q/DQ values + metadata"], "teal"
)
for a, b in [(303, 368), (643, 708), (1018, 1083)]:
    s.arrow(f"M{a} 289 H{b}", "amber")
s.text(
    68,
    393,
    "Bias, LayerNorm, positional embeddings and all activations remain floating point in this PTQ path.",
    15,
    MUTED,
)
s.panel(
    38,
    468,
    1364,
    310,
    "02  QUANTIZATION-AWARE TRAINING",
    "Dynamic per-tensor input scale + per-output-channel weight scale",
)
s.box(
    68, 544, 235, 107, "Master parameters", ["Floating weights and gradients", "Updated by AdamW"]
)
s.box(
    368,
    544,
    275,
    107,
    "Fake quantize",
    ["Input Q/DQ + weight Q/DQ", "No fixed observer calibration"],
    "amber",
)
s.box(
    708,
    544,
    310,
    107,
    "Floating Conv / Linear",
    ["Use fake-quantized operands", "Bias remains floating point"],
)
s.box(
    1083, 544, 280, 107, "Task loss", ["Classification cross-entropy", "Then backpropagate"], "teal"
)
for a, b in [(303, 368), (643, 708), (1018, 1083)]:
    s.arrow(f"M{a} 595 H{b}")
s.arrow("M1223 651 V704 H185 V651", "teal", True)
s.text(
    700,
    697,
    "Backward pass: straight-through estimator replaces the zero derivative of round",
    14,
    TEAL,
    anchor="middle",
)
s.text(
    68,
    748,
    "QAT checkpoints reconstruct their wrappers on load. Attention products, softmax and normalization remain floating.",
    14,
    MUTED,
)
s.text(
    48,
    822,
    "Supported symmetric precisions: 2–16 bits; 32 selects identity. W4 and W8 are simulations, not integer deployments.",
    14,
    MUTED,
)
s.save("quantization.svg")
print("Wrote three vector diagrams")
