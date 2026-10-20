"""Shared YAML defaults for TensorFlow CLI scripts.

CLI arguments override YAML; paths are interpreted relative to repository root
when not absolute. Dataset roots are never made relative to the config folder.
"""
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_run_config(filename):
    config_path = Path(filename).expanduser().resolve()
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected YAML mapping: {config_path}")
    for key in ("category", "dataset_root", "output_dir"):
        if not cfg.get(key):
            raise ValueError(f"Missing {key} in {config_path}")
    if cfg["category"] not in ("screw", "hazelnut"):
        raise ValueError(f"Unsupported MVTec category: {cfg['category']}")
    if int(cfg.get("image_size", 128)) != 128:
        raise ValueError("Paper architecture is fixed to 128x128")
    cfg["_config_file"] = str(config_path)
    return cfg


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def apply_config(args, parser, mode):
    if not args.config:
        return args
    cfg = load_run_config(args.config)
    mapping = {
        "dataset_root": "dataset_root",
        "category": "category",
        "epochs": "epochs",
        "batch_size": "batch_size",
        "latent_dim": "latent_dim",
        "learning_rate": "learning_rate",
        "seed": "seed",
        "output_dir": "output",
    }
    if mode != "train":
        mapping.pop("epochs", None)
        mapping.pop("batch_size", None)
        mapping.pop("learning_rate", None)
        mapping.pop("seed", None)
    for source, target in mapping.items():
        if getattr(args, target, None) is None and source in cfg:
            setattr(args, target, cfg[source])
    if mode in ("evaluate", "explain"):
        checkpoint = cfg.get("checkpoint")
        if args.checkpoint_dir is None and checkpoint:
            # E.g. models/screw_VAE_GAN_30000/screw_VAE_GAN_30000_model
            args.checkpoint_dir = str(resolve_path(checkpoint).parent)
        if args.checkpoint_epochs is None:
            # Historical weights can be 30000 epochs even when YAML epochs: 10
            import re
            m = re.search(r"_VAE_GAN_(\\d+)", str(checkpoint or ""))
            args.checkpoint_epochs = int(m.group(1)) if m else 30000
        if mode == "explain":
            settings = cfg.get("explanation", {})
            if args.method is None:
                args.method = settings.get("method", "residual")
            if args.num_samples is None:
                args.num_samples = int(settings.get("num_samples", 100))
            if args.image is None:
                kind = settings.get("anomaly_type", "good")
                index = int(settings.get("image_index", 0))
                folder = resolve_path(args.dataset_root) / args.category / "test" / kind
                paths = sorted(folder.glob("*.png"))
                if not paths or not 0 <= index < len(paths):
                    parser.error(f"No image at index {index} in {folder}")
                args.image = str(paths[index])
    return args


def require_args(args, parser, names):
    missing = [name for name in names if getattr(args, name, None) is None]
    if missing:
        parser.error("Missing " + ", ".join("--" + n.replace("_", "-") for n in missing)
                     + "; provide --config or explicit command-line options")
