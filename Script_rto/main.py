import os
import argparse
import yaml
from trainer import ChestXrayHandler
import torch.multiprocessing as mp
# =====================================================
# BOOLEAN PARSER
# =====================================================
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()

    true_set = {"yes", "y", "true", "t", "1"}
    false_set = {"no", "n", "false", "f", "0"}

    if v in true_set:
        return True
    elif v in false_set:
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid boolean value '{v}'. Expected one of {true_set | false_set}"
        )
        
# =====================================================
# PARSER
# =====================================================

# def parse_args():

#     # Parse only --config first
#     config_parser = argparse.ArgumentParser(add_help=False)
#     config_parser.add_argument("--config", default="../configs/config_HCV.yaml")
#     config_args, remaining = config_parser.parse_known_args()

#     # Load YAML
#     with open(config_args.config, "r") as f:
#         raw_cfg = yaml.safe_load(f)

#     # Convert nested YAML to flat {key: value}
#     config = {}
#     for key, entry in raw_cfg.items():
#         if isinstance(entry, dict) and "value" in entry:
#             config[key] = entry["value"]
#         else:
#             config[key] = entry

#     # Build main parser
#     parser = argparse.ArgumentParser(description="Train DenseNetViT")

#     # Add CLI override args
#     for key, value in config.items():
#         if isinstance(value, bool):
#             parser.add_argument(f"--{key}", type=str, default=None)
#         else:
#             parser.add_argument(f"--{key}", type=type(value), default=None)

#     args = parser.parse_args(remaining)

#     # Merge CLI + config
#     final_cfg = {}
#     for key, value in config.items():
#         cli_value = getattr(args, key)
#         if cli_value is None:
#             final_cfg[key] = value
#         else:
#             if isinstance(value, bool):  # bool handling
#                 final_cfg[key] = cli_value.lower() == "true"
#             else:
#                 final_cfg[key] = cli_value

#     # --- save_dir must be a STRING ---
#     if not isinstance(final_cfg["save_dir"], str):
#         raise ValueError(f"save_dir must be a string, got {final_cfg['save_dir']}")

#     os.makedirs(final_cfg["save_dir"], exist_ok=True)

#     # Auto resume
#     last_ckpt = os.path.join(final_cfg["save_dir"], "last_checkpoint.pth")
#     if final_cfg.get("resume") is None and os.path.exists(last_ckpt):
#         final_cfg["resume"] = last_ckpt

#     # Convert dict --> object
#     class Obj: pass
#     cfg_obj = Obj()
#     for k, v in final_cfg.items():
#         setattr(cfg_obj, k, v)

#     return cfg_obj



# =====================================================
# PARSER
# =====================================================


def to_abs(path):
    """Convert a possibly-relative path to absolute."""
    if path is None:
        return None
    return os.path.abspath(os.path.join(os.getcwd(), path))


def parse_args():

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "..", "configs", "config_HCV.yaml")

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=DEFAULT_CONFIG)
    config_args, remaining = config_parser.parse_known_args()

    config_path = os.path.abspath(config_args.config)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    config = {}
    for key, entry in raw_cfg.items():
        if isinstance(entry, dict) and "value" in entry:
            config[key] = entry["value"]
        else:
            config[key] = entry

    parser = argparse.ArgumentParser(description="Train DenseNetViT")

    for key, value in config.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", type=str, default=None)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=None)

    args = parser.parse_args(remaining)

    final_cfg = {}
    for key, value in config.items():
        cli_value = getattr(args, key)
        if cli_value is None:
            final_cfg[key] = value
        else:
            if isinstance(value, bool):
                final_cfg[key] = cli_value.lower() == "true"
            else:
                final_cfg[key] = cli_value

    # ---- Path Resolver ----
    def fix_path(p):
        if p is None:
            return None
        if isinstance(p, str) and not os.path.isabs(p):
            base = os.path.dirname(os.path.abspath(config_args.config))
            return os.path.abspath(os.path.join(base, p))
        return p

    final_cfg["img_dir"]   = fix_path(final_cfg["img_dir"])
    final_cfg["csv_path"]  = fix_path(final_cfg["csv_path"])
    final_cfg["save_dir"]  = fix_path(final_cfg["save_dir"])

    # if the argument is to test the model, you will need to load it
    if final_cfg.get("resume") is not None or final_cfg.get("test_model"):
        final_cfg["resume"] = fix_path(final_cfg["resume"])

    os.makedirs(final_cfg["save_dir"], exist_ok=True)

    class Obj: pass
    cfg_obj = Obj()
    for k, v in final_cfg.items():
        setattr(cfg_obj, k, v)

    return cfg_obj

# =====================================================
# MAIN
# =====================================================
def main():
    args = parse_args()

    # Create trainer first (this determines rank)
    trainer = ChestXrayHandler(args)

    # Only master rank prints arguments
    if trainer.is_master:
        print("\n===== Parsed Arguments =====")
        for k, v in vars(args).items():
            print(f"{k}: {v}")
        print("===========================\n")

    print(args.test_model)

    if args.test_model:
        trainer.test()
    else:
        trainer.run()


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    # REQUIRED FOR torchrun OR multiprocessing
    mp.set_start_method("spawn", force=True)

    main()