# src/elec_pinn/utils/logging.py

import logging
import logging.config
import sys
from pathlib import Path

def setup_logging(cfg: dict):
    if not isinstance(cfg, dict):
        raise ValueError("Logging configuration must be a dict.")

    # 1) Full dictConfig?
    if "version" in cfg:
        logging.config.dictConfig(cfg)
        return

    # 2) Simple config – map your keys
    level_name = cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt   = cfg.get("fmt", cfg.get("format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    datefmt = cfg.get("datefmt", None)

    handlers = []

    # Console handler (always on unless you add console: false)
    if cfg.get("console", True):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        handlers.append(ch)

    # File handler, if requested
    if cfg.get("to_file", False):
        filename = cfg.get("file", cfg.get("filename"))
        if filename:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(filename, mode=cfg.get("file_mode", "a"))
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            handlers.append(fh)

    # Apply to root logger
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)

    # Quiet noisy libraries, if you like
    for module, lvl in cfg.get("quiet_modules", {}).items():
        logging.getLogger(module).setLevel(getattr(logging, lvl.upper(), logging.WARNING))
