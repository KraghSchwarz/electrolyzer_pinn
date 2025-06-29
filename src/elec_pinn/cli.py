# src/elec_pinn/cli.py

import argparse
import yaml
import logging
import numpy as np
import torch
from pathlib import Path

# your package modules
from elec_pinn.data.preprocessing import Preprocessor
from elec_pinn.models import FullPINN, GPINN, FPINN, ANN
#from elec_pinn.training.trainer import Trainer
from elec_pinn.utils.logging import setup_logging

def load_config(path: str) -> dict:
    """Read and parse the YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]):
    """
    Takes a list of strings like ["training.epochs=200","model.type=GPINN"]
    and updates the nested cfg dict in place.
    """
    for ov in overrides:
        keypath, val = ov.split('=', 1)
        parts = keypath.split('.')
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        # simple literal parsing
        if val.lower() in ('true','false'):
            parsed = val.lower() == 'true'
        else:
            try:    parsed = int(val)
            except:
                try:    parsed = float(val)
                except: parsed = val
        d[parts[-1]] = parsed

def get_model(cfg: dict):
    """Factory to choose the right PINN class."""
    cls_map = {
        'FullPINN':    FullPINN,
        'GPINN':       GPINN,
        'PhysicsPINN': FPINN,
        'NoPINN':      ANN
    }
    ModelCls = cls_map[cfg['model']['type']]
    mcfg = cfg['model']
    tcfg = cfg['training']
    input_dim = len( cfg['data']['feature_names'] )

    return ModelCls(
        input_dim,
        mcfg['f_hidden_dim'],
        mcfg['g_hidden_dim'],
        mcfg['f_layers'],
        mcfg['g_layers'],
        tcfg['lr'],
        mcfg['pde_weight']
    )

def main():
    parser = argparse.ArgumentParser(
        prog="elec-pinn",
        description="Train & evaluate an electrolyzer PINN"
    )
    parser.add_argument(
        '-c', '--config', required=True,
        help="Path to your config.yaml"
    )
    parser.add_argument(
        '--override', '-o', nargs='*', default=[],
        help="Override config values, e.g. --override training.epochs=200"
    )
    args = parser.parse_args()

    # 1) load + override config
    cfg = load_config(args.config)
    if args.override:
        apply_overrides(cfg, args.override)

    # 2) set seeds
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    # 3) data preprocessing
    dp = Preprocessor(cfg['data']['dataset_name'])
    df = dp.preprocess(
        t0=cfg['data']['t0'],
        t1=cfg['data']['t1'],
        plot_fit=True,
        plot_raw=False
    )

    # 4) scaling & loaders
    from elec_pinn.data.scaler_loader import ScalerLoader
    scaler = ScalerLoader(
        feature_cols=cfg['data']['feature_names'],
        target_cols=cfg['data']['target_names'],
        scale_range=tuple(cfg['data']['scale_range'])
    ).fit(df)
    train_loader, val_loader, test_loader, all_loader = scaler.get_loaders(
        df,
        f_train=cfg['data']['train_frac'],
        f_val=cfg['data']['val_frac'],
        f_test=1-cfg['data']['train_frac']-cfg['data']['val_frac'],
        batch_sizes=tuple(cfg['training']['batch_sizes'])
    )

    # 5) model & trainer
    model = get_model(cfg)


if __name__ == "__main__":
    main()
