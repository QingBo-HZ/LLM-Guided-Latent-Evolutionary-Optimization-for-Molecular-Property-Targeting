#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import argparse
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from pl_models import PSVAEModel
from data import bpe_dataset
from data.mol_bpe import Tokenizer
from utils.logger import print_log
from utils.nn_utils import VAEEarlyStopping
from utils.nn_utils import common_config, predictor_config, encoder_config
from utils.nn_utils import ps_vae_config


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    pl.utilities.seed.seed_everything(seed=seed)


def train(model, train_loader, valid_loader, test_loader, args):
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,
        save_last=True,
        save_top_k=args.save_top_k,
        mode="min",
    )
    print_log("Using vae kl warmup early stopping strategy")
    anneal_step = args.kl_warmup + (args.max_beta // args.step_beta - 1) * args.kl_anneal_iter
    early_stop_callback = VAEEarlyStopping(
        finish_anneal_step=anneal_step,
        monitor=args.monitor,
        patience=args.patience,
        mode="min",
    )
    callbacks = [checkpoint_callback, early_stop_callback]
    if args.log_lr:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    trainer_config = {
        "gpus": args.gpus,
        "max_epochs": args.epochs,
        "default_root_dir": args.save_dir,
        "callbacks": callbacks,
        "gradient_clip_val": args.grad_clip,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "log_every_n_steps": args.log_every_n_steps,
    }
    if args.gpus is not None and len(str(args.gpus).split(",")) > 1:
        trainer_config["accelerator"] = "dp"

    trainer = pl.Trainer(**trainer_config)
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, dataloaders=test_loader)


def parse():
    parser = argparse.ArgumentParser(description="resume training PS-VAE for molecule generation")
    parser.add_argument("--resume_from_checkpoint", type=str, required=True)
    parser.add_argument("--train_set", type=str, required=True)
    parser.add_argument("--valid_set", type=str, required=True)
    parser.add_argument("--test_set", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--step_beta", type=float, default=0.001)
    parser.add_argument("--max_beta", type=float, default=0.03)
    parser.add_argument("--kl_warmup", type=int, default=2000)
    parser.add_argument("--kl_anneal_iter", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--monitor", type=str, default="val_loss")
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--log_lr", action="store_true")
    parser.add_argument("--seed", type=int, default=2021)

    parser.add_argument("--props", type=str, nargs="+", choices=["qed", "sa", "logp", "gsk3b", "jnk3"], default=["logp"])
    parser.add_argument("--predictor_hidden_dim", type=int, default=200)
    parser.add_argument("--node_hidden_dim", type=int, default=384)
    parser.add_argument("--graph_embedding_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--max_pos", type=int, default=50)
    parser.add_argument("--atom_embedding_dim", type=int, default=50)
    parser.add_argument("--piece_embedding_dim", type=int, default=100)
    parser.add_argument("--pos_embedding_dim", type=int, default=50)
    parser.add_argument("--piece_hidden_dim", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    setup_seed(args.seed)
    print_log(args)
    print_log("loading data ...")
    tokenizer = Tokenizer(args.vocab)
    vocab = tokenizer.chem_vocab
    train_loader = bpe_dataset.get_dataloader(
        args.train_set, tokenizer, batch_size=args.batch_size,
        shuffle=args.shuffle, num_workers=args.num_workers,
    )
    valid_loader = bpe_dataset.get_dataloader(
        args.valid_set, tokenizer, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )
    test_loader = bpe_dataset.get_dataloader(
        args.test_set, tokenizer, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    print("creating model ...")
    config = {**common_config(args), **encoder_config(args, vocab), **predictor_config(args)}
    config.update(ps_vae_config(args, tokenizer))
    model = PSVAEModel(config, tokenizer)
    print_log(f"config: {config}")
    print(model)
    print_log(f"resume_from_checkpoint: {args.resume_from_checkpoint}")
    train(model, train_loader, valid_loader, test_loader, args)
