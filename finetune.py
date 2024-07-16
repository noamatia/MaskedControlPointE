import os
import torch
import wandb
import argparse
import pandas as pd
from datetime import datetime
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from masked_control_point_e import MaskedControlPointE
from masked_control_shapenet import (
    SOURCE_UID,
    TARGET_UID,
    masked_labels_path,
    MaskedControlShapeNet,
)

torch.set_float32_matmul_precision("high")

DATASETS_DIR = "data/datasets"
OUTPUTS_DIR = "/scratch/noam/masked_control_point_e/outputs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--data_csv_val", type=str)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=7e-5*0.4)
    parser.add_argument("--subset_size", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1024)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--grad_acc_steps", type=int, default=11)
    parser.add_argument("--validation_freq", type=int, default=100)
    parser.add_argument("--cond_drop_prob", type=float, default=0.5)
    parser.add_argument("--prompt_key", type=str, default="utterance")
    parser.add_argument("--num_validation_samples", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default="masked_control_point_e")
    parser.add_argument("--data_csv_train", type=str, default="chair_armrests/train.csv")
    args = parser.parse_args()
    return args


def build_name(args):
    date_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    beta = f"beta_{args.beta}" if args.beta is not None else "no_beta"
    dataset_name = args.data_csv_train.replace("/", "_").replace(".csv", "")
    subset_size = "full" if args.subset_size is None else f"subset_{args.subset_size}"
    return (
        f"{date_str}_{dataset_name}_{subset_size}_{args.part}_{args.prompt_key}_{beta}"
    )


def load_df(data_csv, subset_size):
    df = pd.read_csv(os.path.join(DATASETS_DIR, data_csv))
    for uid_key in [SOURCE_UID, TARGET_UID]:
        df = df[
            df.apply(
                lambda row: os.path.exists(masked_labels_path(row[uid_key])),
                axis=1,
            )
        ]
    if subset_size is not None:
        df = df.head(subset_size)
    return df


def main(args):
    masked = args.beta is not None
    name = build_name(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = load_df(args.data_csv_train, args.subset_size)
    train_dataset = MaskedControlShapeNet(
        df=train_df,
        masked=masked,
        device=device,
        part=args.part,
        num_points=args.num_points,
        batch_size=args.batch_size,
        prompt_key=args.prompt_key,
    )
    train_data_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    validation_df = (
        load_df(args.data_csv_val, args.num_validation_samples)
        if args.data_csv_val
        else train_df
    ).head(args.num_validation_samples)
    validation_dataset = MaskedControlShapeNet(
        masked=masked,
        device=device,
        part=args.part,
        df=validation_df,
        num_points=args.num_points,
        prompt_key=args.prompt_key,
        batch_size=args.num_validation_samples,
    )
    validation_data_loader = DataLoader(
        dataset=validation_dataset, batch_size=args.num_validation_samples
    )
    wandb.init(project=args.wandb_project, name=name, config=vars(args))
    model = MaskedControlPointE(
        lr=args.lr,
        dev=device,
        masked=masked,
        beta=args.beta,
        timesteps=args.timesteps,
        num_points=args.num_points,
        batch_size=args.batch_size,
        cond_drop_prob=args.cond_drop_prob,
        validation_data_loader=validation_data_loader,
    )
    wandb.watch(model)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        save_weights_only=True,
        every_n_epochs=args.epochs / 10,
        dirpath=os.path.join(OUTPUTS_DIR, name),
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=args.grad_acc_steps,
        check_val_every_n_epoch=args.validation_freq,
    )
    trainer.fit(model, train_data_loader, validation_data_loader)
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
