import os
import sys
from datetime import datetime
from typing import Dict, Tuple

import torch
import yaml
from accelerate import Accelerator
from easydict import EasyDict
from objprint import objstr
from torch.utils.data import DataLoader

from src.fttransformer import FTTransformer
from src.tabular_data import TabularMetadata, build_datasets, merge_tables
from src.utils import Logger, same_seeds


def prepare_batch(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], accelerator: Accelerator):
    x_cont, x_cat, targets = batch
    x_cont_tensor = x_cont.to(accelerator.device) if x_cont.shape[1] > 0 else None
    x_cat_tensor = x_cat.to(accelerator.device) if x_cat.shape[1] > 0 else None
    targets_tensor = targets.to(accelerator.device)
    return x_cont_tensor, x_cat_tensor, targets_tensor


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    accelerator: Accelerator,
    epoch: int,
    step: int,
    num_epochs: int,
) -> Tuple[int, float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        x_cont, x_cat, targets = prepare_batch(batch, accelerator)
        with accelerator.accumulate(model):
            logits = model(x_cont, x_cat)
            loss = loss_fn(logits, targets)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        step += 1

        gathered_loss = accelerator.gather_for_metrics(loss.detach() * targets.size(0))
        gathered_targets = accelerator.gather_for_metrics(targets)
        gathered_preds = accelerator.gather_for_metrics(logits.argmax(dim=-1))

        batch_size = gathered_targets.shape[0]
        loss_value = gathered_loss.sum().item() / max(batch_size, 1)
        correct = (gathered_preds == gathered_targets).sum().item()
        accuracy_value = correct / batch_size if batch_size else 0.0

        total_loss += gathered_loss.sum().item()
        total_correct += correct
        total_samples += batch_size

        accelerator.log({
            "Train/Total Loss": loss_value,
            "Train/Accuracy": accuracy_value,
        }, step=step)
        accelerator.print(
            f"Epoch [{epoch + 1}/{num_epochs}] Training [{batch_idx + 1}/{len(train_loader)}] "
            f"Loss: {loss_value:.4f} Acc: {accuracy_value:.4%}",
            flush=True,
        )

    if total_samples == 0:
        return step, 0.0, 0.0

    mean_loss = total_loss / total_samples
    mean_accuracy = total_correct / total_samples
    accelerator.log({
        "Train/mean loss": mean_loss,
        "Train/mean accuracy": mean_accuracy,
    }, step=epoch)
    accelerator.print(
        f"Epoch [{epoch + 1}/{num_epochs}] Training summary Loss: {mean_loss:.4f} Acc: {mean_accuracy:.4%}",
        flush=True,
    )
    return step, mean_loss, mean_accuracy


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    loss_fn: torch.nn.Module,
    accelerator: Accelerator,
    epoch: int,
    step: int,
    num_epochs: int,
) -> Tuple[float, Dict[str, float], int]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(val_loader):
        x_cont, x_cat, targets = prepare_batch(batch, accelerator)
        logits = model(x_cont, x_cat)
        loss = loss_fn(logits, targets)

        step += 1

        gathered_loss = accelerator.gather_for_metrics(loss.detach() * targets.size(0))
        gathered_targets = accelerator.gather_for_metrics(targets)
        gathered_preds = accelerator.gather_for_metrics(logits.argmax(dim=-1))

        batch_size = gathered_targets.shape[0]
        loss_value = gathered_loss.sum().item() / max(batch_size, 1)
        correct = (gathered_preds == gathered_targets).sum().item()
        accuracy_value = correct / batch_size if batch_size else 0.0

        total_loss += gathered_loss.sum().item()
        total_correct += correct
        total_samples += batch_size

        accelerator.log({
            "Val/Total Loss": loss_value,
            "Val/Accuracy": accuracy_value,
        }, step=step)
        accelerator.print(
            f"Epoch [{epoch + 1}/{num_epochs}] Validation [{batch_idx + 1}/{len(val_loader)}] "
            f"Loss: {loss_value:.4f} Acc: {accuracy_value:.4%}",
            flush=True,
        )

    if total_samples == 0:
        metrics = {"Val/mean loss": 0.0, "Val/mean accuracy": 0.0}
        accelerator.log(metrics, step=epoch)
        return 0.0, metrics, step

    mean_loss = total_loss / total_samples
    mean_accuracy = total_correct / total_samples
    metrics = {"Val/mean loss": mean_loss, "Val/mean accuracy": mean_accuracy}
    accelerator.log(metrics, step=epoch)
    accelerator.print(
        f"Epoch [{epoch + 1}/{num_epochs}] Validation summary Loss: {mean_loss:.4f} Acc: {mean_accuracy:.4%}",
        flush=True,
    )
    return mean_accuracy, metrics, step


def create_model(metadata: TabularMetadata, model_config: EasyDict) -> FTTransformer:
    backbone_kwargs = FTTransformer.get_default_kwargs(model_config.n_blocks)
    overrides = dict(model_config.backbone_kwargs) if model_config.backbone_kwargs else {}
    backbone_kwargs.update(overrides)
    return FTTransformer(
        n_cont_features=len(metadata.continuous_columns),
        cat_cardinalities=list(metadata.categorical_cardinalities),
        **backbone_kwargs,
    )


if __name__ == "__main__":
    with open("config.yml", "r", encoding="utf-8") as config_file:
        config = EasyDict(yaml.safe_load(config_file))

    same_seeds(config.seed)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging_dir = os.path.join(os.getcwd(), "logs", f"{config.trainer.run_name}_{timestamp}")
    accelerator = Accelerator(
        cpu=False,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        log_with=["tensorboard"],
        logging_dir=logging_dir,
        mixed_precision=config.trainer.mixed_precision,
    )
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.splitext(os.path.basename(__file__))[0])
    accelerator.print(objstr(config))

    accelerator.print("Load dataset...")
    raw_table = merge_tables(
        config.data.files,
        label_column=config.data.label_column,
        target_column=config.data.target_column,
    )
    train_dataset, val_dataset, metadata = build_datasets(
        raw_table,
        label_column=config.data.label_column,
        target_column=config.data.target_column,
        val_ratio=config.data.val_ratio,
        seed=config.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    accelerator.print(
        f"Continuous features ({len(metadata.continuous_columns)}): {metadata.continuous_columns}"
    )
    accelerator.print(
        f"Categorical features ({len(metadata.categorical_columns)}): {metadata.categorical_columns}"
    )

    model = create_model(metadata, config.model)
    optimizer = torch.optim.AdamW(
        model.make_parameter_groups(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.trainer.num_epochs
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    checkpoint_root = os.path.join(os.getcwd(), "model_store", config.trainer.output_dir)
    best_dir = os.path.join(checkpoint_root, "best")
    checkpoint_dir = os.path.join(checkpoint_root, "checkpoint")
    if accelerator.is_main_process:
        os.makedirs(best_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    best_acc = 0.0
    best_metrics: Dict[str, float] = {}
    best_epoch = -1
    starting_epoch = 0
    train_step = 0
    val_step = 0

    resume_from = config.trainer.resume_from
    if resume_from:
        resume_dir = os.path.join(os.getcwd(), "model_store", resume_from, "checkpoint")
        if os.path.isdir(resume_dir):
            accelerator.print(f"Resuming from {resume_dir}")
            accelerator.load_state(resume_dir)
            metadata_path = os.path.join(resume_dir, "epoch.pth.tar")
            if os.path.exists(metadata_path):
                saved_state = torch.load(metadata_path, map_location="cpu")
                starting_epoch = int(saved_state.get("epoch", -1)) + 1
                best_acc = float(saved_state.get("best_acc", 0.0))
                best_metrics = saved_state.get("best_metrics", {})
                best_epoch = int(saved_state.get("best_epoch", starting_epoch - 1))
                train_step = starting_epoch * len(train_loader)
                val_step = train_step

    accelerator.print("Start training!")
    num_epochs = config.trainer.num_epochs

    for epoch in range(starting_epoch, num_epochs):
        train_step, train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, loss_fn, accelerator, epoch, train_step, num_epochs
        )

        scheduler.step()
        accelerator.log({"Train/LR": optimizer.param_groups[0]["lr"]}, step=epoch)

        val_accuracy, val_metrics, val_step = validate(
            model, val_loader, loss_fn, accelerator, epoch, val_step, num_epochs
        )

        if val_accuracy > best_acc:
            accelerator.print(
                f"New best accuracy: {val_accuracy:.4%} (previous {best_acc:.4%}) at epoch {epoch + 1}"
            )
            accelerator.save_state(best_dir)
            best_acc = val_accuracy
            best_metrics = val_metrics
            best_epoch = epoch

        accelerator.print("Checkpointing...")
        accelerator.save_state(checkpoint_dir)
        if accelerator.is_main_process:
            torch.save(
                {
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "best_metrics": best_metrics,
                    "best_epoch": best_epoch,
                },
                os.path.join(checkpoint_dir, "epoch.pth.tar"),
            )

    accelerator.print(f"Best accuracy: {best_acc:.4%} at epoch {best_epoch + 1 if best_epoch >= 0 else 'N/A'}")
    accelerator.print(f"Best metrics: {best_metrics}")
    accelerator.end_training()
    sys.exit(0)
