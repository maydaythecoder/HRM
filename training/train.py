"""
Training script for the Hierarchical Reasoning Model.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from omegaconf import OmegaConf

from data.dataset import HRMDataset, collate_fn
from models.neural.hierarchical_neural import HierarchicalNeuralReasoner, build_neural_reasoner


class HRMLoss(nn.Module):
    """
    Multi-task loss function for hierarchical reasoning.

    Combines losses for fact extraction, relation prediction, and conclusion generation.
    """

    def __init__(
        self,
        fact_weight: float = 1.0,
        relation_weight: float = 1.0,
        conclusion_weight: float = 1.0,
    ) -> None:
        """
        Initialize the loss function.

        Args:
            fact_weight: Weight for fact extraction loss.
            relation_weight: Weight for relation prediction loss.
            conclusion_weight: Weight for conclusion generation loss.
        """
        super().__init__()
        self.fact_weight = fact_weight
        self.relation_weight = relation_weight
        self.conclusion_weight = conclusion_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        fact_embeddings: torch.Tensor,
        relation_logits: torch.Tensor,
        relation_weights: torch.Tensor,
        conclusion_theme_logits: torch.Tensor,
        conclusion_summary_embeddings: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            fact_embeddings: Predicted fact embeddings.
            relation_logits: Predicted relation predicate logits.
            relation_weights: Predicted relation confidence scores.
            conclusion_theme_logits: Predicted conclusion theme logits.
            conclusion_summary_embeddings: Predicted conclusion summary embeddings.
            targets: Optional ground truth targets.

        Returns:
            Dictionary of loss components and total loss.
        """
        losses = {}

        if targets is None:
            fact_loss = torch.tensor(0.0, device=fact_embeddings.device)
            relation_loss = torch.tensor(0.0, device=relation_logits.device)
            conclusion_loss = torch.tensor(0.0, device=conclusion_theme_logits.device)
        else:
            fact_loss = self._compute_fact_loss(fact_embeddings, targets)
            relation_loss = self._compute_relation_loss(
                relation_logits,
                relation_weights,
                targets,
            )
            conclusion_loss = self._compute_conclusion_loss(
                conclusion_theme_logits,
                conclusion_summary_embeddings,
                targets,
            )

        total_loss = (
            self.fact_weight * fact_loss
            + self.relation_weight * relation_loss
            + self.conclusion_weight * conclusion_loss
        )

        losses["fact_loss"] = fact_loss
        losses["relation_loss"] = relation_loss
        losses["conclusion_loss"] = conclusion_loss
        losses["total_loss"] = total_loss

        return losses

    def _compute_fact_loss(
        self,
        fact_embeddings: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss for fact extraction."""
        if "fact_embeddings" not in targets:
            return torch.tensor(0.0, device=fact_embeddings.device)

        target_embeddings = targets["fact_embeddings"]
        return self.mse_loss(fact_embeddings, target_embeddings)

    def _compute_relation_loss(
        self,
        relation_logits: torch.Tensor,
        relation_weights: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss for relation prediction."""
        if "relation_predicates" not in targets:
            return torch.tensor(0.0, device=relation_logits.device)

        predicate_loss = self.ce_loss(
            relation_logits.view(-1, relation_logits.size(-1)),
            targets["relation_predicates"].view(-1),
        )

        if "relation_weights" in targets:
            weight_loss = self.mse_loss(relation_weights, targets["relation_weights"])
            return predicate_loss + 0.1 * weight_loss

        return predicate_loss

    def _compute_conclusion_loss(
        self,
        theme_logits: torch.Tensor,
        summary_embeddings: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss for conclusion generation."""
        if "conclusion_themes" not in targets:
            return torch.tensor(0.0, device=theme_logits.device)

        theme_loss = self.ce_loss(theme_logits, targets["conclusion_themes"])

        if "conclusion_summaries" in targets:
            summary_loss = self.mse_loss(
                summary_embeddings,
                targets["conclusion_summaries"],
            )
            return theme_loss + 0.1 * summary_loss

        return theme_loss


def train_epoch(
    model: HierarchicalNeuralReasoner,
    dataloader: DataLoader,
    criterion: HRMLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    log_every: int = 10,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: The neural reasoner model.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.
        epoch: Current epoch number.
        writer: Optional TensorBoard writer.
        log_every: Log every N batches.

    Returns:
        Dictionary of average losses for the epoch.
    """
    model.train()

    total_losses = {
        "fact_loss": 0.0,
        "relation_loss": 0.0,
        "conclusion_loss": 0.0,
        "total_loss": 0.0,
    }

    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        observations = batch["observations"]

        optimizer.zero_grad()

        result = model.analyze(observations, reconstruct_objects=False)

        fact_embeddings = torch.stack([fe.embedding for fe in result.fact_embeddings])
        relation_logits = torch.stack(
            [re.predicate_embedding for re in result.relation_embeddings]
        )
        relation_weights = torch.tensor(
            [re.weight for re in result.relation_embeddings],
            device=device,
        )

        conclusion_theme_logits = torch.stack(
            [ce.theme_embedding for ce in result.conclusion_embeddings]
        )
        conclusion_summary_embeddings = torch.stack(
            [ce.summary_embedding for ce in result.conclusion_embeddings]
        )

        targets = None
        if "facts" in batch or "relations" in batch or "conclusions" in batch:
            targets = _prepare_targets(batch, device)

        losses = criterion(
            fact_embeddings,
            relation_logits,
            relation_weights,
            conclusion_theme_logits,
            conclusion_summary_embeddings,
            targets,
        )

        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [
                p
                for module in [model.sensory_encoder, model.relation_network, model.abstract_reasoner]
                for p in module.parameters()
            ],
            max_norm=1.0,
        )
        optimizer.step()

        for key in total_losses:
            total_losses[key] += losses[key].item()

        num_batches += 1

        if batch_idx % log_every == 0:
            pbar.set_postfix(
                {
                    "loss": losses["total_loss"].item(),
                    "fact": losses["fact_loss"].item(),
                    "rel": losses["relation_loss"].item(),
                    "conc": losses["conclusion_loss"].item(),
                }
            )

            if writer is not None:
                global_step = epoch * len(dataloader) + batch_idx
                for key, value in losses.items():
                    writer.add_scalar(f"train/{key}", value.item(), global_step)

    for key in total_losses:
        total_losses[key] /= num_batches

    return total_losses


def validate(
    model: HierarchicalNeuralReasoner,
    dataloader: DataLoader,
    criterion: HRMLoss,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model: The neural reasoner model.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device to run on.
        writer: Optional TensorBoard writer.
        epoch: Current epoch number.

    Returns:
        Dictionary of average losses for validation.
    """
    model.eval()

    total_losses = {
        "fact_loss": 0.0,
        "relation_loss": 0.0,
        "conclusion_loss": 0.0,
        "total_loss": 0.0,
    }

    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            observations = batch["observations"]

            result = model.analyze(observations, reconstruct_objects=False)

            fact_embeddings = torch.stack([fe.embedding for fe in result.fact_embeddings])
            relation_logits = torch.stack(
                [re.predicate_embedding for re in result.relation_embeddings]
            )
            relation_weights = torch.tensor(
                [re.weight for re in result.relation_embeddings],
                device=device,
            )

            conclusion_theme_logits = torch.stack(
                [ce.theme_embedding for ce in result.conclusion_embeddings]
            )
            conclusion_summary_embeddings = torch.stack(
                [ce.summary_embedding for ce in result.conclusion_embeddings]
            )

            targets = None
            if "facts" in batch or "relations" in batch or "conclusions" in batch:
                targets = _prepare_targets(batch, device)

            losses = criterion(
                fact_embeddings,
                relation_logits,
                relation_weights,
                conclusion_theme_logits,
                conclusion_summary_embeddings,
                targets,
            )

            for key in total_losses:
                total_losses[key] += losses[key].item()

            num_batches += 1

    for key in total_losses:
        total_losses[key] /= num_batches

    if writer is not None:
        for key, value in total_losses.items():
            writer.add_scalar(f"val/{key}", value, epoch)

    return total_losses


def _prepare_targets(batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """Prepare ground truth targets from batch."""
    targets = {}

    if "facts" in batch:
        pass

    if "relations" in batch:
        pass

    if "conclusions" in batch:
        pass

    return targets


def main(config_path: str) -> None:
    """Main training function."""
    config = OmegaConf.load(config_path)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    model = build_neural_reasoner(
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        num_predicate_classes=config.model.num_predicate_classes,
        num_theme_classes=config.model.num_theme_classes,
        device=device,
    )

    criterion = HRMLoss(
        fact_weight=config.loss.fact_weight,
        relation_weight=config.loss.relation_weight,
        conclusion_weight=config.loss.conclusion_weight,
    )

    optimizer = optim.AdamW(
        [
            {"params": model.sensory_encoder.parameters()},
            {"params": model.relation_network.parameters()},
            {"params": model.abstract_reasoner.parameters()},
        ],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs,
    )

    train_dataset = HRMDataset([])
    val_dataset = HRMDataset([])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_fn,
    )

    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    checkpoint_dir = Path(config.logging.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(config.training.num_epochs):
        train_losses = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            writer,
            log_every=config.logging.log_every,
        )

        val_losses = validate(
            model,
            val_loader,
            criterion,
            device,
            writer,
            epoch,
        )

        scheduler.step()

        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_losses['total_loss']:.4f}")
        print(f"  Val Loss: {val_losses['total_loss']:.4f}")

        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": {
                        "sensory_encoder": model.sensory_encoder.state_dict(),
                        "relation_network": model.relation_network.state_dict(),
                        "abstract_reasoner": model.abstract_reasoner.state_dict(),
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_losses["total_loss"],
                },
                checkpoint_path,
            )
            print(f"  Saved checkpoint to {checkpoint_path}")

        if epoch % config.logging.save_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": {
                        "sensory_encoder": model.sensory_encoder.state_dict(),
                        "relation_network": model.relation_network.state_dict(),
                        "abstract_reasoner": model.abstract_reasoner.state_dict(),
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_losses["total_loss"],
                },
                checkpoint_path,
            )

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hierarchical Reasoning Model")
    parser.add_argument(
        "--config",
        type=str,
        default="training/config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    main(args.config)

