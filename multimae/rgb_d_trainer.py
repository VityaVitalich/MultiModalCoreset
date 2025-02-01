from base_trainer import BaseTrainer
import logging
import torch
from typing import Any, Dict, List, Literal, Union

logger = logging.getLogger("MultiMAE")


def masked_berhu_loss(preds, target, mask_valid=None):
    if mask_valid is None:
        mask_valid = torch.ones_like(preds).bool()
    if preds.shape[1] != mask_valid.shape[1]:
        mask_valid = mask_valid.repeat_interleave(preds.shape[1], 1)

    diff = preds - target
    diff[~mask_valid] = 0
    with torch.no_grad():
        c = max(torch.abs(diff).max() * 0.2, 1e-5)

    l1_loss = torch.abs(diff)
    l2_loss = (torch.square(diff) + c**2) / 2.0 / c
    berhu_loss = (
        l1_loss[torch.abs(diff) < c].sum() + l2_loss[torch.abs(diff) >= c].sum()
    )

    return berhu_loss / mask_valid.sum()


class RgbDepthTrainer(BaseTrainer):
    @torch.no_grad()
    def compute_score(
        self,
        model_outputs: torch.Tensor,
        target: torch.Tensor,  # pyright: ignore unused
    ) -> float:
        """Compute metrics based on model output.

        The function is used to compute model metrics. For further logging and
        and checkpoint tracking. Any metrics could be logged, but only scalar metrics
        are used to track checkpoints.

        Args:
            model_outputs: one prediction from model forward.
            ground_truths: GT depth.

        Returns:
            A dict of metric name and metric value(s).
        """

        # TAKEN FROM NYU METRICS MULTIMAE https://github.com/EPFL-VILAB/MultiMAE/blob/main/run_finetuning_depth.py
        # map to the original scale
        # preds = preds * NYU_STD + NYU_MEAN
        # target = target * NYU_STD + NYU_MEAN
        preds = model_outputs["depth"].to("cpu")
        target = target.to("cpu")
        mask_valid = None

        if mask_valid is None:
            mask_valid = torch.ones_like(preds).bool()
        if preds.shape[1] != mask_valid.shape[1]:
            mask_valid = mask_valid.repeat_interleave(preds.shape[1], 1)

        n = mask_valid.sum()
        
        print(target.size(), preds.size())
        diff = torch.abs(preds - target)
        diff[~mask_valid] = 0

        max_rel = torch.maximum(
            preds / torch.clamp_min(target, 1e-6), target / torch.clamp_min(preds, 1e-6)
        )
        max_rel = max_rel[mask_valid]

        log_diff = torch.log(torch.clamp_min(preds, 1e-6)) - torch.log(
            torch.clamp_min(target, 1e-6)
        )
        log_diff[~mask_valid] = 0

        metrics = {
            "berhu_loss": masked_berhu_loss(preds, target),
            "rmse": (diff.square().sum() / n).sqrt(),
            "rel": (diff / torch.clamp_min(target, 1e-6))[mask_valid].mean(),
            "srel": (diff**2 / torch.clamp_min(target, 1e-6))[mask_valid].mean(),
            "log10": (log_diff.square().sum() / n).sqrt(),
            "delta_1": (max_rel < 1.25).float().mean(),
            "delta_2": (max_rel < (1.25**2)).float().mean(),
            "delta_3": (max_rel < (1.25**3)).float().mean(),
        }
        return metrics

    def compute_loss(
        self,
        model_output: Any,
        ground_truth: torch.Tensor,  # pyright: ignore unused
    ) -> torch.Tensor:
        """Compute loss for backward.

        The function is called every iteration in training loop to compute loss.

        Args:
            model_output: raw model output as is.
            ground_truth: raw depth from dataloader.
        """
        loss = masked_berhu_loss(preds=model_output["depth"], target=ground_truth)
        return loss

    def log_metrics(
        self,
        phase: Literal["train", "val"],
        metrics: Union[Dict[str, Any], None] = None,
        epoch: Union[int, None] = None,
        losses: Union[List[float], None] = None,  # pyright: ignore unused
        iterations: Union[List[int], None] = None,  # pyright: ignore unused
    ):
        """Log metrics.

        The metrics are computed based on the whole epoch data, so the granularity of
        metrics is epoch, so when the metrics are not None, the epoch is not None to.
        The loss is computed every iteraton, so when the loss values are passed, the
        corresponding iterations are also passed to the function. The metrics are
        computed on validation phase, but can also be computed for train phase. The
        loss is computed only during train phase to report the validation loss, compute
        it in the `compute_metrics` function.

        Args:
            phase: wether the metrics were collected during train or validatoin.
            metrics: a dict that is returned from `compute_metrics` every epoch.
            epoch: number of epoch after which the metrics were computed.
            losses: a list of loss values.
            iterations: a list of iteration number for corresponding loss values.
        """
        if metrics is not None:
            logger.info(f"Epoch: {epoch}; metrics on {phase}: {metrics}")
