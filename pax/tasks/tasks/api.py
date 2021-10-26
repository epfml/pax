from typing import Any, List, Mapping, Tuple

import torch
from pax.tasks.datasets.api import Batch, Dataset
from pax.tasks.models.api import Buffers, Params
from pax.utils.accumulators import running_avg_step


class Task:
    name: str
    config: Mapping[str, Any]
    train: Dataset
    test: Dataset

    def init(self, seed: int = 0) -> Tuple[Params, Buffers]:
        pass

    def loss(
        self,
        params: Params,
        batch: Batch,
        buffers: Buffers = None,
        is_training: bool = True,
    ) -> float:
        pass

    def evaluate_batch(
        self,
        params: Params,
        batch: Batch,
        buffers: Buffers = None,
        is_training: bool = False,
    ) -> Mapping[str, torch.Tensor]:
        pass

    def evaluate(
        self,
        params: Params,
        dataset: Dataset = None,
        buffers: Buffers = None,
        is_training: bool = False,
        max_batches: int = None,
        batch_size: int = None,
    ) -> Mapping[str, torch.Tensor]:

        if dataset is None:
            dataset = self.test

        if batch_size is None:
            batch_size = self.config["eval_batch_size"]

        mean_stats = None
        with torch.no_grad():
            for batch in dataset.iterator(
                batch_size=batch_size, shuffle=False, repeat=False, num_workers=1
            ):
                results = self.evaluate_batch(
                    params, batch, buffers=buffers, is_training=is_training
                )
                mean_stats = running_avg_step(mean_stats, results, len(batch))
        return mean_stats.avg
