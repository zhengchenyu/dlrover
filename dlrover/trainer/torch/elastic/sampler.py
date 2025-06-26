# Copyright 2023 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
import math
import os
from typing import Dict, Iterable, Iterator, Optional, TypeVar, Union

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler, Sampler

from dlrover.python.common.log import default_logger as logger

T_co = TypeVar("T_co", covariant=True)


class ElasticDistributedSampler(DistributedSampler):
    """ElasticDistributedSampler can checkpoint unused sample indices
    and restore sample indices from the checkpoint to support
    fault-tolerance.

    Example::

    >>> dataset = torchvision.datasets.ImageFolder(
    ...     root=args.training_data,
    ...     transform=transforms.ToTensor(),
    ... )
    >>> sampler = ElasticDistributedSampler(dataset=dataset)
    >>> dataloader = DataLoader(
    ...     dataset=train_data,
    ...     batch_size=args.batch_size,
    ...     num_workers=2,
    ...     sampler=sampler,
    ... )
    >>> for epoch in range(start_epoch, n_epochs):
    ...     sampler.set_epoch(epoch)
    ...     train(dataloader)
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        fix_total_batch_size: bool = False,
    ) -> None:
        if not dist.is_initialized():
            rank = 0 if not rank else rank
            num_replicas = 1 if not num_replicas else num_replicas

        super(ElasticDistributedSampler, self).__init__(
            dataset,
            num_replicas,
            rank,
            shuffle,
            seed,
            drop_last,
        )
        self._epoch_checkpoint: Dict[int, int] = {}
        self._init_gradient_accumulation_batches(fix_total_batch_size)
        self._init_num_samples(len(self.dataset))

    def _init_gradient_accumulation_batches(self, fix_total_batch_size):
        # num_replicas: As described in DistributedSampler, num_replicas is the
        # number of processes participating in distributed training.
        # gradient_accumulation_batches: The number of micro-batch that should be trained
        # before gradients are accumulated.
        # gradient_accumulation_steps: The number of micro-batch in this rank that
        # should be trained before gradients are accumulated.
        if fix_total_batch_size:
            assert self.num_replicas is not None
            max_worker_num = int(os.getenv("WORKER_NUM", 1))
            assert max_worker_num > 0, "worker number must be greater than 0"
            local_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
            assert local_size > 0, "local world size must be greater than 0"
            self.gradient_accumulation_batches = max_worker_num * local_size
            assert self.gradient_accumulation_batches >= self.num_replicas
            self.gradient_accumulation_steps = int(
                self.gradient_accumulation_batches / self.num_replicas
            )
            remainder = self.gradient_accumulation_batches % self.num_replicas
            if self.rank < remainder:
                self.gradient_accumulation_steps += 1
        else:
            self.gradient_accumulation_batches = self.num_replicas
            self.gradient_accumulation_steps = 1

    def __iter__(self) -> Iterator[T_co]:
        indices = []  # type: ignore
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if self.epoch not in self._epoch_checkpoint:
            self._init_num_samples(len(self.dataset))
        completed_num = self._epoch_checkpoint.get(self.epoch, 0)
        indices = indices[completed_num:]
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # fmt: off
        indices = [indices[i] for i in range(len(indices)) if (i % self.gradient_accumulation_batches) % self.num_replicas == self.rank]
        # fmt: on
        assert len(indices) == self.num_samples

        return iter(indices)

    def _init_num_samples(self, remaining_samples):
        if (
            self.drop_last
            and remaining_samples % self.gradient_accumulation_batches != 0
        ):
            num = math.ceil(
                (remaining_samples - self.gradient_accumulation_batches)
                / self.gradient_accumulation_batches
            )
        else:
            num = math.ceil(
                remaining_samples / self.gradient_accumulation_batches
            )
        self.total_size = num * self.gradient_accumulation_batches
        self.num_samples = num * self.gradient_accumulation_steps

    def state_dict(self, iter_step, micro_batch_size):
        """Checkpoint the index of the last completed sample.
        In DDP training, the completed number of sample of each
        step is the micro_batch_size * num_replicas.
        """
        assert (iter_step % self.gradient_accumulation_steps) == 0
        completed_num = (
            iter_step * micro_batch_size // self.gradient_accumulation_steps
        ) * self.gradient_accumulation_batches
        state = {
            "completed_num": completed_num,
            "epoch": self.epoch,
        }
        return state

    def load_state_dict(self, state: Dict[str, int]):
        """
        Restore the uncompleted shards from a checkpoint. The shard
        client will send uncompleted shards to the DLRover job master.
        The master will assign those shards to workers to restore training.
        """
        self.epoch = int(state.get("epoch", 0))
        completed_num = int(state.get("completed_num", 0))
        dataset_size = len(self.dataset)
        if completed_num > dataset_size:
            completed_num = completed_num % dataset_size
        remaining_samples = dataset_size - completed_num
        self._epoch_checkpoint[self.epoch] = completed_num
        self._init_num_samples(remaining_samples)
        logger.info(
            "Load epoch = %s, completed num = %s, num_samples = %s",
            self.epoch,
            completed_num,
            self.num_samples,
        )


class ElasticBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,
        drop_last: bool = False,
        fix_total_batch_size: bool = False,
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.rank = self.sampler.rank
        self._init_gradient_accumulation_batches(
            fix_total_batch_size, self.sampler.rank, self.sampler.num_replicas
        )

    def _init_gradient_accumulation_batches(
        self, fix_total_batch_size, rank, num_replicas
    ):
        # num_replicas: As described in DistributedSampler, num_replicas is the
        # number of processes participating in distributed training.
        # gradient_accumulation_batches: The number of micro-batch that should be trained
        # before gradients are accumulated.
        # gradient_accumulation_steps: The number of micro-batch in this rank that
        # should be trained before gradients are accumulated.
        if fix_total_batch_size:
            assert num_replicas is not None
            max_worker_num = int(os.getenv("WORKER_NUM", 1))
            assert max_worker_num > 0, "worker number must be greater than 0"
            local_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
            assert local_size > 0, "local world size must be greater than 0"
            self.gradient_accumulation_batches = max_worker_num * local_size
            assert self.gradient_accumulation_batches >= num_replicas
            self.gradient_accumulation_steps = int(
                self.gradient_accumulation_batches / num_replicas
            )
            remainder = self.gradient_accumulation_batches % num_replicas
            if rank < remainder:
                self.gradient_accumulation_steps += 1
        else:
            self.gradient_accumulation_batches = num_replicas
            self.gradient_accumulation_steps = 1

    def __iter__(self) -> Iterator[list[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        sampler_iter = iter(self.sampler)
        if self.drop_last:
            # Create multiple references to the same iterator
            args = [sampler_iter] * self.batch_size
            for batch_droplast in zip(*args):
                yield [*batch_droplast]
        else:
            index = 0
            batch_size = self.batch_size
            complete_batch_num = (
                len(self.sampler)
                // (self.batch_size * self.gradient_accumulation_steps)
            ) * self.gradient_accumulation_steps
            last_batch_size = int(
                (len(self.sampler) - complete_batch_num * self.batch_size)
                // self.gradient_accumulation_steps
            )
            if index >= complete_batch_num:
                batch_size = last_batch_size
            batch = [*itertools.islice(sampler_iter, batch_size)]
            while batch:
                yield batch
                index += 1
                if index >= complete_batch_num:
                    batch_size = last_batch_size
                batch = [*itertools.islice(sampler_iter, batch_size)]

    def __len__(self) -> int:
        if self.drop_last:
            return (
                len(self.sampler)
                // (self.batch_size * self.gradient_accumulation_steps)
            ) * self.gradient_accumulation_steps
        else:
            return (
                (
                    len(self.sampler)
                    + self.batch_size * self.gradient_accumulation_steps
                    - 1
                )
                // (self.batch_size * self.gradient_accumulation_steps)
            ) * self.gradient_accumulation_steps
