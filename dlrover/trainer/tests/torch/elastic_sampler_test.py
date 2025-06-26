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
import os
import random
import unittest

import numpy as np
from torch.utils.data import Dataset

from dlrover.trainer.torch.elastic.sampler import (
    ElasticBatchSampler,
    ElasticDistributedSampler,
)


class SimpleDataset(Dataset):
    def __init__(self, len=60001):
        self.data = np.arange(0, len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class ElasticDistributedSamplerTest(unittest.TestCase):
    def test_checkpoint(self):
        dataset = SimpleDataset()
        sampler = ElasticDistributedSampler(
            dataset=dataset,
            num_replicas=2,
            rank=0,
            shuffle=False,
        )
        batch_size = 8
        step = 0
        sampler_state = None
        for i, v in enumerate(sampler):
            if i % batch_size == 0:
                step = i / batch_size
            if step == 4:
                sampler_state = sampler.state_dict(step, batch_size)
                break
        self.assertEqual(sampler_state["completed_num"], 64)

        sampler = ElasticDistributedSampler(
            dataset=dataset,
            num_replicas=3,
            rank=0,
            shuffle=False,
        )
        sampler.load_state_dict(sampler_state)
        val = next(iter(sampler))
        self.assertEqual(val, 64)

        for i in sampler:
            pass
        sampler.set_epoch(1)
        val = next(iter(sampler))
        self.assertEqual(val, 0)

        sampler = ElasticDistributedSampler(
            dataset=dataset,
            num_replicas=2,
            rank=0,
            shuffle=False,
            drop_last=True,
        )
        sampler.load_state_dict(sampler_state)
        val = next(iter(sampler))
        self.assertEqual(val, 64)

    def test_checkpoint_with_scaling(self):
        dataset = SimpleDataset(len=60000)
        # 1 Train with 8 replicas, epoch is 0
        batch_size = 8
        step = 0
        checkpoint_step = 4
        num_replicas = 8
        sampler = ElasticDistributedSampler(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=0,
            shuffle=False,
        )
        sampler.set_epoch(0)

        # 2 Save the checkpoint
        sampler_state = None
        val = 0
        for i, v in enumerate(sampler):
            self.assertEqual(val, v)
            val += num_replicas
            if i % batch_size == 0:
                step = i / batch_size
            if step == checkpoint_step:
                sampler_state = sampler.state_dict(step, batch_size)
                break
        self.assertEqual(
            sampler_state["completed_num"], 8 * batch_size * checkpoint_step
        )

        # 3 Resume with 6 replicas from checkpoint, and epoch is 0
        sampler.set_epoch(0)
        num_replicas = 6
        sampler = ElasticDistributedSampler(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=0,
            shuffle=False,
        )
        sampler.load_state_dict(sampler_state)
        val = 8 * batch_size * checkpoint_step
        for i in sampler:
            self.assertEqual(val, i)
            val += num_replicas

        # 4 Continue, but epoch is 1
        sampler.set_epoch(1)
        val = 0
        for i in sampler:
            self.assertEqual(val, i)
            val += num_replicas

    def test_fix_total_batch_size(self):
        dataset_length = 800
        dataset = SimpleDataset(len=dataset_length)
        os.environ["WORKER_NUM"] = "7"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        batch_size = 10

        # Case 1: real worker size is 7, do not drop last
        num_replicas = 7
        for rank in range(num_replicas):
            sampler = ElasticDistributedSampler(
                dataset=dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                fix_total_batch_size=True,
            )
            self.assertEqual(sampler.gradient_accumulation_batches, 7)
            self.assertEqual(sampler.gradient_accumulation_steps, 1)
            # The total size must be greater than or equal to 800 and
            # rounded up to the nearest multiple of 7.
            self.assertEqual(sampler.total_size, 805)
            # The length of the sample should be 115, which is 805 divided by 7
            self.assertEqual(len(sampler), 115)

            # Iterate over the sampler and save the state at a random step.
            # We should save the state when all total batch is done, Otherwise
            # it will cause inconsistency.
            checkpoint_index = (
                random.randint(0, len(sampler)) // batch_size * batch_size
            )
            for i, v in enumerate(sampler):
                if checkpoint_index == i:
                    assert (
                        i % batch_size == 0
                    ), "Checkpoint index must be a multiple of batch size"
                    step = i / batch_size
                    sampler_state = sampler.state_dict(step, batch_size)
                    break
                self.assertEqual(v, i * num_replicas + rank)

            # Load from checkpoint and continue iteration
            completed_num = sampler_state["completed_num"]
            sampler.load_state_dict(sampler_state)
            self.assertEqual(sampler.total_size, 805 - completed_num)
            self.assertEqual(
                len(sampler), (805 - completed_num) // num_replicas
            )
            for i, v in enumerate(sampler):
                expected_value = completed_num + i * num_replicas + rank
                if expected_value >= 800:
                    expected_value = expected_value - 800 + completed_num
                self.assertEqual(v, expected_value)

        # Case 2: real worker size is 3, do not drop last
        num_replicas = 3
        for rank in range(num_replicas):
            num_micro_batche_per_total_batch = 3 if rank == 0 else 2
            sampler = ElasticDistributedSampler(
                dataset=dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                fix_total_batch_size=True,
            )
            self.assertEqual(sampler.gradient_accumulation_batches, 7)
            self.assertEqual(
                sampler.gradient_accumulation_steps,
                num_micro_batche_per_total_batch,
            )
            # The total size must be greater than or equal to 800 and
            # rounded up to the nearest multiple of 7.
            self.assertEqual(sampler.total_size, 805)
            # The number of sample must be equal to 805 divided by 7 * their steps.
            # For rank0, steps is 3, should be 805 / 7 * 3 = 345.
            # for rank1 and rank2, steps is 2. should be 805 / 7 * 2 = 230.
            self.assertEqual(
                len(sampler), 805 / 7 * num_micro_batche_per_total_batch
            )

            # Iterate over the sampler and save the state at a random step.
            # We should save the state when all total batch is done, Otherwise
            # it will cause inconsistency.
            # For rank0, per total batch, will train 3 micro batches.
            # For rank1 and rank2, per total batch, will train 2 micro batches.
            checkpoint_index = (
                random.randint(0, len(sampler))
                // (num_micro_batche_per_total_batch * batch_size)
                * ((num_micro_batche_per_total_batch * batch_size))
            )
            current_start = 0
            current_end = current_start + 7
            current_value = current_start + rank
            for i, v in enumerate(sampler):
                if checkpoint_index == i:
                    assert (
                        i % batch_size == 0
                    ), "Checkpoint index must be a multiple of batch size"
                    step = i / batch_size
                    sampler_state = sampler.state_dict(step, batch_size)
                    break
                if current_value >= current_end:
                    current_start += 7
                    current_end = current_start + 7
                    current_value = current_start + rank
                self.assertEqual(v, current_value)
                current_value += num_replicas

            # Load from checkpoint and continue iteration
            completed_num = sampler_state["completed_num"]
            sampler.load_state_dict(sampler_state)
            self.assertEqual(sampler.total_size, 805 - completed_num)
            self.assertEqual(
                len(sampler),
                (805 - completed_num) // 7 * num_micro_batche_per_total_batch,
            )

            current_start = completed_num
            current_end = current_start + 7
            current_value = completed_num + rank
            for i, v in enumerate(sampler):
                if current_value >= current_end:
                    current_start += 7
                    current_end = current_start + 7
                    current_value = current_start + rank
                if current_value >= 800:
                    current_value = current_value - 800 + completed_num
                self.assertEqual(v, current_value)
                current_value += num_replicas

        # Case 3: real worker size is 7, drop last
        num_replicas = 7
        for rank in range(num_replicas):
            sampler = ElasticDistributedSampler(
                dataset=dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                drop_last=True,
                fix_total_batch_size=True,
            )
            self.assertEqual(sampler.gradient_accumulation_batches, 7)
            self.assertEqual(sampler.gradient_accumulation_steps, 1)
            # total size must be less than or equal to 800 and
            # rounded down to the nearest multiple of 7.
            self.assertEqual(sampler.total_size, 798)
            # The length of the sample should be 114, which is 798 divided by 7
            self.assertEqual(len(sampler), 114)

            # Iterate over the sampler and save the state at a random step.
            # We should save the state when all total batch is done, Otherwise
            # it will cause inconsistency.
            checkpoint_index = (
                random.randint(0, len(sampler)) // batch_size * batch_size
            )
            for i, v in enumerate(sampler):
                if checkpoint_index == i:
                    assert (
                        i % batch_size == 0
                    ), "Checkpoint index must be a multiple of batch size"
                    step = i / batch_size
                    sampler_state = sampler.state_dict(step, batch_size)
                    break
                self.assertEqual(v, i * num_replicas + rank)

            # Load from checkpoint and continue iteration
            completed_num = sampler_state["completed_num"]
            sampler.load_state_dict(sampler_state)
            self.assertEqual(sampler.total_size, 798 - completed_num)
            self.assertEqual(
                len(sampler), (798 - completed_num) // num_replicas
            )
            for i, v in enumerate(sampler):
                expected_value = completed_num + i * num_replicas + rank
                self.assertEqual(v, expected_value)

        # Case 4: real worker size is 4, drop last
        num_replicas = 3
        for rank in range(num_replicas):
            num_micro_batche_per_total_batch = 3 if rank == 0 else 2
            sampler = ElasticDistributedSampler(
                dataset=dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                drop_last=True,
                fix_total_batch_size=True,
            )
            self.assertEqual(sampler.gradient_accumulation_batches, 7)
            self.assertEqual(
                sampler.gradient_accumulation_steps,
                num_micro_batche_per_total_batch,
            )
            # total size must be less than or equal to 800 and
            # rounded down to the nearest multiple of 7.
            self.assertEqual(sampler.total_size, 798)
            # The number of sample must be equal to 798 divided by 7 * their steps.
            # For rank0, steps is 3, should be 798 / 7 * 3 = 342.
            # for rank1 and rank2, steps is 2. should be 798 / 7 * 2 = 228.
            self.assertEqual(
                len(sampler), 798 / 7 * num_micro_batche_per_total_batch
            )

            # Iterate over the sampler and save the state at a random step.
            # We should save the state when all total batch is done, Otherwise
            # it will cause inconsistency.
            # For rank0, per total batch, will train 3 micro batches.
            # For rank1 and rank2, per total batch, will train 2 micro batches.
            checkpoint_index = (
                random.randint(0, len(sampler))
                // (num_micro_batche_per_total_batch * batch_size)
                * ((num_micro_batche_per_total_batch * batch_size))
            )
            current_start = 0
            current_end = current_start + 7
            current_value = current_start + rank
            for i, v in enumerate(sampler):
                if checkpoint_index == i:
                    assert (
                        i % batch_size == 0
                    ), "Checkpoint index must be a multiple of batch size"
                    step = i / batch_size
                    sampler_state = sampler.state_dict(step, batch_size)
                    break
                if current_value >= current_end:
                    current_start += 7
                    current_end = current_start + 7
                    current_value = current_start + rank
                self.assertEqual(v, current_value)
                current_value += num_replicas

            # Load from checkpoint and continue iteration
            completed_num = sampler_state["completed_num"]
            sampler.load_state_dict(sampler_state)
            self.assertEqual(sampler.total_size, 798 - completed_num)
            self.assertEqual(
                len(sampler),
                (798 - completed_num) // 7 * num_micro_batche_per_total_batch,
            )

            current_start = completed_num
            current_end = current_start + 7
            current_value = completed_num + rank
            for i, v in enumerate(sampler):
                if current_value >= current_end:
                    current_start += 7
                    current_end = current_start + 7
                    current_value = current_start + rank
                self.assertEqual(v, current_value)
                current_value += num_replicas

    def test_batch_sample_fix_total_batch_size(self):
        dataset_length = 800
        dataset = SimpleDataset(len=dataset_length)
        os.environ["WORKER_NUM"] = "7"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        batch_size = 10

        # Case 1: real worker size is 7, do not drop last
        num_replicas = 7
        for rank in range(num_replicas):
            sampler = ElasticDistributedSampler(
                dataset=dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                fix_total_batch_size=True,
            )
            # The total size of sample must be greater than or equal to 800 and
            # rounded up to the nearest multiple of 7, is equal to 805.
            self.assertEqual(sampler.total_size, 805)
            batch_sampler = ElasticBatchSampler(
                sampler, batch_size=batch_size, fix_total_batch_size=True
            )

            # The length of batch sample is 805/10/7 = 11.42, rounded up to 12
            self.assertEqual(len(batch_sampler), 12)
            for i, batch in enumerate(batch_sampler):
                if i < 11:
                    self.assertEqual(len(batch), batch_size)
                else:
                    # The length of last batch =
                    # (sampler.total_size - 11 * num_replicas * batch_size) /batch_size =
                    # (805 - 11 * 7 * 10) / 7 = 5
                    self.assertEqual(len(batch), 5)
                for j, v in enumerate(batch):
                    expected_value = (
                        i * num_replicas * batch_size + j * num_replicas + rank
                    )
                    if expected_value >= 800:
                        expected_value = expected_value - 800
                    self.assertEqual(v, expected_value)

        # Case 2: real worker size is 3, do not drop last
        num_replicas = 3
        for rank in range(num_replicas):
            num_micro_batche_per_total_batch = 3 if rank == 0 else 2
            sampler = ElasticDistributedSampler(
                dataset=dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                fix_total_batch_size=True,
            )
            # The total size of sample must be greater than or equal to 800 and
            # rounded up to the nearest multiple of 7, is equal to 805.
            self.assertEqual(sampler.total_size, 805)
            batch_sampler = ElasticBatchSampler(
                sampler, batch_size=batch_size, fix_total_batch_size=True
            )
            # The length of batch sample is 805/10/7 = 11.42, rounded up to 12
            # For rank0, steps is 3, the length of batch sample is 12 * 3
            # For rank1 and rank2, steps is 2, the length of batch sample is 12 * 2
            self.assertEqual(
                len(batch_sampler), 12 * num_micro_batche_per_total_batch
            )
            current_start = 0
            current_end = current_start + 7
            current_value = current_start + rank
            for i, batch in enumerate(batch_sampler):
                # For rank0, the length of batch is 345.
                # The length of batch0 - batch32 should be 10.
                # The length of batch33 - batch35 should be 5.
                # The same with rank1 and rank2.
                if i < 11 * num_micro_batche_per_total_batch:
                    self.assertEqual(len(batch), batch_size)
                else:
                    # The length of last batch =
                    # (sampler.total_size - 11 * num_replicas * batch_size)/batch_size =
                    # (805 - 11 * 7 * 10) / 7 = 5
                    self.assertEqual(len(batch), 5)
                for j, v in enumerate(batch):
                    if current_value >= current_end:
                        current_start += 7
                        current_end = current_start + 7
                        current_value = current_start + rank
                    if current_value >= 800:
                        current_value = current_value - 800
                    self.assertEqual(v, current_value)
                    current_value += num_replicas

        # Case 3: real worker size is 7, drop last
        num_replicas = 7
        for rank in range(num_replicas):
            sampler = ElasticDistributedSampler(
                dataset=dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                drop_last=True,
                fix_total_batch_size=True,
            )
            # total size must be less than or equal to 800 and
            # rounded down to the nearest multiple of 7.
            self.assertEqual(sampler.total_size, 798)
            batch_sampler = ElasticBatchSampler(
                sampler,
                batch_size=batch_size,
                drop_last=True,
                fix_total_batch_size=True,
            )

            # The length of batch sample is 798/10/7 = 11.4, rounded down to 11
            self.assertEqual(len(batch_sampler), 11)
            for i, batch in enumerate(batch_sampler):
                self.assertEqual(len(batch), batch_size)
                for j, v in enumerate(batch):
                    expected_value = (
                        i * num_replicas * batch_size + j * num_replicas + rank
                    )
                    self.assertEqual(v, expected_value)

        # Case 4: real worker size is 3, drop last
        num_replicas = 3
        for rank in range(num_replicas):
            num_micro_batche_per_total_batch = 3 if rank == 0 else 2
            sampler = ElasticDistributedSampler(
                dataset=dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                drop_last=True,
                fix_total_batch_size=True,
            )
            # total size must be less than or equal to 800 and
            # rounded down to the nearest multiple of 7.
            self.assertEqual(sampler.total_size, 798)
            batch_sampler = ElasticBatchSampler(
                sampler,
                batch_size=batch_size,
                drop_last=True,
                fix_total_batch_size=True,
            )
            # The length of batch sample is 798/10/7 = 11.4, rounded down to 11
            # For rank0, steps is 3, the length of batch sample is 11 * 3
            # For rank1 and rank2, steps is 2, the length of batch sample is 11 * 2
            self.assertEqual(
                len(batch_sampler), 11 * num_micro_batche_per_total_batch
            )
            current_start = 0
            current_end = current_start + 7
            current_value = current_start + rank
            for i, batch in enumerate(batch_sampler):
                self.assertEqual(len(batch), batch_size)
                for j, v in enumerate(batch):
                    if current_value >= current_end:
                        current_start += 7
                        current_end = current_start + 7
                        current_value = current_start + rank
                    if current_value >= 800:
                        current_value = current_value - 800
                    self.assertEqual(v, current_value)
                    current_value += num_replicas
