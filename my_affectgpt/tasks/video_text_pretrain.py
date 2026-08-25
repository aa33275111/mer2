"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from my_affectgpt.common.registry import registry
from my_affectgpt.tasks.base_task import BaseTask


@registry.register_task("video_text_pretrain")
class VideoTextPretrainTask(BaseTask): # 所有内容继承自 video_text_pretrain task
    def __init__(self):
        super().__init__()

    # 2026-08-25: 实现验证集评测 (基于 val CE loss; 选最优 epoch 用, 最终指标仍是 test10 的 EW-F1)
    def valid_step(self, model, samples):
        loss = model(samples)["loss"]
        return [loss.item()]

    def after_evaluation(self, val_result, split_name, epoch):
        # runner 用 agg_metrics 选最优 (越大越好), 因此取负 loss
        avg_loss = sum(val_result) / max(len(val_result), 1)
        return {
            "agg_metrics": -avg_loss,
            "val_loss": avg_loss,
            "epoch": epoch,
        }
