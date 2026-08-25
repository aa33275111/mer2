import os
import logging
import warnings

import config

from my_affectgpt.common.registry import registry
from my_affectgpt.datasets.datasets.base_dataset import BaseDataset
from my_affectgpt.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from my_affectgpt.datasets.datasets.mer2026ov_dataset import MER2026OV_Dataset
from my_affectgpt.datasets.datasets.human_dataset import Human_Dataset
from my_affectgpt.datasets.datasets.mercaptionplus_dataset import MERCaptionPlus_Dataset


# get name -> dataset_cls
def get_name2cls(dataset):
    if dataset == 'Human': return Human_Dataset()
    if dataset == 'MERCaptionPlus': return MERCaptionPlus_Dataset()
    if dataset == 'MER2026OV': return MER2026OV_Dataset()
    print ('dataset cls not provided!')
    return None


@registry.register_builder("mercaptionplus")
class MERCaptionPlus_Builder(BaseDatasetBuilder):
    train_dataset_cls = MERCaptionPlus_Dataset

    def build_datasets(self):
        logging.info("Building datasets MERCaptionPlus_Dataset")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        # 2026-08-25: 训练集 = mc+ train csv, 验证集 = mc+ val csv (各 300 条, 见 split_mcplus_val.py)
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        datasets['val'] = dataset_cls(
            # 用 train processor (eval 未配置会是 None, 会导致 transform 崩溃)
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            label_path=config.PATH_TO_LABEL['MERCaptionPlusVal'],
            )
        return datasets
    

@registry.register_builder("human")
class Human_Builder(BaseDatasetBuilder):
    train_dataset_cls = Human_Dataset

    def build_datasets(self):
        logging.info("Building datasets Human_Dataset")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets


@registry.register_builder("mer2026ov")
class MER2026OV_Builder(BaseDatasetBuilder):
    train_dataset_cls = MER2026OV_Dataset

    def build_datasets(self):
        logging.info("Building datasets MER2026OV_Dataset")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets