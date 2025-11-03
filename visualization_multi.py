import os
import sys
from datetime import datetime
from typing import Dict

import monai
import pytz
import torch
import yaml
import cv2
from PIL import Image
import numpy as np
import shutil
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory
import albumentations as A
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from src import utils
from src.models import give_model
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, load_pretrain_model
import warnings
warnings.filterwarnings('ignore')

def load_model_with_safetensors(model, checkpoint_path, accelerator):
    from safetensors.torch import load_file
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict)
    return model

def wram_up(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
                    train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
                    post_trans: monai.transforms.Compose, accelerator: Accelerator, epoch: int, step: int):
    # wram up
    model.train()
    for i, image_batch in enumerate(train_loader):
        logits = model(image_batch[0])
        total_loss = 0
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits, image_batch[1])
            total_loss += alpth * loss
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch[1])

        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        # break
    scheduler.step(epoch)
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update({
        f'Train/mean {metric_name}': float(batch_acc.mean())})
    return step


def get_mask_visualization(output_path, val_loader: torch.utils.data.DataLoader, 
                           test_loader: torch.utils.data.DataLoader, inference: monai.inferers.Inferer, 
                           post_trans: monai.transforms.Compose):  
    if not os.path.exists(output_path):
        os.makedirs(output_path) 
    else:
        shutil.rmtree(output_path)
        os.makedirs(output_path)

    # 可视化    
    for i, image_batch in enumerate(val_loader):
        logits = inference(image_batch[0], model)
        logits = post_trans(logits)
        for j, test_batch in enumerate(test_loader):
            if i == j :
                or_size = (test_batch[1].size()[-2], test_batch[1].size()[-1])
                original_image = transforms.ToPILImage()(test_batch[0][0].cpu())
                file_name = test_batch[2]
                overlayed_image = np.array(original_image)
                original_mask = np.zeros_like(original_image, dtype=np.uint8) #np.array(original_image)
                # original_output_path = os.path.join(output_path, f'{file_name}_original.png')
                # imageio.imwrite(original_output_path, original_image)
                channel_colors = [(255, 0, 0), (255, 255, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255)]
                for channel in range(logits.shape[1]):
                    channel_output = logits[:, channel, :, :]
                    origin_channel = test_batch[1][:, channel, :, :]
                    channel_output = F.interpolate(channel_output.unsqueeze(0), size=or_size, mode='nearest').squeeze(0)
                    mask_array = channel_output.cpu().squeeze().numpy() * 255
                    origin_array = origin_channel.cpu().squeeze().numpy() * 255
                    pred_map = np.zeros_like(original_image, dtype=np.uint8)
                    origin_map = np.zeros_like(original_image, dtype=np.uint8)
                    # error_map = np.zeros_like(original_image, dtype=np.uint8)
                    interest_mask = mask_array == 255
                    pred_map[interest_mask,0] = channel_colors[channel][0]
                    pred_map[interest_mask,1] = channel_colors[channel][1]
                    pred_map[interest_mask,2] = channel_colors[channel][2]
                    origin_mask = origin_array == 255
                    origin_map[origin_mask,0] = channel_colors[channel][0]
                    origin_map[origin_mask,1] = channel_colors[channel][1]
                    origin_map[origin_mask,2] = channel_colors[channel][2]
                    # matches = (mask_array == origin_array) & interest_mask
                    # differences = ~matches & interest_mask
                    # error_map[matches, 1] = 255  # 绿色
                    # error_map[differences, 0] = 255  # 红色

                    # label_output_path = os.path.join(output_path, f'{file_name}_{channel+1}.png')
                    # imageio.imwrite(label_output_path, pred_map)
                    # error_output_path = os.path.join(output_path, f'{file_name}_{channel+1}_error.png')
                    # imageio.imwrite(error_output_path, error_map)
                    overlayed_image = cv2.addWeighted(overlayed_image, 0.5, pred_map, 0.5, 0)
                    original_mask = cv2.addWeighted(original_mask, 0.5, origin_map, 0.5, 0)
                    accelerator.print(f'{file_name}_channel{channel+1} has been added!')
                # Save the overlaid image
                overlayed_output_path = os.path.join(output_path, f'{file_name}_overlay.png')
                origin_mask_path = os.path.join(output_path, f'{file_name}_original_mask.png')
                imageio.imwrite(overlayed_output_path, overlayed_image)
                imageio.imwrite(origin_mask_path, original_mask)
                accelerator.print(f'{file_name}_overlay has been saved!')
            else:
                accelerator.print(f'ficture is not matched')
        

    
            
          


if __name__ == '__main__':
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    utils.same_seeds(50)
    logging_dir = os.getcwd() + '/logs/' + config.finetune.checkpoint +str(datetime.now())
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], project_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print('Load Model...')
    model = give_model(config)
    

    visualization_path = config.visualization.visualization_path + f"/{config.finetune.model_choose}/"
    
    image_size = config.dataset.CVC_ClinicDB.image_size
    
    accelerator.print('Load Dataloader...')
    if config.trainer.dataset_choose == 'CVC_ClinicDB':
        from src.CVCLoder import get_dataloader
        train_loader, val_loader = get_dataloader(config,dataset_choose='CVC_ClinicDB')
        include_background = False
    elif config.trainer.dataset_choose == 'Kvasir_SEG':
        from src.CVCLoder import get_dataloader
        train_loader, val_loader = get_dataloader(config,dataset_choose='Kvasir_SEG')
        include_background = False
    elif config.trainer.dataset_choose == 'EDD_seg':
        from src.EDDLoader import get_dataloader
        train_loader, val_loader, test_loader = get_dataloader(config)
        include_background = True

    inference = monai.inferers.SlidingWindowInferer(roi_size=ensure_tuple_rep(image_size, 2), overlap=0.5,
                                                    sw_device=accelerator.device, device=accelerator.device)
    metrics = {
        'dice_metric': monai.metrics.DiceMetric(include_background=include_background,
                                                reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=True),
        'miou_metric':monai.metrics.MeanIoU(include_background=include_background),
        'f1': monai.metrics.ConfusionMatrixMetric(include_background=include_background, metric_name='f1 score'),
        'precision': monai.metrics.ConfusionMatrixMetric(include_background=include_background, metric_name="precision"),
        'recall': monai.metrics.ConfusionMatrixMetric(include_background=include_background, metric_name="recall"),
        'hd95_metric': monai.metrics.HausdorffDistanceMetric(percentile=95, include_background=include_background, reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=True)
    }
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
    ])
    
    # 定义训练参数
    optimizer = optim_factory.create_optimizer_v2(model, opt=config.trainer.optimizer,
                                                  weight_decay=config.trainer.weight_decay,
                                                  lr=config.trainer.lr, betas=(0.9, 0.95))
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup,
                                              max_epochs=config.trainer.num_epochs)
    loss_functions = {
        'focal_loss': monai.losses.FocalLoss(to_onehot_y=False),
        'dice_loss': monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True),
    }
    
    # 加载最优模型
    checkpoint_path = os.path.join(os.getcwd(), "model_store", config.finetune.checkpoint, "best", "model.safetensors")
    model = load_model_with_safetensors(model, checkpoint_path, accelerator)
    # model = load_pretrain_model(f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/pytorch_model.bin", model,
    #                             accelerator)
    # 加载验证
    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler,
                                                                                train_loader, val_loader)
    # warm up
    step = 0
    for epoch in range(0, config.trainer.warmup):
        step = wram_up(model, loss_functions, train_loader,optimizer, scheduler, metrics,post_trans, accelerator, epoch, step)
    
    # ====================================================visualization==========================================================
    # # GT visualization
    # visualization(path1_files=img_path, path1=gt_output_path, path2=gt_output_path, output_path=gt_visualization_path, error=False)
    
    # models' visualization
    # get_mask(img_path=img_path, output_path=mask_output_path, accelerator=accelerator)
    # visualization(path1_files=img_path, path1=gt_output_path, path2=mask_output_path, output_path=visualization_path, error=True)
    get_mask_visualization(output_path=visualization_path, val_loader=val_loader, test_loader=test_loader, inference=inference, post_trans=post_trans)