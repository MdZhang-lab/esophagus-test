
from src.DuAT.DuAT import DuAT
from src.Unetr.Unetr import u_netr
from src.SwinUNETR.SwinUNETR import swin_unetr
from src.CFPnet.CFPnet import CFPNet
from src.TransUnet.TransUnet import TransUNet
from src.CVCUNETR.CVCUNETR import CVCUnetr
from src.Unet.Unet import UNet
from src.FCBFormer.models import FCBFormer
from src.CVCUNETR.NewCVC import CVC_Unetr



def give_model(config):
    if config.finetune.model_choose == 'CVC_UNETR':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = CVC_Unetr(**config.models.cvc_unetr.branch1)
        else:
            model = CVC_Unetr(**config.models.cvc_unetr.branch5)
    elif config.finetune.model_choose == 'TransUNet':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = TransUNet(**config.models.trans_unet.branch1)
        else:
            model = TransUNet(**config.models.trans_unet.branch5)
    elif config.finetune.model_choose == 'CFPNet':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = CFPNet(**config.models.cfp_net.branch1)
        else:
            model = CFPNet(**config.models.cfp_net.branch5)
    elif config.finetune.model_choose == 'UNETR':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = u_netr(**config.models.u_netr.branch1)
        else:
            model = u_netr(**config.models.u_netr.branch5)
    elif config.finetune.model_choose == 'SWINUNETR':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = swin_unetr(**config.models.swin_unetr.branch1)
        else:
            model = swin_unetr(**config.models.swin_unetr.branch5)
    elif config.finetune.model_choose == 'DuAT':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = DuAT(**config.models.duat.branch1)
        else:
            model = DuAT(**config.models.duat.branch5)
    elif config.finetune.model_choose == 'UNet':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = UNet(**config.models.unet.branch1)
        else:
            model = UNet(**config.models.unet.branch5)
    elif config.finetune.model_choose == 'FCBFormer':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = FCBFormer(**config.models.FCBFormer.branch1)
        else:
            model = FCBFormer(**config.models.FCBFormer.branch5)
    return model