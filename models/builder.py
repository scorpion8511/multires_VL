import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms

def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH
        
def get_encoder(model_name, target_img_size=224, ckpt_path: str | None = None):
    """Return an encoder model and its preprocessing transforms.

    If ``ckpt_path`` is provided it is used directly; otherwise the
    corresponding environment variable (``UNI_CKPT_PATH`` or
    ``CONCH_CKPT_PATH``) is expected to be set.
    """

    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    elif model_name == 'uni_v1':
        if ckpt_path is None:
            HAS_UNI, UNI_CKPT_PATH = has_UNI()
            assert HAS_UNI, 'UNI is not available'
        else:
            UNI_CKPT_PATH = ckpt_path
        model = timm.create_model(
            "vit_large_patch16_224",
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        if ckpt_path is None:
            HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
            assert HAS_CONCH, 'CONCH is not available'
        else:
            CONCH_CKPT_PATH = ckpt_path
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))

    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(
        mean=constants['mean'],
        std=constants['std'],
        target_img_size=target_img_size,
    )

    return model, img_transforms
