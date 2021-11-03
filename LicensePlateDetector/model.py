import torch
from .data import cfg_mnet
from .models.retina import Retina


def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def load_model(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class LicensePlateDetector:
    def __init__(self):
        torch.set_grad_enabled(False)
        cfg = cfg_mnet

        # net and model
        self.net = Retina(cfg=cfg, phase='test')
        self.net = load_model(self.net,'LicensePlateDetector/weights/mnet_plate.pth')
        self.net.eval()
        print('Finished loading model!')
        self.resize = 1

    def __call__(self, img):
        loc, conf, landms = self.net(img)  # forward pass
        return loc, conf, landms
