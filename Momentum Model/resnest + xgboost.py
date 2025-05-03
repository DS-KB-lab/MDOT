
"""ResNeSt + XGBoost models"""

import torch
from xgboost import XGBClassifier
from .resnet import ResNet, Bottleneck

__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269']
from .build import RESNEST_MODELS_REGISTRY

_url_format = 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [('22405ba7', 'resnest101')]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}


def resnest101(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest101'], progress=True, check_hash=True))
    return model

def xgboost(objective, intermediate_output, train_label1, val_data, val_label1):
    xgbmodel = XGBClassifier(objective='multi:softprob', num_class= ECG_data_class)
    xgbmodel.fit(intermediate_output, train_label1)
    xgbmodel.score(val_data, val_label1)