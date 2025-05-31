import warnings

try:
    from mmcv.utils import Registry, build_from_cfg
except ImportError:
    # Trong MMCV 2.2.0, Registry và build_from_cfg đã chuyển sang mmengine
    try:
        from mmengine.registry import Registry, build_from_cfg
    except ImportError:
        # Fallback đơn giản cho Registry
        class Registry:
            def __init__(self, name):
                self.name = name
                self._module_dict = {}
            
            def register_module(self, name=None, force=False, module=None):
                def _register(cls):
                    self._module_dict[name or cls.__name__] = cls
                    return cls
                if module is not None:
                    return _register(module)
                return _register
            
            def get(self, key):
                return self._module_dict.get(key)
        
        def build_from_cfg(cfg, registry, default_args=None):
            # Fallback đơn giản
            if isinstance(cfg, dict):
                args = cfg.copy()
                obj_type = args.pop('type')
                if isinstance(obj_type, str):
                    obj_cls = registry.get(obj_type)
                else:
                    obj_cls = obj_type
                return obj_cls(**args)
            else:
                raise TypeError(f'cfg must be a dict, but got {type(cfg)}')

from torch import nn

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
LOSSES = Registry('loss')
SEGMENTORS = Registry('segmentor')


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """

    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)


def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
