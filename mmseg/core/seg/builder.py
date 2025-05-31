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

PIXEL_SAMPLERS = Registry('pixel sampler')


def build_pixel_sampler(cfg, **default_args):
    """Build pixel sampler for segmentation map."""
    return build_from_cfg(cfg, PIXEL_SAMPLERS, default_args)
