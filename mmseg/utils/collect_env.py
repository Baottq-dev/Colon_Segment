try:
    from mmcv.utils import collect_env as collect_base_env
    from mmcv.utils import get_git_hash
except ImportError:
    # Trong MMCV 2.2.0, các function này có thể đã bị di chuyển hoặc bỏ
    try:
        from mmengine.utils import collect_env as collect_base_env
        from mmengine.utils import get_git_hash
    except ImportError:
        def collect_base_env():
            return {}
        def get_git_hash():
            return "unknown"

import mmseg


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMSegmentation'] = f'{mmseg.__version__}+{get_git_hash()[:7]}'

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print('{}: {}'.format(name, val))
