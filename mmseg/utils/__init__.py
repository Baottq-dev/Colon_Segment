from .collect_env import collect_env
from .logger import get_root_logger, print_log

def register_all_modules(init_default_scope=True):
    """Register all modules for mmseg."""
    # Function để tương thích với mmengine
    # Trong project này không cần làm gì đặc biệt
    pass

__all__ = ['get_root_logger', 'collect_env', 'print_log', 'register_all_modules']
