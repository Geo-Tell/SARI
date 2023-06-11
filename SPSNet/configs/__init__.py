# import configurations_gid
# import configurations_deepglobe
from .configurations_gid import Config as Config_gid
from .configurations_deepglobe import Config as Config_deepglobe

config_factory = {'config_gid': Config_gid(),
                  'config_deepglobe': Config_deepglobe(),
        }

