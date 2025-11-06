"""
Registering data and models (use w/ extension)
"""


class _RegisterHook(object):
    """
    Register this module's classes.
    """

    def __init__(self):
        self._flag = False

    def register(self):
        if self._flag:
            return
        self._register()
        self._flag = True

    def _register(self):
        if self._flag:
            return
        from nam.train.lightning_module import LightningModule
        from nam_slimmable._model import SlimmableWaveNet

        LightningModule.register_net_initializer(
            SlimmableWaveNet.registry_key(), SlimmableWaveNet.init_from_config
        )
        self._flag = True


_register_hook = _RegisterHook()


def register():
    _register_hook.register()
