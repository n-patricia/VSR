
class Registry:
    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj, suffix=None):
        if isinstance(suffix, str):
            name = name + '_' + suffix

        assert (name not in self._obj_map), (f"An object '{name}' was already registered")

        self._obj_map[name] = obj

    def register(self, obj=None, suffix=None):
        if obj is None:
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class, suffix)
                return func_or_class

            return deco

        name = obj.__name__
        self._do_register(name, obj, suffix)

    def get(self, name, suffix='pytorch'):
        ret = self._obj_map.get(name)
        if ret is None:
            ret = self._obj_map.get(f'{name}_{suffix}')
            print(f'Name {name} is not found, use name: {name}_{suffix}')
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


ARCH_REGISTRY = Registry('arch')
DATASET_REGISTRY = Registry('dataset')
LOSS_REGISTRY = Registry('loss')
MODEL_REGISTRY = Registry('model')

