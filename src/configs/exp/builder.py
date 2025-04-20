import pkgutil
import importlib



def build_exp(exp_name="exp_0"):
    module = importlib.import_module(f".{exp_name}", package=__package__)
    func = getattr(module, "get_exp")
    factory = func()
    return factory