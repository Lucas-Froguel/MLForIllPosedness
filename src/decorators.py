from src.settings import kernels

def register(func):
    kernels[func.__name__] = func
    return func

