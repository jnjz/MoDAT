from .MoDAT import MoDAT


def build_model(type, **kwargs):
    if type == 'MoDAT':
        return MoDAT(**kwargs)
    else:
        raise ValueError
