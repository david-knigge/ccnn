from omegaconf import OmegaConf
import pandas as pd


def flatten_configdict(
    cfg: OmegaConf,
    separation_mark: str = ".",
):
    """
    Returns a nested OmecaConf dict as a flattened dict, merged with the separation mark.
    Example: With separation_mark == '.', {'data':{'this': 1, 'that': 2} is returned as
    {'data.this': 1, 'data.that': 2}.

    :param cfg:
    :param sep:
    :return:
    """
    cfgdict = OmegaConf.to_container(cfg)
    cfgdict = pd.json_normalize(cfgdict, sep=separation_mark)

    return cfgdict.to_dict(orient="records")[0]
