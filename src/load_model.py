# load_model.py
#
# Loads the trained QuakeXNet model straight out of this repo, so a fresh
# clone runs without patching seisbench's site-packages.
#
# The README describes an alternative install (copy quakexnet.py into
# seisbench/models/, register it in __init__.py, drop the weights into the
# seisbench cache) which enables sbm.QuakeXNet.from_pretrained("base",
# version_str="3"). That still works; this helper just doesn't require it.

import importlib.util
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(_HERE, "models", "quakexnet", "base.pt.v3")


def load_quakexnet(weights=WEIGHTS, device=None, eval_mode=True):
    """
    Return a QuakeXNet instance with the v3 weights loaded.

    Classes are ['eq', 'px', 'no', 'su'] — px is the explosion class and no is
    noise, which the detection scripts ignore.
    """
    spec = importlib.util.spec_from_file_location(
        "quakexnet", os.path.join(_HERE, "quakexnet.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.QuakeXNet()

    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)

    if device is not None:
        model.to(device)
    if eval_mode:
        model.eval()

    return model
