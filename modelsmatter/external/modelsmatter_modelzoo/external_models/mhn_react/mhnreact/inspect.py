# -*- coding: utf-8 -*-
"""
Author: Philipp Seidl
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz
Contact: seidl@ml.jku.at

File contains functions that
"""

import os

import torch

from . import model

MODEL_PATH = "data/model/"  # TODO: get rid of this


def smarts2svg(smarts, useSmiles=True, highlightByReactant=True, save_to=""):
    """
    draws smiles of smarts to an SVG and displays it in the Notebook,
    or optinally can be saved to a file `save_to`
    adapted from https://www.kesci.com/mw/project/5c7685191ce0af002b556cc5
    """
    # adapted from https://www.kesci.com/mw/project/5c7685191ce0af002b556cc5
    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from IPython.display import SVG, display
    from rdkit import Chem, Geometry, RDConfig
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    rxn = AllChem.ReactionFromSmarts(smarts, useSmiles=useSmiles)
    d = Draw.MolDraw2DSVG(900, 100)

    # rxn = AllChem.ReactionFromSmarts('[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][NH2:6]>CC(O)C.[Pt]>[CH3:1][C:2](=[O:3])[NH:6][CH3:5].[OH2:4]',useSmiles=True)
    colors = [(0.3, 0.7, 0.9), (0.9, 0.7, 0.9), (0.6, 0.9, 0.3), (0.9, 0.9, 0.1)]
    try:
        d.DrawReaction(rxn, highlightByReactant=highlightByReactant)
        d.FinishDrawing()

        txt = d.GetDrawingText()
        # self.assertTrue(txt.find("<svg") != -1)
        # self.assertTrue(txt.find("</svg>") != -1)

        svg = d.GetDrawingText()
        svg2 = svg.replace("svg:", "")
        svg3 = SVG(svg2)
        display(svg3)

        if save_to != "":
            with open(save_to, "w") as f_handle:
                f_handle.write(svg3.data)
    except:
        print("Error drawing")

    return svg2


def list_models(model_path=MODEL_PATH):
    """returns a list of loadable models"""
    return dict(enumerate(list(filter(lambda k: str(k)[-3:] == ".pt", os.listdir(model_path)))))


def load_clf(model_fn="", model_path=MODEL_PATH, device="cpu", model_type="mhn"):
    """returns the model with loaded weights given a filename"""
    import json

    config_fn = "_".join(model_fn.split("_")[-2:]).split(".pt")[0]
    conf_dict = json.load(open(f"{model_path}{config_fn}_config.json"))
    train_conf_dict = json.load(open(f"{model_path}{config_fn}_config.json"))

    # specify the config the saved model had
    conf = model.ModelConfig(**conf_dict)
    conf.device = device
    print(conf.__dict__)

    if model_type == "staticQK":
        clf = model.StaticQK(conf)
    elif model_type == "mhn":
        clf = model.MHN(conf)
    elif model_type == "segler":
        clf = model.SeglerBaseline(conf)
    elif model_type == "fortunato":
        clf = model.SeglerBaseline(conf)
    else:
        raise NotImplementedError("model_type", model_type, "not found")

    # load the model
    PATH = model_path + model_fn
    params = torch.load(PATH, map_location=torch.device(conf.device))
    clf.load_state_dict(params, strict=False)
    if "templates+noise" in params.keys():
        print("loading templates+noise")
        clf.templates = params["templates+noise"]
        # clf.templates.to(clf.config.device)
    return clf
