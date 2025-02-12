from math import prod
import sys
from mhnreact.inspect import *
import pandas as pd
from mhnreact.data import load_templates, load_dataset_from_csv
from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun
import pickle
import torch
import time
from argparse import ArgumentParser
import json
import h5py, math


def run_ssretroeval(
    model_name, model_path, instance_name, oroot, iroot, batch_size, large_dataset
):  ## Directly from MHNReact
    import numpy as np
    from joblib import Memory

    cachedir = "data/cache/"
    memory = Memory(cachedir, verbose=0, bytes_limit=80e9)
    clf = load_clf(model_name, model_path, model_type="mhn", device="cpu")
    X, y, template_list, test_reactants_can = load_dataset_from_csv(
        csv_path=os.path.join(iroot, f"{instance_name}_MHN_prepro.csv.gz"),
        split_col="split",
        input_col="prod_smiles",
        ssretroeval=True,
        reactants_col="reactants_can",
        ret_df=False,
    )  # TODO: Clean up
    print(len(list(template_list.values())))
    print("testing on the real test set ;)")
    from mhnreact.data import load_templates
    from mhnreact.retroeval import run_templates, topkaccuracy
    from mhnreact.utils import sort_by_template_and_flatten
    from mhnreact.train import featurize_smiles
    from mhnreact.molutils import smarts2appl, LargeDataset, convert_smiles_to_fp

    split = "test"

    a = list(template_list.keys())
    # assert list(range(len(a))) == a
    templates = list(template_list.values())
    # templates = [*templates, *expert_templates]
    template_product_smarts = [str(s).split(">")[0] for s in templates]
    if large_dataset is not None:
        print("Large datset features:", large_dataset)
        large_dataset = LargeDataset(
            large_dataset["fp_type"] if "fp_type" in large_dataset.keys() else "morgan",
            large_dataset["template_fp_type"]
            if "template_fp_type" in large_dataset.keys()
            else "rdk",
            large_dataset["fp_size"] if "fp_size" in large_dataset.keys() else 4096,
            large_dataset["fp_radius"] if "fp_radius" in large_dataset.keys() else 2,
            template_list,
            large_dataset["njobs"] if "njobs" in large_dataset.keys() else -1,
            large_dataset["only_templates_in_batch"]
            if "only_templates_in_batch" in large_dataset.keys()
            else False,
            large_dataset["verbose"] if "verbose" in large_dataset.keys() else False,
            # cachedir=cachedir,
        )
        cached_fp = os.path.join(cachedir, f"data.hdf5")

        bs = 10000
        with h5py.File(cached_fp, "a") as f:  # UNCOMMENT THIS
            X_ = X[split]
            molecules = f.create_dataset(
                f"molecules_{split}", (len(X_), large_dataset.fp_size)
            )
            for b in range(math.ceil(len(X_) // bs) + 1):
                X_m = X_[bs * b : min(bs * (b + 1), len(X_))]

                X_m = convert_smiles_to_fp(
                    X_m,
                    which=large_dataset.fp_type,
                    fp_size=large_dataset.fp_size,
                    radius=large_dataset.fp_radius,
                    njobs=large_dataset.njobs,
                )
                molecules[bs * b : min(bs * (b + 1), len(X_))] = X_m

    else:
        X_fp = featurize_smiles(
            X,
            fp_type=large_dataset.fp_type
            if large_dataset is not None
            else "maccs+morganc+topologicaltorsion+erg+atompair+pattern+rdkc+layered+mhfp",
            fp_size=large_dataset.fp_size if large_dataset is not None else 30000,
        )

    # execute all template
    print("execute all templates")

    test_product_smarts = [xi[0] for xi in X["test"]]  # added later
    smarts2appl = memory.cache(
        smarts2appl, ignore=["njobs", "nsplits", "use_tqdm"]
    )  # UNCOMMENT THIS
    print("Running smarts2appl")
    appl = smarts2appl(test_product_smarts, template_product_smarts)
    n_pairs = len(test_product_smarts) * len(template_product_smarts)
    n_appl = len(appl[0])
    print(n_pairs, n_appl, n_appl / n_pairs)

    y_preds = None
    # forward

    print("len(X_fp[test]):", len(X[split]))
    y[split] = np.zeros(len(X[split])).astype(np.int)
    clf.eval()

    if y_preds is None:
        if large_dataset is not None:
            outputs = []
            for b in range(math.ceil(len(X[split]) // batch_size) + 1):
                i_f = batch_size * b
                i_e = min(batch_size * (b + 1), len(X[split]))
                k = list(range(i_f, i_e))
                with h5py.File(cached_fp, "r") as f:
                    X_fp = f[f"molecules_{split}"][k, :]
                y_preds_ = clf.evaluate(
                    X_fp,
                    X_fp,
                    y[split][k],
                    is_smiles=False,
                    split=split,
                    bs=batch_size,
                    only_loss=True,
                    large_dataset=large_dataset,
                    cachedir=cachedir,
                )
                outputs.append(y_preds_)
            print("LEN OUTPUTS", len(outputs))
            y_preds = np.concatenate(outputs, axis=0)

        else:
            y_preds = clf.evaluate(
                X_fp[split],
                X_fp[split],
                y[split],
                is_smiles=False if not large_dataset else True,
                split=split,
                bs=batch_size,
                only_loss=True,
                large_dataset=large_dataset,
            )

        # y_preds = clf.evaluate( # Reintroduce with new model
        #     X_fp[split] if not large_dataset else X[split],
        #     X_fp[split] if not large_dataset else X[split],
        #     y[split],
        #     is_smiles=False if not large_dataset else True,
        #     split=split,
        #     bs=batch_size,
        #     only_loss=True,
        #     large_dataset=large_dataset,
        # )
    template_scores = y_preds  # this should allready be test

    ####
    if y_preds.shape[1] > 100000:
        kth = 200
        print(f"only evaluating top {kth} applicable predicted templates")
        # only take top kth and multiply by applicability matrix
        appl_mtrx = np.zeros_like(y_preds, dtype=bool)
        appl_mtrx[appl[0], appl[1]] = 1

        appl_and_topkth = ([], [])
        for row in range(len(y_preds)):
            argpreds = np.argpartition(-(y_preds[row] * appl_mtrx[row]), kth, axis=0)[
                :kth
            ]
            # if there are less than kth applicable
            mask = appl_mtrx[row][argpreds]
            argpreds = argpreds[mask]
            # if len(argpreds)!=kth:
            #    print('changed to ', len(argpreds))

            appl_and_topkth[0].extend([row for _ in range(len(argpreds))])
            appl_and_topkth[1].extend(list(argpreds))

        appl = appl_and_topkth
    ####

    print("running the templates")
    run_templates = run_templates  # memory.cache( ) ... allready cached to tmp
    prod_idx_reactants, prod_temp_reactants = run_templates(
        test_product_smarts, templates, appl
    )

    flat_results = sort_by_template_and_flatten(
        y_preds, prod_idx_reactants, agglo_fun=sum
    )
    print(len(flat_results), len(flat_results[0]))
    opath = os.path.join(oroot, f"{instance_name}.pickle")
    with open(opath, "wb") as f:
        pickle.dump(flat_results, f)

    if large_dataset is not None:
        os.remove(cached_fp)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-mn", "--model_name", dest="model_name", help="model to run")
    parser.add_argument(
        "-mp", "--model_path", dest="model_path", help="path to model file"
    )
    parser.add_argument(
        "-i", "--instance", dest="instance_name", help="name of instance"
    )
    parser.add_argument(
        "-ip",
        "--ipath",
        dest="iroot",
        default="./input",
        help="directory where orginal data is stored",
    )
    parser.add_argument(
        "-o",
        "--opath",
        dest="oroot",
        default="./output",
        help="directory where results should be stored",
    )
    parser.add_argument(
        "-b", "--batch_size", dest="batch_size", default=32, help="batch size", type=int
    )
    parser.add_argument(
        "-ld", "--large_dataset", dest="large_dataset", default=None, type=json.loads
    )

    args = parser.parse_args()

    run_ssretroeval(
        args.model_name,
        args.model_path,
        args.instance_name,
        args.oroot,
        args.iroot,
        args.batch_size,
        args.large_dataset,
    )
