import argparse
import os
import re
from typing import Generator
import pandas

class HoldoutSplitter:

    def __init__(self, dataset:pandas.DataFrame, target_folder) -> None:
        # save onlz the RxnSmilesClean and split columns
        self.data = dataset[["RxnSmilesClean", "split"]]
        # change the column names
        self.data.columns = ['reactants>reagents>production', "split"]
        self.target_folder = target_folder

    def _remove_reagents(self, reaction_string):
        # remove everthing between > and > from the reaction_string via regex
        reactants, spectators, products = reaction_string.split(">")
        return reactants + ">>" + products

    def remove_reagents(self):
        print("filter the reagents out of the reactions...")
        self.data["reactants>reagents>production"] = self.data["reactants>reagents>production"].apply(self._remove_reagents)


    def save_splits(self):
        print("Saving splits...")    
        for spl in ["train", "valid", "test"]:
            if spl == "valid":
                spl_ = "val"
            else:
                spl_ = spl
            d = self.data[self.data["split"] == spl][self.data.columns[:-1]]
            d = d.reset_index(drop=True)
            d.to_csv(os.path.join(self.target_folder, f"raw_{spl_}.csv"))

if __name__ == '__main__':  

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--target_folder', type=str, required=True)
    args = parser.parse_args()


    splitter = HoldoutSplitter(pandas.read_csv(args.dataset), args.target_folder)
    splitter.remove_reagents()
    splitter.save_splits()