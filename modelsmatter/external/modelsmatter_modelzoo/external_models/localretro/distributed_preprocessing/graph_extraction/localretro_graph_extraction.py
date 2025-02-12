import json
import pickle
from external_models.aidd_localretro.LocalTemplate.template_extractor import extract_from_reaction
from collections import defaultdict

import os
import argparse
from dgl.data.utils import save_graphs, load_graphs
from external.modelsmatter_modelzoo.ssbenchmark.ssmodels.model_localretro import model_localretro

class LocalRetroGraphJoiner:
    # Join Graph GDL Files created by slurm
    
    def __init__(self, graph_batch_folder_path: str, output_file_path: str):
        self.graph_batch_folder_path = graph_batch_folder_path
        self.output_file_path = output_file_path

    def _get_all_folders(self):
        # get all folders in the self.graph_batch_folder_path
        folders = [f for f in os.listdir(self.graph_batch_folder_path) if os.path.isdir(os.path.join(self.graph_batch_folder_path, f))]
        print(f"Found folders: {folders}")
        folders.sort(key=lambda x: int(x.split('_')[-1]))
        numbers = list(map(lambda x: int(x.split('_')[-1]), folders))
        # get the maximum number from the numbers list
        max_number = max(numbers)
        assert len(folders) == max_number + 1, "missing files"
        print(f"found {len(folders)} folders, expected {max_number + 1}")
        print(f"sorted input files: {folders}" )

        # add the path to the folders
        folders = list(map(lambda x: self.graph_batch_folder_path + '/' + x, folders))
        print(folders)
        return folders

    @staticmethod
    def _find_bin_file_in_folder(folder_path):
        files = [f for f in os.listdir(folder_path) if f.endswith('.bin')]
        assert len(files) == 1, "found more than one bin file in the folder"
        return files[0]

    def join_graphs_in_folder(self):
        sorted_folders = self._get_all_folders()
        graphs = []

        for folder in sorted_folders:
            graph_bin_file_path = folder + '/' + self._find_bin_file_in_folder(folder)
            print(f"loading graph from {graph_bin_file_path}")
            graph, label_dict = load_graphs(graph_bin_file_path)
            graphs += graph
        
        print(f"Saving graphs to {self.output_file_path}")
        print(f"Number of graphs: {len(graphs)}")

        save_graphs(self.output_file_path, graphs)


if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('-gp', '--graph_batch_folder_path', help='The graph folders')
    parser.add_argument('-out', '--output_file_path', help='The output file path')
    

    args = parser.parse_args().__dict__
    
    print(args)
    
    model_instance = model_localretro(module_path="<Path>/external_models/aidd_localretro")
    
    localRetroGraphJoiner = LocalRetroGraphJoiner(args['graph_batch_folder_path'], args['output_file_path'])
    localRetroGraphJoiner.join_graphs_in_folder()