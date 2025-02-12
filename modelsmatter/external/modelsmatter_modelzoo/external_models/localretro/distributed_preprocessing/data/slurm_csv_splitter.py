import argparse
import os
from typing import Generator
import pandas

class DataSplitter:

    def __init__(self, dataset:pandas.DataFrame, chunk_size, target_folder, file_name, localretro_graph_creation, output_folder_name) -> None:
        self.dataset = dataset

        if not localretro_graph_creation:
            # rename columns for compatibility with LocalRetro
            self.dataset.columns = ['', 'reactants>reagents>production']

        self.chunk_size = chunk_size
        self.target_folder = target_folder
        self.file_name = file_name
        self.localretro_graph_creation = localretro_graph_creation
        self.output_folder_name = output_folder_name

    # split pandas dataframe into n parts
    def split(self) -> Generator:
        for i in range(0, len(self.dataset), self.chunk_size):
            yield self.dataset[i:i+self.chunk_size]
    
    def save(self, chunk, i):
        if self.localretro_graph_creation:
            # mkdir
            folder_name = f'{self.target_folder}/{self.output_folder_name}_batch_{i}'
            os.mkdir(folder_name)

            chunk.to_csv(f'{folder_name}/{self.file_name}.csv', index=False)
        else:
            chunk.to_csv('%s/%s_batch_%s.csv' % (self.target_folder, self.file_name, i), index=False)

    
    def run(self):
        for i, chunk in enumerate(self.split()):
            self.save(chunk, i)

    @staticmethod
    def get_file_name_without_file_ending(file_path:str):
        return file_path.split('/')[-1].split('.')[0]


if __name__ == '__main__':  

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--chunk_size', type=int, required=True)
    parser.add_argument('--target_folder', type=str, required=True)
    parser.add_argument('--localretro_graph_creation', action='store_true', required=False, default = False)
    parser.add_argument('--output_folder_name', type=str, required=False)
    args = parser.parse_args()

    file_name = DataSplitter.get_file_name_without_file_ending(args.dataset)

    splitter = DataSplitter(pandas.read_csv(args.dataset), args.chunk_size, args.target_folder, file_name, args.localretro_graph_creation, args.output_folder_name)
    splitter.run()