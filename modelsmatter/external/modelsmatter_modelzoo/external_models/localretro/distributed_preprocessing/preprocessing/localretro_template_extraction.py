import json
import pickle
from external_models.aidd_localretro.LocalTemplate.template_extractor import extract_from_reaction
from collections import defaultdict

import os
import argparse
import pandas
from rdkit import Chem
from external.modelsmatter_modelzoo.ssbenchmark.ssmodels.model_localretro import model_localretro

class LocalRetroSlurmPreprocessing:
    # refactoring of localretros preprocessing code
    
    def __init__(self, data_path, output_dir, retro = True, verbose = False, use_stereo = True, labeling = False):

        self.output_dir = output_dir
        print("output dir: ", self.output_dir)

        self.retro = retro
        self.verbose = verbose
        self.use_stereo = use_stereo

        self.extractor = self.build_template_extractor()

        self.data_path = data_path
        if not os.path.isdir(data_path):
            self.reactions = pandas.read_csv(data_path)['reactants>reagents>production']

        self.labeling = labeling

    def build_template_extractor(self):
        # Build template extractor function based on class variables

        setting = {'verbose': False, 'use_stereo': False, 'use_symbol': False, 'max_unmap': 5, 'retro': False, 'remote': True, 'least_atom_num': 2}
        setting['verbose'] = self.verbose
        setting['use_stereo'] = self.use_stereo
        setting['retro'] = self.retro
        if self.retro:
            setting['use_symbol'] = True

        print ('Template extractor setting:', setting)
        return lambda x: extract_from_reaction(x, setting)

    def get_reaction_template(self, rxn, _id = 0):
        # Extract reaction template from a reaction
        rxn = {'reactants': rxn.split('>>')[0], 'products': rxn.split('>>')[1], '_id': _id}
        result = self.extractor(rxn)
        return rxn, result

    def get_reaction_templates(self, reactions: list):
        # get the reaction templates

        reaction_templates = []
        
        for i, reaction in enumerate(reactions):
            try:
                ##################### PARALLELIZATION #####################
                rxn, reaction_template = self.get_reaction_template(reaction, i)
                if 'reactants' not in reaction_template or 'reaction_smarts' not in reaction_template.keys():
                    print ('\ntemplate problem: id: %s' % i)
                    continue
                
                reaction_templates.append(reaction_template)

            except Exception as e:
                print (i, e)
                
            if i % 100 == 0:
                print ("Number of processed reactions {}".format(i), end='', flush=True)

        return reaction_templates

    def get_reaction_templates_for_labeling(self, reactions:list):

        reaction_strings = []
        reaction_templates = []
        processed_reactions = []

        ##################### PARALLELIZATION #####################
        for i, reaction in enumerate(reactions):
            try:
                rxn, reaction_template = self.get_reaction_template(reaction, i)
                reaction_strings.append(reaction)
                reaction_templates.append(reaction_template)
                processed_reactions.append(rxn)
                

            except Exception as e:
                # CAREFUL HERE, WE NEED TO ADD EMPTY REACTIONS BECAUSE THEY AARE STILL USED
                reaction_strings.append(reaction)
                reaction_templates.append(None)
                processed_reactions.append(None)
                
                print (i, e)
                
            if i % 100 == 0:
                print ("Number of processed reactions {}".format(i), end='', flush=True)

        return reaction_strings, processed_reactions, reaction_templates

    def extract_and_save_templates(self):
        if self.labeling:
            reaction_strings, processed_reactions, reaction_templates = self.get_reaction_templates_for_labeling(self.reactions)
        else:
            # run the templates extractoin
            reaction_templates = self.get_reaction_templates(self.reactions)

        # save the templates
        with open(self.output_dir, 'wb') as f:
            if self.labeling:
                pickle.dump(reaction_strings, f)
                pickle.dump(processed_reactions, f)
            pickle.dump(reaction_templates, f)


    def join_all_pkl_in_folder(self):
        # join all the pkl files in the folder

        # get all pkl files in the self.data_path folder
        files = [f for f in os.listdir(self.data_path) if f.endswith('.pkl')]
        print(f"input files: {files}" )
        # sort files by the part number in the file name e.g. "raw_train_batch_template_0.pkl"
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        numbers = list(map(lambda x: int(x.split('_')[-1].split('.')[0]), files))
        # get the maximum number from the numbers list
        max_number = max(numbers)
        assert len(files) == max_number + 1, "missing files"
        print(f"found {len(files)} files, expected {max_number + 1}")
        print(f"sorted input files: {files}" )
        
        reaction_strings = []
        processed_reactions = []
        reaction_templates = []
        
        for file in files:
            with open(os.path.join(self.data_path, file), 'rb') as f:
                if self.labeling:
                    reaction_strings += pickle.load(f)
                    processed_reactions += pickle.load(f)
                # append the elements of the list
                reaction_templates += pickle.load(f)

        # pickle the result
        with open(self.output_dir, 'wb') as f:
            if self.labeling:
                pickle.dump(reaction_strings, f)
                pickle.dump(processed_reactions, f)
            pickle.dump(reaction_templates, f)


if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', help='Dataset to use')

    parser.add_argument('-out', '--output_dir', help='Output path')

    parser.add_argument('-r', '--retro', default=True,  help='Retrosyntheis or forward synthesis (True for retrosnythesis)')
    parser.add_argument('-v', '--verbose', default=False,  help='Verbose during template extraction')
    parser.add_argument('-stereo', '--use-stereo', default=True,  help='Use stereo info in template extraction')

    parser.add_argument('-join', '--join_pkl', default=False,  help='Join the pkl files')
    parser.add_argument('-labeler', '--use_labeler', default=False,  help='Use the labeler version of template extraction')
    args = parser.parse_args().__dict__
    
    print(args)
    
    model_instance = model_localretro(module_path="<Path>/external_models/aidd_localretro")
    
    localRetroSlurmPreprocessing = LocalRetroSlurmPreprocessing(data_path = args["dataset_path"], 
                                                                    output_dir = args["output_dir"],
                                                                        retro = args["retro"], 
                                                                            verbose = args["verbose"], 
                                                                                use_stereo = args["use_stereo"],
                                                                                    labeling=args["use_labeler"],)

    if args["join_pkl"]:
        localRetroSlurmPreprocessing.join_all_pkl_in_folder()
    else:
        localRetroSlurmPreprocessing.extract_and_save_templates()