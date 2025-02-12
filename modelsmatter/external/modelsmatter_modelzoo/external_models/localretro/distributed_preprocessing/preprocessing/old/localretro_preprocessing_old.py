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
    
    def __init__(self, data_path, output_dir, retro = True, verbose = False, use_stereo = True):

        self.output_dir = output_dir
        print("output dir: ", self.output_dir)

        self.retro = retro
        self.verbose = verbose
        self.use_stereo = use_stereo

        self.extractor = self.build_template_extractor()

        self.data_path = data_path
        self.reactions = pandas.read_csv(data_path)['reactants>reagents>production']
            

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

    def get_full_template(self, template, H_change, Charge_change, Chiral_change):
        H_code = ''.join([str(H_change[k+1]) for k in range(len(H_change))])
        Charge_code = ''.join([str(Charge_change[k+1]) for k in range(len(Charge_change))])
        Chiral_code = ''.join([str(Chiral_change[k+1]) for k in range(len(Chiral_change))])
        if Chiral_code == '':
            return '_'.join([template, H_code, Charge_code])
        else:
            return '_'.join([template, H_code, Charge_code, Chiral_code])

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

    def process_reaction_templates(self, reaction_templates: list):
        # process the reaction templates

        TemplateEdits = {}
        TemplateCs = {}
        TemplateHs = {}
        TemplateSs = {}
        TemplateFreq = defaultdict(int)
        templates_A = defaultdict(int)
        templates_B = defaultdict(int)
        
        # loop through reaction_templates
        for i, reaction_template in enumerate(reaction_templates):

            try:
                reactant = reaction_template['reactants']
                template = reaction_template['reaction_smarts']
                edits = reaction_template['edits']
                H_change = reaction_template['H_change']
                Charge_change = reaction_template['Charge_change']
                if self.use_stereo:
                    Chiral_change = reaction_template['Chiral_change']
                else:
                    Chiral_change = {}
                template_H = self.get_full_template(template, H_change, Charge_change, Chiral_change)
                if template_H not in TemplateHs.keys():
                    TemplateEdits[template_H] = {edit_type: edits[edit_type][2] for edit_type in edits}
                    TemplateHs[template_H] = H_change
                    TemplateCs[template_H] = Charge_change
                    TemplateSs[template_H] = Chiral_change
    
                TemplateFreq[template_H] += 1

                if self.retro:
                    for edit_type, bonds in edits.items():
                        bonds = bonds[0]
                        if len(bonds) > 0:
                            if edit_type in ['A', 'R']:
                                templates_A[template_H] += 1
                            else:
                                templates_B[template_H] += 1

                else:
                    for edit_type, bonds in edits.items():
                        bonds = bonds[0]
                        if len(bonds) > 0:
                            if edit_type != 'A':
                                templates_A['%s_%s' % (template_H, edit_type)] += 1
                            else:
                                templates_B['%s_%s' % (template_H, edit_type)] += 1
                    
            except Exception as e:
                print (i, e)
                
            if i % 100 == 0:
                print ('\r i = %s, # of template: %s, # of atom template: %s, # of bond template: %s' % (i, len(TemplateFreq), len(templates_A), len(templates_B)), end='', flush=True)
        print ('\n total # of template: %s' %  len(TemplateFreq))
        
        if self.retro:
            derived_templates = {'atom':templates_A, 'bond': templates_B}
        else:
            derived_templates = {'real':templates_A, 'virtual': templates_B}
            
        TemplateInfos = pandas.DataFrame({'Template': k, 'edit_site':TemplateEdits[k], 'change_H': TemplateHs[k], 'change_C': TemplateCs[k], 'change_S': TemplateSs[k], 'Frequency': TemplateFreq[k]} for k in TemplateHs.keys())
        TemplateInfos.to_csv('%s/template_infos.csv' % self.output_dir)
        
        return derived_templates
    
    def export_template(self, derived_templates):
        for k in derived_templates.keys():
            local_templates = derived_templates[k]
            templates = []
            template_class = []
            template_freq = []
            sorted_tuples = sorted(local_templates.items(), key=lambda item: item[1])
            c = 1
            for t in sorted_tuples:
                templates.append(t[0])
                template_freq.append(t[1])
                template_class.append(c)
                c += 1
            #template_dict = {templates[i]:i+1  for i in range(len(templates)) }
            template_df = pandas.DataFrame({'Template' : templates, 'Frequency' : template_freq, 'Class': template_class})

            template_df.to_csv('%s/%s_templates.csv' % (self.output_dir, k))
        return

    def extract_and_save_templates(self):
        # run the templates extractoin
        reaction_templates = self.get_reaction_templates(self.reactions)

        # pickle the result
        with open(self.output_dir, 'wb') as f:
            pickle.dump(reaction_templates, f)

    def run_template_creation(self):
        # RUN THE TEMPLATE EXTRACTION
        # load the readtion template

        with open(self.output_dir, 'rb') as f:
            reaction_templates = pickle.load(f)

        derived_templates = self.process_reaction_templates(reaction_templates)
        self.export_template(derived_templates)

    

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', help='Dataset to use')

    parser.add_argument('-out', '--output_dir', help='Output path')

    parser.add_argument('-r', '--retro', default=True,  help='Retrosyntheis or forward synthesis (True for retrosnythesis)')
    parser.add_argument('-v', '--verbose', default=False,  help='Verbose during template extraction')
    parser.add_argument('-stereo', '--use-stereo', default=True,  help='Use stereo info in template extraction')

    parser.add_argument("-c", "--create_templates",type=bool, default = False, help="Create the templates or load them from a file")

    args = parser.parse_args().__dict__
    

    model_instance = model_localretro(module_path="<Path>/external_models/LocalRetro")
    
    localRetroSlurmPreprocessing = LocalRetroSlurmPreprocessing(data_path = args["dataset_path"], output_dir = args["output_dir"] , retro = args["retro"], verbose = args["verbose"], use_stereo = args["use_stereo"])
    
    if args["create_templates"]:
        localRetroSlurmPreprocessing.extract_and_save_templates()
    else:
        localRetroSlurmPreprocessing.run_template_creation()
