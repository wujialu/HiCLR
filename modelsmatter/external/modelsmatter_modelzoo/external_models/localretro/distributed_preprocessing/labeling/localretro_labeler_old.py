from external_models.LocalRetro.LocalTemplate.template_extractor import extract_from_reaction
from collections import defaultdict

import os
import argparse
import pandas
from rdkit import Chem
from external.modelsmatter_modelzoo.ssbenchmark.ssmodels.model_localretro import model_localretro

class LocalRetroSlurmLabeler:

    def __init__(self, training_templates_path, raw_data_path, output_dir, split, retro = True, verbose = False, use_stereo = True, max_edit_n = 8, min_template_n = 1):

        self.training_templates_path = training_templates_path
        self.raw_data_path = raw_data_path
        self.output_dir = output_dir

        self.retro = retro
        self.verbose = verbose
        self.use_stereo = use_stereo

        self.max_edit_n = max_edit_n
        self.min_template_n = min_template_n

        self.extractor = self.build_template_extractor()
        self.split = split

        self.reactions = pandas.read_csv(raw_data_path)['reactants>reagents>production']

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

    def get_edit_site_retro(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        A = [a for a in range(mol.GetNumAtoms())]
        B = []
        for atom in mol.GetAtoms():
            others = []
            bonds = atom.GetBonds()
            for bond in bonds:
                atoms = [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
                other = [a for a in atoms if a != atom.GetIdx()][0]
                others.append(other)
            b = [(atom.GetIdx(), other) for other in sorted(others)]
            B += b
        return A, B
    
    def get_edit_site_forward(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        A = [a for a in range(mol.GetNumAtoms())]
        B = []
        for atom in mol.GetAtoms():
            others = []
            bonds = atom.GetBonds()
            for bond in bonds:
                atoms = [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
                other = [a for a in atoms if a != atom.GetIdx()][0]
                others.append(other)
            b = [(atom.GetIdx(), other) for other in sorted(others)]
            B += b
        V = []
        for a in A:
            V += [(a,b) for b in A if a != b and (a,b) not in B]
        return V, B

    def load_training_atom_bond_templates(self):
        training_template_dicts = {}
        if self.retro:
            keys = ['atom', 'bond']
        else:
            keys = ['real', 'virtual']
            
        for site in keys:
            template_df = pandas.read_csv('%s/%s_templates.csv' % (self.training_templates_path, site))
            template_dict = {template_df['Template'][i]:template_df['Class'][i] for i in template_df.index}
            print ('loaded %s %s templates' % (len(template_dict), site))
            training_template_dicts[site] = template_dict
                                            
        training_template_infos = pandas.read_csv('%s/template_infos.csv' % self.training_templates_path)
        training_template_infos = {t: {'edit_site': eval(e), 'frequency': f} for t, e, f in zip(training_template_infos['Template'], training_template_infos['edit_site'], training_template_infos['Frequency'])}
        print ('loaded total %s templates' % len(training_template_infos))
        return training_template_dicts, training_template_infos

    def get_reaction_templates(self):
        # get the reaction templates
        print (f"Extracting Reaction templates from Split: {self.split}", end='', flush=True)
        reaction_templates = []
        processed_reactions = []
        reaction_strings = []
        ##################### PARALLELIZATION #####################
        for i, reaction in enumerate(self.reactions):
            try:
                
                
                rxn, reaction_template = self.get_reaction_template(reaction, i)
                
                reaction_templates.append(reaction_template)
                processed_reactions.append(rxn)
                reaction_strings.append(reaction)

            except Exception as e:
                # CAREFUL HERE, WE NEED TO ADD EMPTY REACTIONS BECAUSE THEY AARE STILL USED
                reaction_strings.append(reaction)
                reaction_templates.append(None)
                processed_reactions.append(None)
                
                print (i, e)
                
            if i % 100 == 0:
                print ("Number of processed reactions {}".format(i), end='', flush=True)

        return reaction_strings, processed_reactions, reaction_templates


    
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

    def labeling_dataset(self, original_reaction_strings:list , processed_reactions:list, reaction_templates: list, training_template_dicts, training_template_infos):
        # labeling the dataset
        reactants = []
        products = []
        reagents = []
        labels = []
        frequency = []
        success = 0

        # loop through processed_reactions and reaction_templates
        for i, (original_reaction_string, processed_reaction, reaction_template) in enumerate(zip(original_reaction_strings, processed_reactions, reaction_templates)):

            product = original_reaction_string.split('>>')[1]
            reagent = ''
            rxn_labels = []
            try:

                template = reaction_template['reaction_smarts']
                reactant = reaction_template['reactants']
                product = reaction_template['products']
                reagent = '.'.join(reaction_template['necessary_reagent'])
                edits = {edit_type: edit_bond[0] for edit_type, edit_bond in reaction_template['edits'].items()}
                H_change, Charge_change, Chiral_change = reaction_template['H_change'], reaction_template['Charge_change'], reaction_template['Chiral_change']
                if self.use_stereo:
                    Chiral_change = reaction_template['Chiral_change']
                else:
                    Chiral_change = {}
                template_H = self.get_full_template(template, H_change, Charge_change, Chiral_change)
                
                if template_H not in training_template_infos.keys():
                    reactants.append(reactant)
                    products.append(product)
                    reagents.append(reagent)
                    labels.append(rxn_labels)
                    frequency.append(0)
                    continue
                    
            except Exception as e:
                print (i, e)
                reactants.append(reactant)
                products.append(product)
                reagents.append(reagent)
                labels.append(rxn_labels)
                frequency.append(0)
                continue
            
            edit_n = 0
            for edit_type in edits:
                if edit_type == 'C':
                    edit_n += len(edits[edit_type])/2
                else:
                    edit_n += len(edits[edit_type])
                
                
            if edit_n <= self.max_edit_n:
                try:
                    success += 1
                    if self.retro:
                        atom_sites, bond_sites = self.get_edit_site_retro(product)
                        for edit_type, edit in edits.items():
                            for e in edit:
                                if edit_type in ['A', 'R']:
                                    rxn_labels.append(('a', atom_sites.index(e), training_template_dicts['atom'][template_H]))
                                else:
                                    rxn_labels.append(('b', bond_sites.index(e), training_template_dicts['bond'][template_H]))
                        reactants.append(reactant)
                        products.append(product)
                        reagents.append(reagent)       
                        labels.append(rxn_labels)
                        frequency.append(training_template_infos[template_H]['frequency'])
                    else:
                        if len(reagent) != 0:
                            reactant = '%s.%s' % (reactant, reagent)
                        virtual_sites, real_sites = self.get_edit_site_forward(reactant)
                        for edit_type, bonds in edits.items():
                            for bond in bonds:
                                if edit_type != 'A':
                                    rxn_labels.append(('r', real_sites.index(bond), training_template_dicts['real']['%s_%s' % (template_H, edit_type)]))
                                else:
                                    rxn_labels.append(('v', virtual_sites.index(bond), training_template_dicts['virtual']['%s_%s' % (template_H, edit_type)]))
                        reactants.append(reactant)
                        products.append(reactant)
                        reagents.append(reagent)
                        labels.append(rxn_labels)
                        frequency.append(training_template_infos[template_H]['frequency'])
                    
                except Exception as e:
                    print (i,e)
                    reactants.append(reactant)
                    products.append(product)
                    reagents.append(reagent)
                    labels.append(rxn_labels)
                    frequency.append(0)
                    continue
                    
                if i % 100 == 0:
                    print ('\r Processing %s data..., success %s data (%s/%s)' % (self.raw_data_path, success, i, len(processed_reactions)), end='', flush=True)
            else:
                print ('\nReaction # %s has too many edits (%s)...may be wrong mapping!' % (i, edit_n))
                reactants.append(reactant)
                products.append(product)
                reagents.append(reagent)
                labels.append(rxn_labels)
                frequency.append(0)
        
        df = pandas.DataFrame({'Reactants': reactants, 'Products': products, 'Reagents': reagents, 'Labels': labels, 'Frequency': frequency})
        return df


    def run_labeling(self):
        # Load training templates
        training_template_dicts, training_template_infos = self.load_training_atom_bond_templates()
        reaction_strings, processed_reactions, reaction_templates = self.get_reaction_templates()

        labeled_dataset = self.labeling_dataset(reaction_strings, processed_reactions, reaction_templates, training_template_dicts, training_template_infos)
        labeled_dataset.to_csv('%s/preprocessed_%s.csv' % (self.output_dir, self.split))

        return labeled_dataset

#### joining

    def combine_preprocessed_data(self, train_pre, val_pre, test_pre):

        train_valid = train_pre
        val_valid = val_pre
        test_valid = test_pre
        
        train_valid['Split'] = ['train'] * len(train_valid)
        val_valid['Split'] = ['val'] * len(val_valid)
        test_valid['Split'] = ['test'] * len(test_valid)
        all_valid = train_valid.append(val_valid, ignore_index=True)
        all_valid = all_valid.append(test_valid, ignore_index=True)
        all_valid['Mask'] = [int(f>=self.min_template_n) for f in all_valid['Frequency']]
        print ('Valid data size: %s' % len(all_valid))
        all_valid.to_csv('%s/labeled_data.csv' % self.output_dir, index = None)
        return

    def make_simulate_output(self, split = 'test'):
        df = pandas.read_csv('%s/preprocessed_%s.csv' % (self.output_dir, split))
        with open('%s/simulate_output.txt' % self.output_dir, 'w') as f:
            f.write('Test_id\tReactant\tProduct\t%s\n' % '\t'.join(['Edit %s\tProba %s' % (i+1, i+1) for i in range(self.max_edit_n)]))
            for i in df.index:
                labels = []
                for y in eval(df['Labels'][i]):
                    if y != 0:
                        labels.append(y)
                if len(labels) == 0:
                    lables = [(0, 0)]
    #             print (['%s\t%s' % (l, 1.0) for l in labels])
                string_labels = '\t'.join(['%s\t%s' % (l, 1.0) for l in labels])
                f.write('%s\t%s\t%s\t%s\n' % (i, df['Reactants'][i], df['Products'][i], string_labels))
        return 

    def join_results(self):
        # load the preprocessed data without an index

        train_pre = pandas.read_csv('%s/preprocessed_train.csv' % self.output_dir, index_col=0)
        val_pre = pandas.read_csv('%s/preprocessed_val.csv' % self.output_dir, index_col=0)
        test_pre = pandas.read_csv('%s/preprocessed_test.csv' % self.output_dir, index_col=0)
        
        self.make_simulate_output()
        self.combine_preprocessed_data(train_pre, val_pre, test_pre)


    



    

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp', '--training_template_path', help='The saved training templates')
    parser.add_argument('-raw', '--raw_data_path', help='The raw file')
    parser.add_argument("-o", "--output_dir", help="Output directory")
    parser.add_argument('-s', '--split', type=str,  help='The split to process (train, valid, test)')

    parser.add_argument('-r', '--retro', default=True,  help='Retrosyntheis or forward synthesis (True for retrosnythesis)')
    parser.add_argument('-v', '--verbose', default=False,  help='Verbose during template extraction')
    parser.add_argument('-stereo', '--use-stereo', default=True,  help='Use stereo info in template extraction')
    parser.add_argument('-max', '--max-edit-n', default=8,  help='Maximum number of edit number')
    parser.add_argument('-min', '--min-template-n', type=int, default=1,  help='Minimum of template frequency')
    parser.add_argument('-j', '--join', type=bool, default =False,  help='Instead of running on a split, join all the parts')
    args = parser.parse_args().__dict__
    

    model_instance = model_localretro(module_path="<Path>/external_models/aidd_localretro")
    
    # instanciate the class


    localRetroSlurmLabeler = LocalRetroSlurmLabeler(training_templates_path= args["training_template_path"], raw_data_path = args["raw_data_path"], output_dir= args["output_dir"], split = args["split"] , retro = args["retro"], verbose = args["verbose"], use_stereo = args["use_stereo"], max_edit_n = args["max_edit_n"], min_template_n = args["min_template_n"])

    if args["join"]:
        localRetroSlurmLabeler.join_results()
    else:
        localRetroSlurmLabeler.run_labeling()
