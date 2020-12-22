from estnltk import Text
from estnltk.converters import json_to_text
from estnltk.taggers import Retagger
from estnltk.taggers import CompoundTokenTagger

import os
import re
import time

class TokenSplitter( Retagger ):
    """Splits tokens into smaller tokens based on regular expression patterns.""" 
    conf_param = ['patterns', 'break_group_name']
    
    def __init__(self, patterns, break_group_name:str='end'):
        # Set input/output layers
        self.input_layers = ['tokens']
        self.output_layer = 'tokens'
        self.output_attributes = ()
        # Set other configuration parameters
        if not (isinstance(break_group_name, str) and len(break_group_name) > 0):
            raise TypeError('(!) break_group_name should be a non-empty string.')
        self.break_group_name = break_group_name
        # Assert that all patterns are regular expressions in the valid format
        if not isinstance(patterns, list):
            raise TypeError('(!) patterns should be a list of compiled regular expressions.')
        # TODO: we use an adhoc way to verify that patterns are regular expressions 
        #       because there seems to be no common way of doing it both in py35 
        #       and py36
        for pat in patterns:
            # Check for the existence of methods/attributes
            has_match   = callable(getattr(pat, "match", None))
            has_search  = callable(getattr(pat, "search", None))
            has_pattern = getattr(pat, "pattern", None) is not None
            for (k,v) in (('method match()',has_match),\
                          ('method search()',has_search),\
                          ('attribute pattern',has_pattern)):
                if v is False:
                    raise TypeError('(!) Unexpected regex pattern: {!r} is missing {}.'.format(pat, k))
            symbolic_groups = pat.groupindex
            if self.break_group_name not in symbolic_groups.keys():
                raise TypeError('(!) Pattern {!r} is missing symbolic group named {!r}.'.format(pat, self.break_group_name))
        self.patterns = patterns

    def _change_layer(self, text, layers, status):
        # Get changeble layer
        changeble_layer = layers[self.output_layer]
        # Iterate over tokens
        add_spans    = []
        remove_spans = []
        for span in changeble_layer:
            token_str = text.text[span.start:span.end]
            for pat in self.patterns:
                m = pat.search(token_str)
                if m:
                    break_group_end = m.end( self.break_group_name )
                    if break_group_end > -1 and \
                       break_group_end > 0  and \
                       break_group_end < len(token_str):
                        # Make the split
                        add_spans.append( (span.start, span.start+break_group_end) )
                        add_spans.append( (span.start+break_group_end, span.end) )
                        remove_spans.append( span )
                        # Once a token has been split, then break and move on to 
                        # the next token ...
                        break
        if add_spans:
            assert len(remove_spans) > 0
            for old_span in remove_spans:
                changeble_layer.remove_span( old_span )
            for new_span in add_spans:
                changeble_layer.add_annotation( new_span )

token_splitter = TokenSplitter(patterns=[re.compile(r'(?P<end>[A-ZÕÄÖÜ]{1}\w+)[A-ZÕÄÖÜ]{1}\w+'),\
                                         re.compile(r'(?P<end>Piebenomme)metsawaht'),\
                                         re.compile(r'(?P<end>maa)peal'),\
                                         re.compile(r'(?P<end>reppi)käest'),\
                                         re.compile(r'(?P<end>Kiidjerwelt)J'),\
                                         re.compile(r'(?P<end>Ameljanow)Persitski'),\
                                         re.compile(r'(?P<end>mõistmas)Mihkel'),\
                                         re.compile(r'(?P<end>tema)Käkk'),\
                                         re.compile(r'(?P<end>Ahjawalla)liikmed'),\
                                         re.compile(r'(?P<end>kohtumees)A'),\
                                         re.compile(r'(?P<end>Pechmann)x'),\
                                         re.compile(r'(?P<end>pölli)Anni'),\
                                         re.compile(r'(?P<end>külla)Rauba'),\
                                         re.compile(r'(?P<end>kohtowannem)Jaak'),\
                                         re.compile(r'(?P<end>rannast)Leno'),\
                                         re.compile(r'(?P<end>wallast)Kiiwita'),\
                                         re.compile(r'(?P<end>wallas)Kristjan'),\
                                         re.compile(r'(?P<end>Pedoson)rahul'),\
                                         re.compile(r'(?P<end>pere)Jaan'),\
                                         re.compile(r'(?P<end>kohtu)poolest'),\
                                         re.compile(r'(?P<end>Kurrista)kaudo'),\
                                         re.compile(r'(?P<end>mölder)Gottlieb'),\
                                         re.compile(r'(?P<end>wöörmündri)Jaan'),\
                                         re.compile(r'(?P<end>Oinas)ja'),\
                                         re.compile(r'(?P<end>ette)Leenu'),\
                                         re.compile(r'(?P<end>Tommingas)peab'),\
                                         re.compile(r'(?P<end>wäljaja)Kotlep'),\
                                         re.compile(r'(?P<end>pea)A'),\
                                         re.compile(r'(?P<end>talumees)Nikolai')])

files = {}

with open('divided_corpus.txt', 'r', encoding = 'UTF-8') as f:
    txt = f.readlines()

for fileName in txt:
    file, subdistribution = fileName.split(":")
    files[file] = subdistribution.rstrip("\n")

files_not_working = ['J2rva_Tyri_V22tsa_id22177_1911a.json', \
                     'J2rva_Tyri_V22tsa_id18538_1894a.json', \
                     'J2rva_Tyri_V22tsa_id22155_1911a.json', \
                     'Saare_Kihelkonna_Kotlandi_id18845_1865a.json', \
                     'P2rnu_Halliste_Abja_id257_1844a.json', \
                     'Saare_Kaarma_Loona_id7575_1899a.json', \
                     'J2rva_Tyri_V22tsa_id22266_1913a.json']

start = time.time()
training_texts = []
filenames = {key: value for key, value in files.items() if value in ('1', '2', '3', '4')}
for filename in filenames:
    with open('./vallakohtufailid_json/' + str(filename), 'r', encoding='UTF-8') as file:
        if filename in files_not_working:
            continue
        else:
            training_texts.append(json_to_text(file.read()).tag_layer(['sentences', 'morph_analysis']))
print("Treeningtekstide lugemine võttis aega:", time.time() - start)

from estnltk.taggers.estner.ner_trainer import NerTrainer
from estnltk.taggers.estner.model_storage_util import ModelStorageUtil
from estnltk.core import DEFAULT_PY3_NER_MODEL_DIR

model_dir=DEFAULT_PY3_NER_MODEL_DIR
modelUtil = ModelStorageUtil(model_dir)
nersettings = modelUtil.load_settings()
trainer = NerTrainer(nersettings)

start = time.time()
trainer.train( training_texts, layer='gold_wordner', model_dir='test' )
print("Trainer võttis aega:", time.time() - start)