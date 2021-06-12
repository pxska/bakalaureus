import re

from estnltk import Text
from estnltk.taggers import Retagger
from estnltk.taggers import CompoundTokenTagger
from estnltk.taggers import TokenSplitter

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

c = CompoundTokenTagger(tag_initials = False, tag_abbreviations = False, tag_hyphenations = False)

def preprocess_text(text):
    text.tag_layer(['tokens'])
    token_splitter.retag(text)
    c.tag(text)
    text.tag_layer(['morph_analysis'])
    return text