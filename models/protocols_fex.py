from estnltk.taggers import Tagger, Retagger
from estnltk.taggers.estner.fex import get_shape, get_2d, get_lemma, get_pos, is_prop, get_word_parts, get_case, get_ending, get_4d, get_dand, get_capperiod, get_all_other, contains_upper, contains_lower, contains_alpha, contains_digit, degenerate, b, contains_symbol, split_char
from estnltk.text import Text
from estnltk.layer.layer import Layer
from typing import MutableMapping

class NerEmptyFeatureTagger(Tagger):
    """"Extracts features provided by the morphological analyser pyvabamorf. """
    conf_param = ['settings']

    def __init__(self, settings, input_layers = ('words'), output_layer='ner_features',
                 output_attributes=("lem", "pos", "prop", "pref", "post", "case",
                                    "ending", "pun", "w", "w1", "shape", "shaped", "p1",
                                    "p2", "p3", "p4", "s1", "s2", "s3", "s4", "d2",
                                    "d4", "dndash", "dnslash", "dncomma", "dndot", "up", "iu", "au",
                                    "al", "ad", "ao", "aan", "cu", "cl", "ca", "cd",
                                    "cp", "cds", "cdt", "cs", "bdash", "adash",
                                    "bdot", "adot", "len", "fsnt", "lsnt", "gaz",
                                    "prew", "next", "iuoc", "pprop", "nprop", "pgaz",
                                    "ngaz", "F")):
        self.settings = settings
        self.output_layer = output_layer
        self.output_attributes = output_attributes
        self.input_layers = input_layers

    def _make_layer(self, text: Text, layers: MutableMapping[str, Layer], status: dict):
        layer = Layer(self.output_layer, ambiguous=True, attributes=self.output_attributes, text_object=text)
        for token in text.words:
            LEM = '_'.join(token.root_tokens[0]) + ('+' + token.ending[0] if token.ending[0] else '')
            if not LEM:
                LEM = token.text
            layer.add_annotation(token, lem=None, pos=None,
                                 prop=None,
                                 pref=None,
                                 post=None,
                                 case=None, ending=None,
                                 pun=None, w=None, w1=None, shape=None, shaped=None, p1=None,
                                 p2=None, p3=None, p4=None, s1=None, s2=None, s3=None, s4=None, d2=None, d4=None,
                                 dndash=None,
                                 dnslash=None, dncomma=None, dndot=None, up=None, iu=None, au=None,
                                 al=None, ad=None, ao=None, aan=None, cu=None, cl=None, ca=None,
                                 cd=None, cp=None, cds=None, cdt=None, cs=None, bdash=None, adash=None,
                                 bdot=None, adot=None, len=None, fsnt=None, lsnt=None, gaz=None, prew=None,
                                 next=None, iuoc=None, pprop=None, nprop=None, pgaz=None, ngaz=None, F=None)

        return layer
    
class NerLocalFeatureWithoutMorphTagger(Retagger):
    """Generates features for a token based on its character makeup."""
    conf_param = ['settings']

    def __init__(self, settings, output_layer='ner_features', output_attributes=(), input_layers=['ner_features']):
        self.settings = settings
        self.output_layer = output_layer
        self.output_attributes = output_attributes
        self.input_layers = input_layers

    def _change_layer(self, text: Text, layers: MutableMapping[str, Layer], status: dict):
        layer = layers[self.output_layer]
        layer.attributes += tuple(self.output_attributes)
        for token in text.ner_features:
            LEM = '_'.join(token.root_tokens[0]) + ('+' + token.ending[0] if token.ending[0] else '')
            if not LEM:
                LEM = token.text


            # Token.
            token.ner_features.w = token.text
            # Lowercased token.
            token.ner_features.w1 = token.text.lower()
            # Token shape.
            token.ner_features.shape = get_shape(token.text)
            # Token shape degenerated.
            token.ner_features.shaped = degenerate(get_shape(token.text))

            # Prefixes (length between one to four).
            token.ner_features.p1 = token.text[0] if len(token.text) >= 1 else None
            token.ner_features.p2 = token.text[:2] if len(token.text) >= 2 else None
            token.ner_features.p3 = token.text[:3] if len(token.text) >= 3 else None
            token.ner_features.p4 = token.text[:4] if len(token.text) >= 4 else None

            # Suffixes (length between one to four).
            token.ner_features.s1 = token.text[-1] if len(token.text) >= 1 else None
            token.ner_features.s2 = token.text[-2:] if len(token.text) >= 2 else None
            token.ner_features.s3 = token.text[-3:] if len(token.text) >= 3 else None
            token.ner_features.s4 = token.text[-4:] if len(token.text) >= 4 else None

            # Two digits
            token.ner_features.d2 = b(get_2d(token.text))
            # Four digits
            token.ner_features.d4 = b(get_4d(token.text))
            # Digits and '-'.
            token.ner_features.dndash = b(get_dand(token.text, '-'))
            # Digits and '/'.
            token.ner_features.dnslash = b(get_dand(token.text, '/'))
            # Digits and ','.
            token.ner_features.dncomma = b(get_dand(token.text, ','))
            # Digits and '.'.
            token.ner_features.dndot = b(get_dand(token.text, '.'))
            # A uppercase letter followed by '.'
            token.ner_features.up = b(get_capperiod(token.text))

            # An initial uppercase letter.
            token.ner_features.iu = b(token.text and token.text[0].isupper())
            # All uppercase letters.
            token.ner_features.au = b(token.text.isupper())
            # All lowercase letters.
            token.ner_features.al = b(token.text.islower())
            # All digit letters.
            token.ner_features.ad = b(token.text.isdigit())
            # All other (non-alphanumeric) letters.
            token.ner_features.ao = b(get_all_other(token.text))
            # Alphanumeric token.
            token.ner_features.aan = b(token.text.isalnum())

            # Contains an uppercase letter.
            token.ner_features.cu = b(contains_upper(token.text))
            # Contains a lowercase letter.
            token.ner_features.cl = b(contains_lower(token.text))
            # Contains a alphabet letter.
            token.ner_features.ca = b(contains_alpha(token.text))
            # Contains a digit.
            token.ner_features.cd = b(contains_digit(token.text))
            # Contains an apostrophe.
            token.ner_features.cp = b(token.text.find("'") > -1)
            # Contains a dash.
            token.ner_features.cds = b(token.text.find("-") > -1)
            # Contains a dot.
            token.ner_features.cdt = b(token.text.find(".") > -1)
            # Contains a symbol.
            token.ner_features.cs = b(contains_symbol(token.text))

            # Before, after dash
            token.ner_features.bdash = split_char(token.text, '-')[0]
            token.ner_features.adash = split_char(token.text, '-')[1]

            # Before, after dot
            token.ner_features.bdot = split_char(token.text, '.')[0]
            token.ner_features.adot = split_char(token.text, '.')[1]

            # Length
            token.ner_features.len = str(len(token.text))

class NerBasicMorphFeatureTagger(Retagger):
    """"Extracts features provided by the morphological analyser pyvabamorf. """
    conf_param = ['settings']

    def __init__(self, settings, input_layers = ('words','morph_analysis'), output_layer='ner_features',
                 output_attributes=("lem", "pos", "prop", "pref", "post", "case",
                                    "ending", "pun", "w", "w1", "shape", "shaped", "p1",
                                    "p2", "p3", "p4", "s1", "s2", "s3", "s4", "d2",
                                    "d4", "dndash", "dnslash", "dncomma", "dndot", "up", "iu", "au",
                                    "al", "ad", "ao", "aan", "cu", "cl", "ca", "cd",
                                    "cp", "cds", "cdt", "cs", "bdash", "adash",
                                    "bdot", "adot", "len", "fsnt", "lsnt", "gaz",
                                    "prew", "next", "iuoc", "pprop", "nprop", "pgaz",
                                    "ngaz", "F")):
        self.settings = settings
        self.output_layer = output_layer
        self.output_attributes = output_attributes
        self.input_layers = input_layers

    def _make_layer(self, text: Text, layers: MutableMapping[str, Layer], status: dict):
        layer = Layer(self.output_layer, ambiguous=True, attributes=self.output_attributes, text_object=text)
        for token in text.words:
            LEM = '_'.join(token.root_tokens[0]) + ('+' + token.ending[0] if token.ending[0] else '')
            if not LEM:
                LEM = token.text
            layer.add_annotation(token, lem=get_lemma(LEM), pos=get_pos(token.partofspeech),
                                 prop=b(is_prop(token.partofspeech)),
                                 pref=get_word_parts(token.root_tokens[0])[0],
                                 post=get_word_parts(token.root_tokens[0])[1],
                                 case=get_case(token.form[0]), ending=get_ending(token.ending),
                                 pun=b(get_pos(token.partofspeech)=="_Z_"), w=None, w1=None, shape=None, shaped=None, p1=None,
                                 p2=None, p3=None, p4=None, s1=None, s2=None, s3=None, s4=None, d2=None, d4=None,
                                 dndash=None,
                                 dnslash=None, dncomma=None, dndot=None, up=None, iu=None, au=None,
                                 al=None, ad=None, ao=None, aan=None, cu=None, cl=None, ca=None,
                                 cd=None, cp=None, cds=None, cdt=None, cs=None, bdash=None, adash=None,
                                 bdot=None, adot=None, len=None, fsnt=None, lsnt=None, gaz=None, prew=None,
                                 next=None, iuoc=None, pprop=None, nprop=None, pgaz=None, ngaz=None, F=None)

        return layer

class NerMorphNoLemmasFeatureTagger(Retagger):
    """"Extracts features provided by the morphological analyser pyvabamorf. """
    conf_param = ['settings']

    def __init__(self, settings, input_layers = ('words','morph_analysis'), output_layer='ner_features',
                 output_attributes=("lem", "pos", "prop", "pref", "post", "case",
                                    "ending", "pun", "w", "w1", "shape", "shaped", "p1",
                                    "p2", "p3", "p4", "s1", "s2", "s3", "s4", "d2",
                                    "d4", "dndash", "dnslash", "dncomma", "dndot", "up", "iu", "au",
                                    "al", "ad", "ao", "aan", "cu", "cl", "ca", "cd",
                                    "cp", "cds", "cdt", "cs", "bdash", "adash",
                                    "bdot", "adot", "len", "fsnt", "lsnt", "gaz",
                                    "prew", "next", "iuoc", "pprop", "nprop", "pgaz",
                                    "ngaz", "F")):
        self.settings = settings
        self.output_layer = output_layer
        self.output_attributes = output_attributes
        self.input_layers = input_layers

    def _make_layer(self, text: Text, layers: MutableMapping[str, Layer], status: dict):
        layer = Layer(self.output_layer, ambiguous=True, attributes=self.output_attributes, text_object=text)
        for token in text.words:
            LEM = None
            if not LEM:
                LEM = token.text
            layer.add_annotation(token, lem=get_lemma(LEM), pos=get_pos(token.partofspeech),
                                 prop=b(is_prop(token.partofspeech)),
                                 pref=None,
                                 post=None,
                                 case=get_case(token.form[0]), ending=None,
                                 pun=b(get_pos(token.partofspeech)=="_Z_"), w=None, w1=None, shape=None, shaped=None, p1=None,
                                 p2=None, p3=None, p4=None, s1=None, s2=None, s3=None, s4=None, d2=None, d4=None,
                                 dndash=None,
                                 dnslash=None, dncomma=None, dndot=None, up=None, iu=None, au=None,
                                 al=None, ad=None, ao=None, aan=None, cu=None, cl=None, ca=None,
                                 cd=None, cp=None, cds=None, cdt=None, cs=None, bdash=None, adash=None,
                                 bdot=None, adot=None, len=None, fsnt=None, lsnt=None, gaz=None, prew=None,
                                 next=None, iuoc=None, pprop=None, nprop=None, pgaz=None, ngaz=None, F=None)

        return layer