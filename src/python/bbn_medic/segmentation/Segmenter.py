import re
import spacy

from bbn_medic.utils.text_utils import spacy_sentencizer_custom_rules


class Segmenter:
    def __init__(self, type_of_segmentation=('sentences', 'spacy')):
        self.type_of_segmentation = type_of_segmentation
        if self.type_of_segmentation == ('sentences', 'spacy'):
            self.spacy_obj_en = spacy.load('en_core_web_sm')
            self.spacy_obj_en.add_pipe('custom_seg_on_newlines_and_numbered_list', before='parser')
        else:
            raise ValueError(f"Unknown segmentation type {self.type_of_segmentation}")

    def split_text_into_sentences(self, text):
        if self.type_of_segmentation == ('sentences', 'spacy'):
            if len(text) > 0:
                doc = self.spacy_obj_en(text)
                filtered_sentences = [s for s in doc.sents if str(s) != "*" and not re.search(r'^[0-9][0-9]*[.: ]\s*$', str(s))]
                sentences = [{"text": str(s), "start_char": s.start_char, "end_char": s.end_char} for s in filtered_sentences]
                return sentences
            else:
                return []
