# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from g2p.g2p.mandarin import chinese_to_ipa
from g2p.g2p.english import english_to_ipa
from g2p.g2p.french import french_to_ipa
from g2p.g2p.korean import korean_to_ipa
from g2p.g2p.german import german_to_ipa
from g2p.g2p.spanish import spanish_to_ipa


def cjekfd_cleaners(text, sentence, language, text_tokenizers):

    if language == "zh":
        text = text_tokenizers["zh"].text_to_sequence(text)
        return chinese_to_ipa(text, sentence, text_tokenizers["zh"])
    elif language == "en":
        text = text_tokenizers["en"].text_to_sequence(text)
        return english_to_ipa(text, text_tokenizers["en"])
    elif language == "fr":
        text = text_tokenizers["fr"].text_to_sequence(text)
        return french_to_ipa(text, text_tokenizers["fr"])
    elif language == "ko":
        text = text_tokenizers["ko"].text_to_sequence(text)
        return korean_to_ipa(text, text_tokenizers["ko"])
    elif language == "de":
        text = text_tokenizers["de"].text_to_sequence(text)
        return german_to_ipa(text, text_tokenizers["de"])
    elif language == "es":
        text = text_tokenizers["es"].text_to_sequence(text)
        return spanish_to_ipa(text, text_tokenizers["de"])
    else:
        raise Exception("Unknown language: %s" % language)
        return None
