# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import os
from typing import List, Pattern, Union
from phonemizer.utils import list2str, str2list
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator


class TextTokenizer:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="|_|", syllable="-", phone="|"),
        preserve_punctuation=True,
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "remove-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        self.preserve_punctuation_marks = ",.?!;:'…"
        self.backend = EspeakBackend(
            language,
            punctuation_marks=self.preserve_punctuation_marks,
            preserve_punctuation=preserve_punctuation,
            with_stress=with_stress,
            tie=tie,
            language_switch=language_switch,
            words_mismatch=words_mismatch,
        )

        self.separator = separator

        if language == "en-us":
            from g2p.g2p.english import text_to_sequence
            self.text_to_sequence = text_to_sequence
        elif language == "cmn":
            from g2p.utils.front_utils import get_frontend_model
            self.text_to_sequence = get_frontend_model(
                frontend_conf="g2p/sources/g2p_chinese_model/config.json", frontend_type="g2p_cn"
            ).get_phoneme_ids
        elif language == "fr-fr":
            from g2p.g2p.french import text_to_sequence
            self.text_to_sequence = text_to_sequence
        elif language == "ko":
            from g2p.g2p.korean import text_to_sequence
            self.text_to_sequence = text_to_sequence
        elif language == "de":
            from g2p.g2p.german import text_to_sequence
            self.text_to_sequence = text_to_sequence
        elif language == "es":
            from g2p.g2p.spanish import text_to_sequence
            self.text_to_sequence = text_to_sequence

    # convert chinese punctuation to english punctuation
    def convert_chinese_punctuation(self, text: str) -> str:
        text = text.replace("，", ",")
        text = text.replace("。", ".")
        text = text.replace("！", "!")
        text = text.replace("？", "?")
        text = text.replace("；", ";")
        text = text.replace("：", ":")
        text = text.replace("、", ",")
        text = text.replace("‘", "'")
        text = text.replace("’", "'")
        text = text.replace("⋯", "…")
        text = text.replace("···", "…")
        text = text.replace("・・・", "…")
        text = text.replace("...", "…")
        return text

    def __call__(self, text, strip=True) -> List[str]:

        text_type = type(text)
        normalized_text = []
        for line in str2list(text):
            line = self.convert_chinese_punctuation(line.strip())
            line = re.sub(r"[^\w\s_,\.\?!;:\'…]", "", line)
            line = re.sub(r"\s*([,\.\?!;:\'…])\s*", r"\1", line)
            line = re.sub(r"\s+", " ", line)
            normalized_text.append(line)
        # print("Normalized test: ", normalized_text[0])
        phonemized = self.backend.phonemize(
            normalized_text, separator=self.separator, strip=strip, njobs=1
        )
        if text_type == str:
            phonemized = re.sub(r"([,\.\?!;:\'…])", r"|\1|", list2str(phonemized))
            phonemized = re.sub(r"\|+", "|", phonemized)
            phonemized = phonemized.rstrip("|")
        else:
            for i in range(len(phonemized)):
                phonemized[i] = re.sub(r"([,\.\?!;:\'…])", r"|\1|", phonemized[i])
                phonemized[i] = re.sub(r"\|+", "|", phonemized[i])
                phonemized[i] = phonemized[i].rstrip("|")
        return phonemized
