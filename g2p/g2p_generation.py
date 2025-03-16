# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from g2p.g2p import PhonemeBpeTokenizer
from g2p.utils.g2p import phonemizer_g2p
import tqdm
from typing import List
import json
import os
import re


def ph_g2p(text, language):

    return phonemizer_g2p(text=text, language=language)


def g2p(text, sentence, language):

    return text_tokenizer.tokenize(text=text, sentence=sentence, language=language)


def is_chinese(char):
    if char >= "\u4e00" and char <= "\u9fa5":
        return True
    else:
        return False


def is_alphabet(char):
    if (char >= "\u0041" and char <= "\u005a") or (
        char >= "\u0061" and char <= "\u007a"
    ):
        return True
    else:
        return False


def is_spanish(char):
    """Check if a character is a Spanish-specific character."""
    return char in "áéíóúüñÁÉÍÓÚÜÑ"


def is_other(char):
    if not (is_chinese(char) or is_alphabet(char) or is_spanish(char)):
        return True
    else:
        return False


def get_segment(text: str) -> List[str]:
    # sentence --> [ch_part, en_part, ch_part, ...]
    segments = []
    types = []
    flag = 0
    temp_seg = ""
    temp_lang = ""

    # Determine the type of each character. type: blank, chinese, alphabet, number, unk and point.
    for i, ch in enumerate(text):
        if is_chinese(ch):
            types.append("zh")
        elif is_spanish(ch):
            types.append("es")  # Identify Spanish characters
        elif is_alphabet(ch):
            # Check for Spanish context - look at surrounding characters
            if i > 0 and types[i - 1] == "es":
                types.append("es")  # Continue Spanish sequence
            elif i < len(text) - 1 and is_spanish(text[i + 1]):
                types.append("es")  # Start Spanish sequence
            else:
                types.append("en")
        else:
            types.append("other")

    # Second pass to resolve language ambiguities in words
    # If a word contains any Spanish characters, mark the whole word as Spanish
    word_start = 0
    in_word = False
    for i in range(len(text)):
        if text[i].isalpha():
            if not in_word:
                word_start = i
                in_word = True
        else:
            if in_word:
                # Check if any character in this word was marked as Spanish
                if "es" in types[word_start:i]:
                    # Mark the entire word as Spanish
                    for j in range(word_start, i):
                        if types[j] in ["en", "es"]:  # Only change alphabetic characters
                            types[j] = "es"
                in_word = False

    # Handle final word if text ends with a word
    if in_word and "es" in types[word_start:]:
        for j in range(word_start, len(text)):
            if types[j] in ["en", "es"]:
                types[j] = "es"

    assert len(types) == len(text)

    for i in range(len(types)):
        # find the first char of the seg
        if flag == 0:
            temp_seg += text[i]
            temp_lang = types[i]
            flag = 1
        else:
            if temp_lang == "other":
                if types[i] == temp_lang:
                    temp_seg += text[i]
                else:
                    temp_seg += text[i]
                    temp_lang = types[i]
            else:
                if types[i] == temp_lang:
                    temp_seg += text[i]
                elif types[i] == "other":
                    temp_seg += text[i]
                else:
                    segments.append((temp_seg, temp_lang))
                    temp_seg = text[i]
                    temp_lang = types[i]
                    flag = 1

    segments.append((temp_seg, temp_lang))
    return segments


def chn_eng_g2p(text: str):
    # now support en, ch, and es (Spanish)
    segments = get_segment(text)
    all_phoneme = ""
    all_tokens = []

    for index in range(len(segments)):
        seg = segments[index]
        phoneme, token = g2p(seg[0], text, seg[1])
        all_phoneme += phoneme + "|"
        all_tokens += token

        if (seg[1] == "en" or seg[1] == "es") and index == len(segments) - 1 and all_phoneme[-2] == "_":
            all_phoneme = all_phoneme[:-2]
            all_tokens = all_tokens[:-1]
    return all_phoneme, all_tokens


text_tokenizer = PhonemeBpeTokenizer()
with open("./g2p/g2p/vocab.json", "r") as f:
    json_data = f.read()
data = json.loads(json_data)
vocab = data["vocab"]

if __name__ == '__main__':
    phone, token = chn_eng_g2p("你好，hello world")
    phone, token = chn_eng_g2p("你好，hello world, Bonjour, 테스트 해 보겠습니다, 五月雨緑")
    print(phone)
    print(token)

    # Test Spanish
    phone, token = chn_eng_g2p("Hola mundo, ¿cómo estás?")
    print("Spanish test:", phone)
    print("Spanish tokens:", token)

    # Test mixed language with Spanish
    phone, token = chn_eng_g2p("你好，hello world, ¡Hola amigos! こんにちは")
    print("Mixed with Spanish:", phone)
    print("Mixed tokens:", token)

    phone, token = text_tokenizer.tokenize("緑", "", "auto")
    print(phone)
    print(token)