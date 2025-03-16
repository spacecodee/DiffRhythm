# Copyright (c) 2025 [Your Name]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Spanish grapheme-to-phoneme conversion module for DiffRhythm.

This module converts Spanish text to phonetic representations compatible with
DiffRhythm's pronunciation system. It handles Spanish-specific features such as:
- Diacritical marks (á, é, í, ó, ú, ü)
- Special characters (ñ)
- Context-dependent pronunciation rules
- Stress patterns
- Regional variants (primarily Latin American Spanish)
"""

import re
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Spanish vowels
_VOWELS = "aeiouáéíóúü"

# Spanish consonants
_CONSONANTS = "bcdfghjklmnñpqrstvwxyz"

# Stressed vowels
_STRESSED_VOWELS = "áéíóú"

# Spanish phoneme dictionary - mapping from Spanish characters to phonetic representation
# Based on Spanish phonology adapted to DiffRhythm's phoneme set
_SPANISH_PHONEMES = {
    'a': 'AA',
    'e': 'EH',
    'i': 'IY',
    'o': 'OW',
    'u': 'UW',
    'á': 'AA1',
    'é': 'EH1',
    'í': 'IY1',
    'ó': 'OW1',
    'ú': 'UW1',
    'ü': 'UW',  # Dieresis in Spanish (only used in güe, güi)
    'b': 'B',   # Initial position
    'v': 'B',   # In Spanish, 'b' and 'v' have the same sound
    'c': 'K',   # Before a, o, u (will be handled for ce, ci)
    'd': 'D',
    'f': 'F',
    'g': 'G',   # Before a, o, u (will be handled for ge, gi)
    'h': '',    # Silent in Spanish
    'j': 'HH',  # Spanish 'j' is similar to English 'h' but stronger
    'k': 'K',
    'l': 'L',
    'm': 'M',
    'n': 'N',
    'ñ': 'NY',  # Similar to "ny" in "canyon"
    'p': 'P',
    'q': 'K',   # Always followed by 'u' which is silent in 'que', 'qui'
    'r': 'R',   # Single 'r'
    's': 'S',
    't': 'T',
    'w': 'W',   # Mostly in foreign words
    'x': 'K S', # Generally 'ks' sound, but varies
    'y': 'Y',   # As a consonant
    'z': 'S',   # In Latin American Spanish (would be 'TH' in Spain)
}

# Phonemes for special digraphs (two-letter combinations)
_DIGRAPHS = {
    'ch': 'CH',  # As in "chico"
    'll': 'Y',   # As in "llamar" (varies by region, using Y for Latin American)
    'qu': 'K',   # As in "queso"
    'rr': 'RR',  # Rolled 'r' as in "perro"
}

# Special combinations with context
_SPECIAL_CASES = {
    'ce': 'S EH',  # 'c' before 'e' is pronounced 's' in Latin America, 'th' in Spain
    'ci': 'S IY',  # 'c' before 'i' is pronounced 's' in Latin America, 'th' in Spain
    'ge': 'HH EH', # 'g' before 'e' is pronounced like 'h'
    'gi': 'HH IY', # 'g' before 'i' is pronounced like 'h'
    'gue': 'G EH', # 'u' is silent, 'g' is hard
    'gui': 'G IY', # 'u' is silent, 'g' is hard
    'güe': 'G W EH', # 'ü' indicates 'u' is pronounced
    'güi': 'G W IY', # 'ü' indicates 'u' is pronounced
    'que': 'K EH', # 'u' is silent
    'qui': 'K IY', # 'u' is silent
    'ya': 'Y AA',
    'ye': 'Y EH',
    'yi': 'Y IY',
    'yo': 'Y OW',
    'yu': 'Y UW',
}

# Words that need special handling due to irregular pronunciation
_EXCEPTION_WORDS = {
    'mexico': 'M EH1 HH IY K OW',
    'méxico': 'M EH1 HH IY K OW',
    'texas': 'T EH1 HH AA S',
    'puerto rico': 'P W EH1 R T OW RR IY1 K OW',
    'barcelona': 'B AA R S EH L OW1 N AA',
    'valencia': 'B AA L EH1 N S IY AA',
    'guerrero': 'G EH RR EH1 R OW',
    'uruguay': 'UW R UW G W AA1 Y',
    'paraguay': 'P AA R AA G W AA1 Y',
    'buenos aires': 'B W EH1 N OW S AA1 IY R EH S',
    'havana': 'AA B AA1 N AA',
    'habana': 'AA B AA1 N AA',
    'venezuela': 'B EH N EH S W EH1 L AA',
    'chile': 'CH IY1 L EH',
    'argentina': 'AA R HH EH N T IY1 N AA',
    'andalucía': 'AA N D AA L UW S IY1 AA',
    'sevilla': 'S EH B IY1 Y AA',
    'españa': 'EH S P AA1 NY AA',
    'paella': 'P AA EH1 Y AA',
    'guillermo': 'G IY Y EH1 R M OW',
}

# List of (regular expression, replacement) pairs for Spanish abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\b" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("sr", "señor"),
        ("sra", "señora"),
        ("srta", "señorita"),
        ("dr", "doctor"),
        ("dra", "doctora"),
        ("ing", "ingeniero"),
        ("lic", "licenciado"),
        ("prof", "profesor"),
        ("av", "avenida"),
        ("c", "calle"),
        ("tel", "teléfono"),
        ("pág", "página"),
        ("etc", "etcétera"),
        ("hospital", "hospital"),
        ("esq", "esquina"),
    ]
]

# Special mapping for fixing phoneme sequences
_special_map = [
    ("RR", "RR"),  # Keep rolled R as is
    ("NY", "NY"),  # Keep ñ pronunciation as is
    ("AA1", "AA1"), # Keep primary stress
    ("EH1", "EH1"),
    ("IY1", "IY1"),
    ("OW1", "OW1"),
    ("UW1", "UW1"),
    # Fix common Spanish phoneme combinations
    ("S|IY|OW1|N", "S|IY|OW1|N"),  # sión
    ("S|IY|AA1", "S|IY|AA1"),      # sía
    ("B|L", "B L"),               # Improve bl cluster
    ("P|L", "P L"),               # Improve pl cluster
    ("G|L", "G L"),               # Improve gl cluster
    ("K|L", "K L"),               # Improve cl cluster
]

# Regular expressions for finding syllables and word patterns
# This is a simplified version - Spanish syllabification has complex rules
_CONSONANT_CLUSTER_RE = re.compile(r'[bcdfghjklmnñpqrstvwxyz]+')
_VOWEL_CLUSTER_RE = re.compile(r'[aeiouáéíóúü]+')

# Number handling
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_percent_number_re = re.compile(r"([0-9\.\,]*[0-9]+%)")
_euros_re = re.compile(r"([0-9\,]*[0-9]+)€")
_fraction_re = re.compile(r"([0-9]+)/([0-9]+)")
_ordinal_re = re.compile(r"([0-9]+)(º|ª)")
_number_re = re.compile(r"[0-9]+")

def _expand_abbreviations(text):
    """Expand Spanish abbreviations."""
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def _remove_commas(m):
    """Remove commas from numbers."""
    return m.group(1).replace(",", "")

def _expand_decimal_point(m):
    """Convert decimal points to spoken form."""
    return m.group(1).replace(".", " coma ")

def _expand_percent(m):
    """Convert percentages to spoken form."""
    return m.group(1).replace("%", " por ciento")

def _expand_euros(m):
    """Convert euro amounts to spoken form."""
    amount = m.group(1).replace(",", "")
    if amount == "1":
        return amount + " euro"
    else:
        return amount + " euros"

def _expand_fraction(m):
    """Convert fractions to words."""
    numerator = int(m.group(1))
    denominator = int(m.group(2))
    
    # Common fractions
    if numerator == 1 and denominator == 2:
        return "un medio"
    if numerator == 1 and denominator == 3:
        return "un tercio"
    if numerator == 1 and denominator == 4:
        return "un cuarto"
    
    # General case
    if numerator == 1:
        if denominator == 1:
            return "un entero"
        else:
            return f"un {_get_ordinal_suffix(denominator)}"
    else:
        return f"{numerator} {_get_ordinal_suffix(denominator)}s"

def _get_ordinal_suffix(n):
    """Get Spanish ordinal suffix."""
    if n == 1:
        return "entero"
    elif n == 2:
        return "medio"
    elif n == 3:
        return "tercio"
    elif n == 4:
        return "cuarto"
    elif n == 5:
        return "quinto"
    elif n == 6:
        return "sexto"
    elif n == 7:
        return "séptimo"
    elif n == 8:
        return "octavo"
    elif n == 9:
        return "noveno"
    elif n == 10:
        return "décimo"
    # More complex cases
    else:
        return f"avo"  # Simplified

def _expand_ordinal(m):
    """Convert ordinals to words."""
    number = int(m.group(1))
    gender = "a" if m.group(2) == "ª" else "o"
    
    if number == 1:
        return "primer" + gender
    elif number == 2:
        return "segund" + gender
    elif number == 3:
        return "tercer" + gender
    elif number == 4:
        return "cuart" + gender
    elif number == 5:
        return "quint" + gender
    elif number == 6:
        return "sext" + gender
    elif number == 7:
        return "séptim" + gender
    elif number == 8:
        return "octav" + gender
    elif number == 9:
        return "noven" + gender
    elif number == 10:
        return "décim" + gender
    else:
        # Simplified handling for higher numbers
        return f"{_expand_number(number)} "

def _expand_number(m):
    """Convert numbers to words."""
    num = int(m.group(0))
    
    # Special cases
    if num == 0:
        return "cero"
    elif num == 1:
        return "uno"
    elif num == 2:
        return "dos"
    # ... and so on for other numbers
    
    # Handle tens
    if 10 <= num < 30:
        if num == 10:
            return "diez"
        elif num == 11:
            return "once"
        elif num == 12:
            return "doce"
        elif num == 13:
            return "trece"
        elif num == 14:
            return "catorce"
        elif num == 15:
            return "quince"
        elif num == 16:
            return "dieciséis"
        elif num == 17:
            return "diecisiete"
        elif num == 18:
            return "dieciocho"
        elif num == 19:
            return "diecinueve"
        elif num == 20:
            return "veinte"
        elif num == 21:
            return "veintiuno"
        elif num == 22:
            return "veintidós"
        elif num == 23:
            return "veintitrés"
        elif num == 24:
            return "veinticuatro"
        elif num == 25:
            return "veinticinco"
        elif num == 26:
            return "veintiséis"
        elif num == 27:
            return "veintisiete"
        elif num == 28:
            return "veintiocho"
        elif num == 29:
            return "veintinueve"
    
    # Basic implementation for other numbers (simplified)
    # In a complete implementation, we'd handle higher numbers
    return str(num)

def normalize_numbers(text):
    """Normalize numbers in Spanish text."""
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_euros_re, _expand_euros, text)
    text = re.sub(_fraction_re, _expand_fraction, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_percent_number_re, _expand_percent, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text

def _preprocess_text(text: str) -> str:
    """
    Preprocess Spanish text for G2P conversion.
    
    Args:
        text: The Spanish text to preprocess.
        
    Returns:
        Preprocessed text ready for phonetic conversion.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-Spanish punctuation but keep spaces
    text = re.sub(r'[^\w\sáéíóúüñ]', '', text)
    
    return text

def _count_syllables(word: str) -> int:
    """
    Count the number of syllables in a Spanish word.
    
    Args:
        word: A Spanish word.
        
    Returns:
        The number of syllables in the word.
    """
    # Remove silent h 
    word = word.replace('h', '')
    
    # Count vowel clusters (each vowel cluster is a syllable)
    vowel_clusters = _VOWEL_CLUSTER_RE.findall(word)
    
    # Handle diphthongs (combinations of weak and strong vowels)
    # This is simplified - Spanish has complex diphthong rules
    syllable_count = len(vowel_clusters)
    
    # Adjust for diphthongs (when weak vowels 'i', 'u' combine with strong vowels)
    for cluster in vowel_clusters:
        if len(cluster) > 1:
            weak_vowels = sum(1 for v in cluster if v in 'iuíú')
            strong_vowels = sum(1 for v in cluster if v in 'aeoáéó')
            # If there are both weak and strong vowels, they form a single syllable
            if weak_vowels > 0 and strong_vowels > 0:
                syllable_count -= (len(cluster) - 1)
    
    return max(1, syllable_count)  # Ensure at least one syllable

def _find_stress_syllable(word: str) -> int:
    """
    Determine which syllable receives stress in a Spanish word.
    
    Args:
        word: A Spanish word.
        
    Returns:
        The index of the stressed syllable (0-based, from the end of the word).
    """
    # Words with written accent mark
    if any(c in _STRESSED_VOWELS for c in word):
        # Find position of stressed vowel
        for i, c in enumerate(reversed(word)):
            if c in _STRESSED_VOWELS:
                # Return approximate syllable position from end
                return min(i // 2, _count_syllables(word) - 1)
    
    # Words ending in vowel, n, or s
    elif word[-1] in 'aeiouáéíóúns':
        return 1  # Stress on penultimate syllable
    
    # All other words
    else:
        return 0  # Stress on final syllable

def _handle_special_cases(word: str) -> str:
    """
    Handle special pronunciation cases in Spanish.
    
    Args:
        word: A Spanish word.
        
    Returns:
        The word with special cases marked for phoneme conversion.
    """
    # Check if word is in exception list
    if word in _EXCEPTION_WORDS:
        return _EXCEPTION_WORDS[word]
    
    # Check common digraphs and mark them
    for digraph, _ in _DIGRAPHS.items():
        word = word.replace(digraph, f"_{digraph}_")
    
    # Replace special character combinations
    for combo, _ in _SPECIAL_CASES.items():
        word = word.replace(combo, f"_{combo}_")
    
    return word

def _apply_spanish_stress_rules(word: str, phonemes: List[str]) -> List[str]:
    """
    Apply Spanish stress rules to add stress markers to vowels.
    
    Args:
        word: The original Spanish word.
        phonemes: The list of phonemes.
        
    Returns:
        The phonemes with stress markers applied.
    """
    # If we have explicit stress markers already, don't modify
    if any('1' in p for p in phonemes):
        return phonemes
        
    syllable_count = _count_syllables(word)
    stress_idx = _find_stress_syllable(word)
    
    if syllable_count <= 1:
        # Single syllable words typically don't need stress marks
        return phonemes
    
    # Find vowel phonemes
    vowel_phonemes = [i for i, p in enumerate(phonemes) if p in ('AA', 'EH', 'IY', 'OW', 'UW')]
    
    # If we have enough vowel phonemes and can determine the stress position
    if vowel_phonemes and stress_idx < len(vowel_phonemes):
        # Apply stress to the appropriate vowel
        stress_position = vowel_phonemes[-(stress_idx + 1)]
        phonemes[stress_position] = phonemes[stress_position] + '1'
    
    return phonemes

def _word_to_phonemes(word: str) -> List[str]:
    """
    Convert a single Spanish word to phonemes.
    
    Args:
        word: A Spanish word.
        
    Returns:
        List of phonemes for the word.
    """
    # Check exception dictionary first
    if word in _EXCEPTION_WORDS:
        return _EXCEPTION_WORDS[word].split()
    
    # Handle special cases
    processed_word = _handle_special_cases(word)
    
    # Convert characters to phonemes
    phonemes = []
    i = 0
    while i < len(processed_word):
        # Skip underscores used for marking
        if processed_word[i] == '_':
            i += 1
            continue
            
        # Check for marked special cases
        if i < len(processed_word) - 1 and processed_word[i] == '_':
            # Find the end of the marked section
            end_idx = processed_word.find('_', i + 1)
            if end_idx > i:
                special_case = processed_word[i+1:end_idx]
                if special_case in _DIGRAPHS:
                    phonemes.append(_DIGRAPHS[special_case])
                elif special_case in _SPECIAL_CASES:
                    phonemes.extend(_SPECIAL_CASES[special_case].split())
                i = end_idx + 1
                continue
        
        # Check for digraphs
        if i < len(processed_word) - 1:
            digraph = processed_word[i:i+2]
            if digraph in _DIGRAPHS:
                phonemes.append(_DIGRAPHS[digraph])
                i += 2
                continue
        
        # Check for special combinations
        for combo_len in range(3, 0, -1):
            if i <= len(processed_word) - combo_len:
                combo = processed_word[i:i+combo_len]
                if combo in _SPECIAL_CASES:
                    phonemes.extend(_SPECIAL_CASES[combo].split())
                    i += combo_len
                    break
        else:
            # If no matches found, process single character
            if i < len(processed_word) and processed_word[i] in _SPANISH_PHONEMES:
                phoneme = _SPANISH_PHONEMES[processed_word[i]]
                if phoneme:  # Skip empty phonemes (like 'h')
                    phonemes.append(phoneme)
            i += 1
    
    # Apply stress rules
    phonemes = _apply_spanish_stress_rules(word, phonemes)
    
    return phonemes

def spanish_text_to_phonemes(text: str) -> str:
    """
    Convert Spanish text to a sequence of phonemes.
    
    Args:
        text: String of Spanish text.
        
    Returns:
        String of space-separated phonemes.
    """
    text = _preprocess_text(text)
    
    # Check if entire phrase is in exception dictionary
    if text in _EXCEPTION_WORDS:
        return _EXCEPTION_WORDS[text]
    
    # Process word by word
    words = text.split()
    result_phonemes = []
    
    for word in words:
        word_phonemes = _word_to_phonemes(word)
        result_phonemes.extend(word_phonemes)
    
    return ' '.join(result_phonemes)

def _spanish_to_ipa(text):
    """
    Convert Spanish text to phonetic form, handling numbers and abbreviations.
    Similar to the _english_to_ipa function, following the same pattern.
    
    Args:
        text: Spanish text input
        
    Returns:
        Processed text with normalized numbers and expanded abbreviations
    """
    text = text.lower()
    text = _expand_abbreviations(text)
    text = normalize_numbers(text)
    return text

def special_map(text):
    """
    Apply special mapping rules to fix phoneme patterns.
    Similar to the special_map function in english.py
    
    Args:
        text: Phoneme text to process
        
    Returns:
        Processed phoneme text with fixes applied
    """
    for regex, replacement in _special_map:
        regex = regex.replace("|", "\|")
        while re.search(r"(^|[_|]){}([_|]|$)".format(regex), text):
            text = re.sub(
                r"(^|[_|]){}([_|]|$)".format(regex), r"\1{}\2".format(replacement), text
            )
    return text

def spanish_to_ipa(text, text_tokenizer):
    """
    Main function to process Spanish text through the full pipeline.
    Follows the same pattern as english_to_ipa in english.py
    
    Args:
        text: Spanish text to process
        text_tokenizer: Tokenizer object to use for processing
        
    Returns:
        IPA representation of the Spanish text
    """
    if isinstance(text, str):
        text = _spanish_to_ipa(text)
    else:
        text = [_spanish_to_ipa(t) for t in text]
        
    phonemes = text_tokenizer(text)
    
    # Add word boundary if needed
    if isinstance(phonemes, str) and phonemes[-1] in "pbtdkgmnñfszxrRlw":
        phonemes += "|_"
        
    if isinstance(text, str):
        return special_map(phonemes)
    else:
        result_ph = []
        for phone in phonemes:
            result_ph.append(special_map(phone))
        return result_ph

def text_to_sequence(text: str) -> str:
    """
    Convert Spanish text to a phoneme sequence for DiffRhythm.
    The main entry point for the Spanish G2P module.
    
    Args:
        text: String of Spanish text.
        
    Returns:
        String of space-separated phonemes.
    """
    return spanish_text_to_phonemes(text)

if __name__ == "__main__":
    # Test the Spanish G2P module
    test_phrases = [
        "hola mundo",
        "buenos días",
        "¿cómo estás?",
        "españa",
        "México",
        "guitarra",
        "quesadilla",
        "cerveza",
        "paella valenciana",
        "churrería",
        "lluvia en sevilla",
        "el quijote de la mancha"
    ]
    
    for phrase in test_phrases:
        phonemes = text_to_sequence(phrase)
        print(f"{phrase} → {phonemes}")