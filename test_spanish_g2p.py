#!/usr/bin/env python3
# Test script for Spanish G2P module

import sys
import os
from g2p.g2p import spanish
from g2p.g2p_generation import chn_eng_g2p, get_segment

def test_spanish_phoneme_conversion():
    """Test direct phoneme conversion using the Spanish module."""
    test_cases = [
        "hola", 
        "buenos días",
        "¿cómo estás?",
        "Me gusta cantar",
        "España",
        "cerveza",
        "guitarra",
        "queso",
        "niño",
        "chihuahua",
        "arroz",
        "perro",
    ]
    
    print("===== Testing Direct Spanish G2P Conversion =====")
    for text in test_cases:
        phonemes = spanish.text_to_sequence(text)
        print(f"{text} → {phonemes}")
    print()

def test_multilingual_detection():
    """Test language detection and multilingual handling."""
    test_cases = [
        "Hola mundo",
        "你好, ¿cómo estás?",
        "Hello and hola",
        "Español y English mixed",
        "México es un país",
        "La niña está cantando",
    ]
    
    print("===== Testing Language Detection with Spanish =====")
    for text in test_cases:
        segments = get_segment(text)
        
        print(f"Text: {text}")
        print("Segments:")
        for seg, lang in segments:
            print(f"  '{seg}' → {lang}")
        
        # Full G2P conversion
        phonemes, tokens = chn_eng_g2p(text)
        print(f"Full phonemes: {phonemes}")
        print("---")
    print()

def test_stress_patterns():
    """Test Spanish stress pattern detection."""
    test_cases = [
        ("casa", "House - stress on first syllable"),
        ("camino", "Path - stress on second syllable"),
        ("árbol", "Tree - explicit stress on first syllable"),
        ("canción", "Song - explicit stress on second syllable"),
        ("español", "Spanish - stress on final syllable"),
        ("madrugada", "Dawn - stress on second-to-last syllable"),
        ("atención", "Attention - explicit stress on final syllable"),
    ]
    
    print("===== Testing Spanish Stress Patterns =====")
    for word, description in test_cases:
        phonemes = spanish.text_to_sequence(word)
        print(f"{word} ({description}) → {phonemes}")
    print()

def test_special_cases():
    """Test Spanish special pronunciation cases."""
    test_cases = [
        ("guerra", "War - silent u"),
        ("guitarra", "Guitar - silent u"),
        ("cigüeña", "Stork - ü indicates u is pronounced"),
        ("bilingüe", "Bilingual - ü indicates u is pronounced"),
        ("queso", "Cheese - qu = k sound"),
        ("cereza", "Cherry - c before e = s sound"),
        ("cinco", "Five - c before i = s sound"),
        ("gente", "People - g before e = h sound"),
        ("girasol", "Sunflower - g before i = h sound"),
    ]
    
    print("===== Testing Spanish Special Cases =====")
    for word, description in test_cases:
        phonemes = spanish.text_to_sequence(word)
        print(f"{word} ({description}) → {phonemes}")
    print()

if __name__ == "__main__":
    # Move to project directory for proper imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run the tests
    test_spanish_phoneme_conversion()
    test_multilingual_detection()
    test_stress_patterns()
    test_special_cases()
    
    print("All tests completed.")