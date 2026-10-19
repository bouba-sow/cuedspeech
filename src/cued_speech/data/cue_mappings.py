"""Cued Speech Mapping Data.

This module contains the mappings between phonemes and cued speech visual cues:
- Consonants to hand shapes (1-8)
- Vowels to hand positions relative to the face
"""

from typing import Dict, Union

# Define the mapping of consonants to hand shapes
CONSONANT_TO_HANDSHAPE: Dict[str, int] = {
    "p": 1, "t": 5, "k": 2, "b": 4, "d": 1, "g": 7, "m": 5, "n": 4,
    "l": 6, "r": 3, "s": 3, "f": 5, "v": 2, "z": 2, "ʃ": 6, "ʒ": 1,
    "ɡ": 7, "ʁ": 3, "j": 8, "w": 6, "ŋ": 8, "ɥ": 4, "ʀ": 3, "c": 2,
    "ɲ": 6, "ɟ": 7, "ʎ": 8
}

# Define vowel positions relative to the face
# Special codes: -1 = right side of mouth, -2 = throat/below chin
# Numbers correspond to MediaPipe face landmark indices
VOWEL_POSITIONS: Dict[str, Union[int, str]] = {
    # Position 1: /a/, /o/, /œ/, /ə/ - Right side of the mouth
    "a": -1,
    "o": -1,
    "œ": -1,
    "ə": -1,
    
    # Position 2: /ɛ̃/, /ø/ - Cheek
    "ɛ̃": 50,
    "ø": 50,
    
    # Position 3: /i/, /ɔ̃/, /ɑ̃/ - Corner of the mouth
    "i": 57,
    "ɔ̃": 57,
    "ɑ̃": 57,
    
    # Position 4: /u/, /ɛ/, /ɔ/ - Chin (below the mouth) - centered, no change needed
    "u": 175,
    "ɛ": 175,
    "ɔ": 175,
    
    # Position 5: /œ̃/, /y/, /e/ - Throat (below the chin)
    "œ̃": -2,
    "y": -2,
    "e": -2,
}

# IPA to LIAPHON mapping for French phoneme processing
IPA_TO_LIAPHON: Dict[str, str] = {
    "a": "a", "ə": "x", "ɛ": "e^", "œ": "x^", "i": "i", "y": "y", "e": "e",
    "u": "u", "ɔ": "o", "o": "o^", "ɑ̃": "a~", "ɛ̃": "e~", "ɔ̃": "o~", "œ̃": "x~",
    " ": "_",
    "b": "b", "c": "k", "d": "d", "f": "f", "ɡ": "g", "j": "j", "k": "k", "l": "l",
    "m": "m", "n": "n", "p": "p", "s": "s", "t": "t", "v": "v", "w": "w", "z": "z",
    "ɥ": "h", "ʁ": "r", "ʃ": "s^", "ʒ": "z^", "ɲ": "gn", "ŋ": "ng",
}

LIAPHON_TO_IPA: Dict[str, str] = {v: k for k, v in IPA_TO_LIAPHON.items()}

# Phoneme sets for categorization
VOWELS = {"a", "e", "ɛ", "i", "o", "ɔ", "u", "ø", "œ", "ə", "y", "ɑ̃", "ɛ̃", "ɔ̃", "œ̃"}
CONSONANTS = {"p", "t", "k", "b", "d", "g", "m", "n", "l", "r", "s", "f", "v", "z", 
              "ʃ", "ʒ", "ɡ", "ʁ", "j", "w", "ŋ", "ɥ", "ʀ", "c", "ɲ"}

def map_syllable_to_cue(syllable: str) -> tuple[int, Union[int, str]]:
    """
    Map a syllable to its corresponding hand shape and hand position.
    
    Args:
        syllable: Syllable in IPA format (e.g., "si", "ne", "ma")
        
    Returns:
        Tuple of (hand_shape, hand_position)
    """
    # Handle complex syllables by breaking them down
    syllable = syllable.strip()
    
    # First, try to find consonant and vowel patterns
    consonant = None
    vowel = None
    
    # Look for nasal vowels first (they are 2 characters)
    nasal_vowels = {"ɔ̃", "ɛ̃", "ɑ̃", "œ̃"}
    for nasal in nasal_vowels:
        if nasal in syllable:
            vowel = nasal
            # Remove the vowel to find consonant
            consonant_part = syllable.replace(nasal, "")
            if consonant_part and consonant_part[-1] in CONSONANTS:
                consonant = consonant_part[-1]
            break
    
    # If no nasal vowel found, look for regular vowels
    if vowel is None:
        for v in VOWELS:
            if v in syllable:
                vowel = v
                # Remove the vowel to find consonant
                consonant_part = syllable.replace(v, "")
                if consonant_part and consonant_part[-1] in CONSONANTS:
                    consonant = consonant_part[-1]
                break
    
    # If we found both consonant and vowel
    if consonant and vowel:
        hand_shape = CONSONANT_TO_HANDSHAPE.get(consonant, 8)
        hand_position = VOWEL_POSITIONS.get(vowel, -1)
        return hand_shape, hand_position
    
    # If only consonant found
    elif consonant:
        hand_shape = CONSONANT_TO_HANDSHAPE.get(consonant, 8)
        hand_position = VOWEL_POSITIONS["a"]  # Default to position 1
        return hand_shape, hand_position
    
    # If only vowel found
    elif vowel:
        hand_shape = 5  # Default hand shape for vowels
        hand_position = VOWEL_POSITIONS.get(vowel, -1)
        return hand_shape, hand_position
    
    # Fallback: try character-by-character analysis
    for char in syllable:
        if char in CONSONANTS:
            consonant = char
            break
    
    for char in syllable:
        if char in VOWELS:
            vowel = char
            break
    
    if consonant or vowel:
        hand_shape = CONSONANT_TO_HANDSHAPE.get(consonant, 5) if consonant else 5
        hand_position = VOWEL_POSITIONS.get(vowel, -1) if vowel else -1
        return hand_shape, hand_position
    
    # Ultimate fallback
    print(f"Warning: Mapping syllable '{syllable}' to default Hand Shape 8 and Position 1")
    return 8, -1


def choose_reference_finger(hand_shape: int) -> int:
    """
    Choose the reference finger landmark index based on hand shape.
    
    Args:
        hand_shape: Hand shape number (1-8)
        
    Returns:
        MediaPipe hand landmark index for the reference finger
    """
    if hand_shape in {1, 6}:
        return 8  # Index finger tip
    else:
        return 12  # Middle finger tip


# Hand landmark connections for rendering
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]

def text_to_ipa(text: str) -> str:
    """
    Convert French text to IPA (International Phonetic Alphabet).
    This is a simplified version - for production use, consider using a proper TTS system.
    
    Args:
        text: French text to convert
        
    Returns:
        IPA representation of the text
    """
    # Simple French to IPA mapping (basic version)
    french_to_ipa = {
        'a': 'a', 'e': 'ə', 'i': 'i', 'o': 'o', 'u': 'y',
        'é': 'e', 'è': 'ɛ', 'à': 'a', 'ù': 'y', 'ô': 'o',
        'â': 'ɑ', 'ê': 'ɛ', 'î': 'i', 'û': 'y', 'ô': 'o',
        'an': 'ɑ̃', 'en': 'ɑ̃', 'in': 'ɛ̃', 'on': 'ɔ̃', 'un': 'œ̃',
        'am': 'ɑ̃', 'em': 'ɑ̃', 'im': 'ɛ̃', 'om': 'ɔ̃', 'um': 'œ̃',
        'b': 'b', 'c': 'k', 'd': 'd', 'f': 'f', 'g': 'ɡ',
        'h': '', 'j': 'ʒ', 'k': 'k', 'l': 'l', 'm': 'm',
        'n': 'n', 'p': 'p', 'q': 'k', 'r': 'ʁ', 's': 's',
        't': 't', 'v': 'v', 'w': 'w', 'x': 'ks', 'y': 'j', 'z': 'z',
        'ç': 's', 'ñ': 'ɲ', 'gn': 'ɲ', 'ch': 'ʃ', 'ph': 'f',
        'th': 't', 'qu': 'k', 'gu': 'ɡ', 'ge': 'ʒ'
    }
    
    # Convert to lowercase and normalize
    text = text.lower().strip()
    
    # Simple conversion (this is a basic implementation)
    # For better results, use a proper French TTS system
    ipa_text = ""
    i = 0
    while i < len(text):
        # Try 2-character combinations first
        if i < len(text) - 1:
            two_char = text[i:i+2]
            if two_char in french_to_ipa:
                ipa_text += french_to_ipa[two_char]
                i += 2
                continue
        
        # Try single character
        if text[i] in french_to_ipa:
            ipa_text += french_to_ipa[text[i]]
        elif text[i].isalpha():
            # Keep unknown letters as-is
            ipa_text += text[i]
        elif text[i] == ' ':
            ipa_text += ' '
        
        i += 1
    
    return ipa_text.strip() 