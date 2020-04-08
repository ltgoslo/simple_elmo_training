#! /bin/env python3
# coding: utf-8

import sys
import numpy as np


# Simple tool to analyze your training corpus before training.
# It will tell how many UTF-8 code units an average character from your corpus requires.
# Depending on this, the standard ELMo maximum word length of 50 code units can crop
# more or less of your precious words.
# If you feel that the ratio of the cropped word types or tokens is too high, you should increase
# the `max_characters_per_token` parameter in `train_elmo.py`

def convert_word_to_char_ids(w, max_length=50, bow_char=258, eow_char=259, pad_char=260,
                             verbose=False):
    # Default max word length in UTF-8 code units in ELMo is 50
    token_length = len(w)
    cropped = False
    word_encoded = w.encode('utf-8', 'ignore')
    original_length = len(word_encoded)
    bytes_per_char = original_length / token_length
    word_encoded = word_encoded[:(max_length - 2)]
    length = len(word_encoded)
    if length < original_length:
        cropped = True

    if verbose:
        code = np.zeros([max_length], dtype=np.int32)
        code[:] = pad_char
        code[0] = bow_char
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        code[len(word_encoded) + 1] = eow_char
        print(w, code, file=sys.stderr)
    return cropped, bytes_per_char


if __name__ == "__main__":
    dictionary = set()
    cropped_types = set()
    cropped_tokens = 0
    code_units_nr = 0
    token_counter = 0

    for line in sys.stdin:  # We read your corpus from the standard input
        sentence = line.strip().split()
        dictionary.update(sentence)
        token_counter += len(sentence)
        for word in sentence:
            is_cropped, bpc = convert_word_to_char_ids(word)
            code_units_nr += bpc
            if is_cropped:
                cropped_types.add(word)
                cropped_tokens += 1

    average_code_units_per_char = code_units_nr / token_counter
    cropped_types_ratio = len(cropped_types) / len(dictionary)
    cropped_tokens_ratio = cropped_tokens / token_counter

    print('Word types in your vocabulary:', len(dictionary))
    print('Word tokens in your corpus:', token_counter)

    print('Average code units (bytes) per character in your corpus: %0.3f'
          % average_code_units_per_char)
    print('Ratio of word types that will be cropped: %0.3f%%' % (cropped_types_ratio * 100))
    print('Ratio of word tokens that will be cropped: %0.3f%%' % (cropped_tokens_ratio * 100))
