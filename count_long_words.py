#! /bin/env python3
# coding: utf-8

import sys
import numpy as np


def convert_word_to_char_ids(word, max_length=50, bow_char=258, eow_char=259, pad_char=260, verbose=False):
    # Default max word length in UTF-8 code units in ELMo is 50
    token_length = len(word)

    cropped = False

    word_encoded = word.encode('utf-8', 'ignore')
    original_length = len(word_encoded)
    bytes_per_char = original_length / token_length

    word_encoded = word_encoded[:(max_length - 2)]
    length = len(word_encoded)
    if length < original_length:
        cropped=True

    if verbose:
        code = np.zeros([max_length], dtype=np.int32)
        code[:] = pad_char

        code[0] = bow_char
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        code[len(word_encoded) + 1] = eow_char
        print(word, code, file=sys.stderr)
    return cropped, bytes_per_char

dictionary = set()
cropped_types = set()
cropped_tokens = 0
code_units_nr = 0
token_counter = 0

for line in sys.stdin:
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

print('Average code units (bytes) per character in your corpus: %0.3f' % average_code_units_per_char)
print('Ratio of word types that will be cropped: %0.3f%%' % (cropped_types_ratio * 100))
print('Ratio of word tokens that will be cropped: %0.3f%%' % (cropped_tokens_ratio * 100))



