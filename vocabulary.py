#!/projects/ltg/python3/bin/python3

# Simple tool to extract vocabulary from the corpus
# for training ELMo models

import sys

THRESHOLD = int(sys.argv[1])  # How many top frequent words you want to keep?

words = {}

for line in sys.stdin:
    res = line.strip().split()
    for word in res:
        if word not in words:
            words[word] = 0
        words[word] += 1

print('\n'.join(['<S>', '</S>', '<UNK>']))

print('Vocabulary:', len(words), file=sys.stderr)

a = sorted(words, key=words.get, reverse=True)[:THRESHOLD]
for w in a:
    print(w)
