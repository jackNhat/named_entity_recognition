import numpy as np
from keras.preprocessing.sequence import pad_sequences


def readfile(filename):
    sentences = []
    sentence = []
    with open(filename) as f:
        content = f.read().split("\n")
    for line in content:
        if len(line) > 0:
            splits = line.split(' ')
            sentence.append([splits[0].strip(), splits[-1].strip()])
    if len(sentence) > 0:
        sentences.append(sentence)
        sentence = []
    return sentences


# define casing s.t. NN can use case information to learn patterns
def get_casing(word, case_lookup):
    casing = 'other'

    num_digits = 0
    for char in word:
        if char.isdigit():
            num_digits += 1

    digit_fraction = num_digits / float(len(word))

    if word.isdigit():  # Is a digit
        casing = 'numeric'
    elif digit_fraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():  # All lower case
        casing = 'allLower'
    elif word.isupper():  # All upper case
        casing = 'allUpper'
    elif word[0].isupper():  # is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif num_digits > 0:
        casing = 'contains_digit'

    return case_lookup[casing]


# return batches ordered by words in sentence
def create_equal_batches(data):
    n_batches = 100
    batch_size = len(data) // n_batches
    num_words = [batch_size * (i + 1) for i in range(0, n_batches)]

    batches = []
    batch_len = []
    z = 0
    start = 0
    for end in num_words:
        for batch in data[start:end]:
            batches.append(batch)
            z += 1
        batch_len.append(z)
        start = end

    return batches, batch_len


def create_batches(data):
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches, batch_len


def create_matrices(sentences, word_to_idx, label_to_idx, case_to_idx, char_to_idx):
    unknown_idx = word_to_idx['UNKNOWN_TOKEN']
    padding_idx = word_to_idx['PADDING_TOKEN']

    dataset = []

    word_count = 0
    unknown_word_count = 0

    for sentence in sentences:
        word_indices = []
        case_indices = []
        char_indices = []
        label_indices = []

        for word, char, label in sentence:
            word_count += 1
            if word in word_to_idx:
                word_idx = word_to_idx[word]
            elif word.lower() in word_to_idx:
                word_idx = word_to_idx[word.lower()]
            else:
                word_idx = unknown_idx
                unknown_word_count += 1
            char_idx = []
            for x in char:
                char_idx.append(char_to_idx[x])
            # Get the label and map to int
            word_indices.append(word_idx)
            case_indices.append(get_casing(word, case_to_idx))
            char_indices.append(char_idx)
            label_indices.append(label_to_idx[label])

        dataset.append([word_indices, case_indices, char_indices, label_indices])

    return dataset


def iterate_minibatches(dataset, batch_len):
    start = 0
    for i in batch_len:
        tokens = []
        caseing = []
        char = []
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t, c, ch, l = dt
            l = np.expand_dims(l, -1)
            tokens.append(t)
            caseing.append(c)
            char.append(ch)
            labels.append(l)

        yield np.asarray(labels), np.asarray(tokens), np.asarray(caseing), np.asarray(char)


def add_char_information(sentences):
    for i, sentence in enumerate(sentences):
        for j, data in enumerate(sentence):
            chars = [c for c in data[0]]
            sentences[i][j] = [data[0], chars, data[1]]
    return sentences


# 0-pads all words
def padding(sentences):
    maxlen = 52
    for sentence in sentences:
        char = sentence[2]
        for x in char:
            maxlen = max(maxlen, len(x))
    for i, sentence in enumerate(sentences):
        sentences[i][2] = pad_sequences(sentences[i][2], 52, padding='post')
    return sentences
