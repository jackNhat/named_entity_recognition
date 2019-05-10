from os.path import join, dirname
import json
import numpy as np
from keras.engine.saving import load_model

from data_utils import create_matrices, padding, add_char_information, create_batches

model_name = join(dirname(__file__), "30_0.5_0.25_200_3_0.0105_Nadam.h5")
model = load_model(model_name)


def read(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def format_data(text):
    sentences = [[item, "O"] for item in text.split(" ")]
    return sentences


word2idx = read("data/word.json")
label2idx = read("data/label2idx.json")
case2idx = read("data/case2idx.json")
char2idx = read("data/char2idx.json")

idx2_label = {v: k for k, v in label2idx.items()}


def predict(word_raw):
    sentence = add_char_information([format_data(word_raw)])
    test_sent = padding(create_matrices(sentence, word2idx, label2idx,
                                        case2idx, char2idx))
    sent_batch, _ = create_batches(test_sent)
    tokens, casing, char, labels = sent_batch[0]
    tokens = np.asarray([tokens])
    casing = np.asarray([casing])
    char = np.asarray([char])
    pred = model.predict([tokens, casing, char], verbose=False)[0]
    pred = pred.argmax(axis=-1)
    pred_sent = list(zip(word_raw.split(), [idx2_label[i] for i in pred]))
    return pred_sent
