import numpy as np
import json
from validation import compute_f1
from keras.models import Model
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, \
    Flatten, concatenate
from data_utils import readfile, create_batches, create_matrices, iterate_minibatches, add_char_information, padding
from keras.utils import plot_model
from keras.initializers import RandomUniform


class CNN_BLSTM(object):

    def __init__(self, EPOCHS, DROPOUT, DROPOUT_RECURRENT, LSTM_STATE_SIZE, CONV_SIZE, LEARNING_RATE, OPTIMIZER):

        self.epochs = EPOCHS
        self.dropout = DROPOUT
        self.dropout_recurrent = DROPOUT_RECURRENT
        self.lstm_state_size = LSTM_STATE_SIZE
        self.conv_size = CONV_SIZE
        self.learning_rate = LEARNING_RATE
        self.optimizer = OPTIMIZER

    def load_data(self):
        """Load data and add character information"""
        self.train_sentences = readfile("data/train.txt")
        self.dev_sentences = readfile("data/dev.txt")
        self.test_sentences = readfile("data/test.txt")

    def add_char_info(self):
        self.train_sentences = add_char_information(self.train_sentences)
        self.dev_sentences = add_char_information(self.dev_sentences)
        self.test_sentences = add_char_information(self.test_sentences)

    def embed(self):
        """Create word- and character-level embeddings"""

        label_set = set()
        words = {}

        # unique words and labels in data
        for dataset in [self.train_sentences, self.dev_sentences, self.test_sentences]:
            for sentence in dataset:
                for token, char, label in sentence:
                    # token ... token, char ... list of chars, label ... BIO labels
                    label_set.add(label)
                    words[token.lower()] = True

        # mapping for labels
        self.label_to_idx = {}
        for label in label_set:
            self.label_to_idx[label] = len(self.label_to_idx)

        # mapping for token cases
        case_to_idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
                       'contains_digit': 6, 'PADDING_TOKEN': 7}
        self.case_embeddings = np.identity(len(case_to_idx), dtype='float32')  # identity matrix used

        # read GLoVE word embeddings
        word_to_idx = {}
        self.word_embeddings = []

        f_embeddings = open("data/glove.50d.txt", encoding="utf-8")

        # loop through each word in embeddings
        for line in f_embeddings:
            split = line.strip().split(" ")
            word = split[0]  # embedding word entry

            if len(word_to_idx) == 0:  # add padding+unknown
                word_to_idx["PADDING_TOKEN"] = len(word_to_idx)
                vector = np.zeros(len(split) - 1)  # zero vector for 'PADDING' word
                self.word_embeddings.append(vector)

                word_to_idx["UNKNOWN_TOKEN"] = len(word_to_idx)
                vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
                self.word_embeddings.append(vector)

            if split[0].lower() in words:
                vector = np.array([float(num) for num in split[1:]])
                self.word_embeddings.append(vector)  # word embedding vector
                word_to_idx[split[0]] = len(word_to_idx)  # corresponding word dict

        self.word_embeddings = np.array(self.word_embeddings)

        # dictionary of all possible characters
        self.char_to_idx = {"PADDING": 0, "UNKNOWN": 1}
        for c in "’ỳ‘°fhXLẹủÀgÂếỒừHơý¼[êớ3BùỜnểỗPứỹAlâ+ÔẵÊ/.Ề-jÓ8CởVqĩẨk* " \
                 "òĐỆd4áỏệrUỐỪ>ỮóÐ]ễụRũ²ằự&ZồÕeẶuẽ0wố6ŨẢDSữẩọưQyèO)K³bắvãàÚạ?MÝÁỔỄÙìmặ27ƠỞửÍƯờỉầịĂềổậJđIpõỵẬộ~ôiY–9" \
                 "Ầð:FxG!a,5%(ísả…NWỨoỡTẫéú“ợEẻỲză\"ẤẠỷc;ấẳ1”ỰtỖỦ'":
            self.char_to_idx[c] = len(self.char_to_idx)

        def write(file_name, data):
            with open(file_name, "w") as f:
                json.dump(data, f)

        write("data/word.json", word_to_idx)
        write("data/label2idx.json", self.label_to_idx)
        write("data/case2idx.json", case_to_idx)
        write("data/char2idx.json", self.char_to_idx)
        # format: [[wordindices], [caseindices], [padded word indices], [label indices]]
        self.train_set = padding(
            create_matrices(self.train_sentences, word_to_idx, self.label_to_idx, case_to_idx, self.char_to_idx))
        self.dev_set = padding(
            create_matrices(self.dev_sentences, word_to_idx, self.label_to_idx, case_to_idx, self.char_to_idx))
        self.test_set = padding(
            create_matrices(self.test_sentences, word_to_idx, self.label_to_idx, case_to_idx, self.char_to_idx))

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    def create_batches(self):
        self.train_batch, self.train_batch_len = create_batches(self.train_set)
        self.dev_batch, self.dev_batch_len = create_batches(self.dev_set)
        self.test_batch, self.test_batch_len = create_batches(self.test_set)

    def tag_dataset(self, dataset, model):
        """Tag data with numerical values"""
        correct_labels = []
        pred_labels = []
        for i, data in enumerate(dataset):
            tokens, casing, char, labels = data
            tokens = np.asarray([tokens])
            casing = np.asarray([casing])
            char = np.asarray([char])
            pred = model.predict([tokens, casing, char], verbose=False)[0]
            pred = pred.argmax(axis=-1)  # Predict the classes
            correct_labels.append(labels)
            pred_labels.append(pred)
        return pred_labels, correct_labels

    def build_model(self):
        """Model layers"""

        # character input
        character_input = Input(shape=(None, 52,), name="Character_input")
        embed_char_out = TimeDistributed(
            Embedding(len(self.char_to_idx), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
            name="Character_embedding")(
            character_input)

        dropout = Dropout(self.dropout)(embed_char_out)

        # CNN
        conv1d_out = TimeDistributed(
            Conv1D(kernel_size=self.conv_size, filters=30, padding='same', activation='tanh', strides=1),
            name="Convolution")(dropout)
        maxpool_out = TimeDistributed(MaxPooling1D(52), name="Maxpool")(conv1d_out)
        char = TimeDistributed(Flatten(), name="Flatten")(maxpool_out)
        char = Dropout(self.dropout)(char)

        # word-level input
        words_input = Input(shape=(None,), dtype='int32', name='words_input')
        words = Embedding(input_dim=self.word_embeddings.shape[0], output_dim=self.word_embeddings.shape[1],
                          weights=[self.word_embeddings],
                          trainable=False)(words_input)

        # case-info input
        casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
        casing = Embedding(output_dim=self.case_embeddings.shape[1], input_dim=self.case_embeddings.shape[0],
                           weights=[self.case_embeddings],
                           trainable=False)(casing_input)

        # concat & BLSTM
        output = concatenate([words, casing, char])
        output = Bidirectional(LSTM(self.lstm_state_size,
                                    return_sequences=True,
                                    dropout=self.dropout,  # on input to each LSTM block
                                    recurrent_dropout=self.dropout_recurrent  # on recurrent input signal
                                    ), name="BLSTM")(output)
        output = TimeDistributed(Dense(len(self.label_to_idx), activation='softmax'), name="Softmax_layer")(output)

        # set up model
        self.model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer)

        self.init_weights = self.model.get_weights()

        plot_model(self.model, to_file='model.png')

        print("Model built. Saved model.png\n")

    def train(self):
        """Default training"""

        self.f1_test_history = []
        self.f1_dev_history = []

        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch, self.epochs))
            for i, batch in enumerate(iterate_minibatches(self.train_batch, self.train_batch_len)):
                labels, tokens, casing, char = batch
                self.model.train_on_batch([tokens, casing, char], labels)

            # compute F1 scores
            pred_labels, correct_labels = self.tag_dataset(self.test_batch, self.model)
            pre_test, rec_test, f1_test = compute_f1(pred_labels, correct_labels, self.idx_to_label)
            self.f1_test_history.append(f1_test)
            print("f1 test ", round(f1_test, 4))

            pred_labels, correct_labels = self.tag_dataset(self.dev_batch, self.model)
            pre_dev, rec_dev, f1_dev = compute_f1(pred_labels, correct_labels, self.idx_to_label)
            self.f1_dev_history.append(f1_dev)
            print("f1 dev ", round(f1_dev, 4), "\n")

        print("Final F1 test score: ", f1_test)

        print("Training finished.")

        # save model
        self.model_name = "{}_{}_{}_{}_{}_{}_{}".format(self.epochs,
                                                        self.dropout,
                                                        self.dropout_recurrent,
                                                        self.lstm_state_size,
                                                        self.conv_size,
                                                        self.learning_rate,
                                                        self.optimizer.__class__.__name__
                                                        )

        model_name = self.model_name + ".h5"
        self.model.save(model_name)
        print("Model weights saved.")

        self.model.set_weights(self.init_weights)  # clear model
        print("Model weights cleared.")

    def writefile(self):
        """Write output to file"""
        output = np.matrix([[int(i) for i in range(self.epochs)], self.f1_test_history, self.f1_dev_history])

        file_name = self.model_name + ".txt"
        with open(file_name, 'wb') as f:
            for line in output:
                np.savetxt(f, line, fmt='%.5f')

        print("Model performance written to file.")

    print("Class initialised.")
