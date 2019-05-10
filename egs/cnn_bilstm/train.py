import matplotlib.pyplot as plt
from keras.optimizers import Nadam

from model import CNN_BLSTM

EPOCHS = 20
DROPOUT = 0.5
DROPOUT_RECURRENT = 0.25
LSTM_STATE_SIZE = 200
CONV_SIZE = 3
LEARNING_RATE = 0.0105
OPTIMIZER = Nadam()

cnn_blstm = CNN_BLSTM(EPOCHS, DROPOUT, DROPOUT_RECURRENT, LSTM_STATE_SIZE, CONV_SIZE, LEARNING_RATE, OPTIMIZER)
cnn_blstm.load_data()
cnn_blstm.add_char_info()
cnn_blstm.embed()
cnn_blstm.create_batches()
cnn_blstm.build_model()
cnn_blstm.train()
cnn_blstm.writefile()

plt.plot(cnn_blstm.f1_test_history, label="F1 test")
plt.plot(cnn_blstm.f1_dev_history, label="F1 dev")
plt.xlabel("Epochs")
plt.ylabel("F1 score")
plt.legend()
plt.show()
