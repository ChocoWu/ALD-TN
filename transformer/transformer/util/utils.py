import pickle
import logging
import sys
import matplotlib.pyplot as plt


def save_to_pickle(path, obj):
    file = open(path, 'wb')
    pickle.dump(obj, file)
    file.close()

    return 1


def load_from_pickle(path):
    file = open(path, 'rb')
    obj = pickle.load(file)
    file.close()

    return obj


def get_logger(pathname):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


# visualiation loss
def visual_loss(loss, save_path):
    """

    :param loss: an array
    :param save_path: save the picture
    :return:
    """
    plt.figure()
    x = [i for i in range(len(loss))]
    plt.plot(x, loss, label="$loss$", color='red', linewidth=2)
    plt.legend()
    plt.savefig(save_path)


def visual_acc(dev_acc, test_acc, fb_acc, tw_acc, save_name):
    plt.figure()
    colors = ['red', 'blue', 'green', 'yellow']
    assert len(dev_acc) == len(test_acc) and len(test_acc) == len(fb_acc) and len(fb_acc) == len(tw_acc)
    x = [i for i in range(len(dev_acc))]
    plt.plot(x, dev_acc, label="dev_acc", color=colors[0], linewidth=2)
    plt.plot(x, test_acc, label="test_acc", color=colors[1], linewidth=2)
    plt.plot(x, fb_acc, label="fb_acc", color=colors[2], linewidth=2)
    plt.plot(x, tw_acc, label='tw_acc', color=colors[3], linewidth=2)

    plt.legend()
    plt.savefig(save_name)


def accuracy(y_true, y_pred, eos, max_word):
    total = 0
    correct = 0
    assert len(y_true) == len(y_pred)
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            if y_pred[i][j] == eos:
                break
            elif y_true[i][j] == y_pred[i][j]:
                correct += 1
                total += 1
            else:
                total += 1
    return correct / total


def evaluate_metrics(x, y_true, y_pred, eos, input_vocab, output_vocab):
    """

    :param x: the original input [batch_size, seq_len]
    :param y_true: normalization input [batch_size, seq_len]
    :param y_pred: predicted normalization input [batch_size, [seq_len]
    :param eos: end of sentence
    :param input_vocab:
    :param output_vocab:
    :return:
    """
    correct_norm = 0.0
    total_norm = 0.0
    total_nsw = 0.0
    p, r, f1 = 0.0, 0.0, 0.0
    assert len(x) == len(y_pred) == len(y_true)

    for src, pred, gold in zip(x, y_pred, y_true):
        try:
            i = 0
            while i != src.index(eos) and i < len(gold):
                if output_vocab.id2word[pred[i]] != input_vocab.id2word[src[i]] and output_vocab.id2word[gold[i]] == output_vocab.id2word[pred[i]]:
                    correct_norm += 1
                if output_vocab.id2word[gold[i]] != input_vocab.id2word[src[i]]:
                    total_nsw += 1
                if output_vocab.id2word[pred[i]] != input_vocab.id2word[src[i]]:
                    total_norm += 1
                i += 1
        except AssertionError:
            print("Invalid data format")
            sys.exit(1)
    # calc p, r, f
    p = correct_norm / total_norm
    r = correct_norm / total_nsw
    if p != 0 and r != 0:
        f1 = (2 * p * r) / (p + r)
    return p, r, f1