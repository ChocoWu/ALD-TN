import pickle
import logging
import matplotlib
matplotlib.use('Agg')
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
