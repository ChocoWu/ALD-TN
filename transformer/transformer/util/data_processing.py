import pandas as pd
import numpy as np
import json
from transformer.util.tokenizer import *
import emoji
from transformer.dataset.dataset import OurDataset, Vocabulary
from transformer.util.utils import *
from allennlp.modules.elmo import batch_to_ids


def replace_symbol(text):
    """
    replace HASHTAG MENTION URL PERCENTAGE EMAIL NUM DATE PHONE TIME MONEY
    :param text:
    :return:
    """
    result = text.lower()
    result = emoji.demojize(result)
    result = re.sub(RE_HASHTAG, 'HASHTAG', result)
    result = re.sub(RE_MENTION, 'MENTION', result)
    result = re.sub(RE_URL, 'URL', result)
    result = re.sub(RE_PERCENTAGE, 'PERCENTAGE', result)
    result = re.sub(RE_EMAIL, 'EMAIL', result)
    result = re.sub(RE_NUM, 'NUM', result)
    result = re.sub(RE_DATE, 'DATE', result)
    result = re.sub(RE_PHONE, 'PHONE', result)
    result = re.sub(RE_TIME, 'TIME', result)
    result = re.sub(RE_MONEY, 'MONEY', result)
    return result


def decode_emoji(text):
    """

    :param text:
    :return:
    """
    return emoji.demojize(text)


def process_lexnorm2015(sourcefile, targetfile):
    """
    lexnorm2015 the format of the data:
    {
        "tid": "",
        "index": "",
        "output": [],
        "input": []
    }
    :param sourcefile: source filename .json
    :param targetfile: The name of the file to be saved, the saved file format is csv
    :return:
    """
    src = []
    tgt = []
    with open(sourcefile) as f:
        source_file = json.load(f)
        print('the total of source data is ', len(source_file))
        for data in source_file:
            tgt.append([replace_symbol(w.strip())for w in data['output']])
            src.append([replace_symbol(w.strip())for w in data['input']])
        pickle.dump([src, tgt], open(targetfile, 'wb'))

    print("Lexnorm2015 data, Done!!!")


def remove_0_length(sourcefile, targetfile):
    """
    if the length of one sentence is zero, remove it
    :param sourcefile:
    :param targetfile:
    :return:
    """
    df = pd.read_csv(sourcefile, names=['src', 'tgt'])
    print(df['src'][0])
    print(df['tgt'][0])
    print('the length of the original data: ', len(df))
    data = df[[len(data) > 0 for data in df['src']]]
    print('the length of processed data: ', len(data))
    data.to_csv(targetfile, index=False, header=False)


def process_trac(sourcefile, targetfile):
    """
    process the data from TRAC, the format of the data is csv(id, context, label)
    :param sourcefile:
    :param targetfile:
    :return:
    """
    df = pd.read_csv(sourcefile, names=['id', 'context', 'label'])
    print("the total number of TRAC dataset is ", len(df))
    df['encode_context'] = df.apply(lambda s: replace_symbol(str(s.context)), axis=1)
    print(df['encode_context'][0])
    print(df['label'][0])
    src = [tokenize(s) for s in df['encode_context']]
    tgt = list(df['label'])
    pickle.dump([src, tgt], open(targetfile, 'wb'))


def merage_dataset(sf1, sf2, tf1):
    """

    :param sf1: source file 1 'src' 'tgt'
    :param sf2: source file 2 'context' 'label'
    :param tf1: target file 'src' 'tgt' 'context' 'label'
    :return:
    """
    df1 = pd.read_csv(sf1, names=['src', 'tgt'])
    df2 = pd.read_csv(sf2, names=['src', 'tgt'])
    length = len(df2)
    print(len(df1))
    print('the length of meraged data is ', length)
    data = pd.concat([df1, df2], axis=0)
    data.to_csv(tf1, index=False, header=False)


def load_data_with_diff_vocab(inputFile, src_vocab, tgt_vocab, max_word=200, max_char=50, data_type ='norm'):
    """
    use different to load data, src_vocab, tgt_vocab, src_vocab including the lexnorm2015 and aggressive dataset
    :param inputFile:
    :param src_vocab:
    :param tgt_vocab:
    :param max_word:
    :param max_char:
    :param data_type:
    :return:
    """
    char_inputs = []
    word_inputs = []
    outputs = []
    src, tgt = pickle.load(open(inputFile, 'rb'))
    for s, t in zip(src, tgt):
        elmo_id = batch_to_ids(s)
        elmo_id = elmo_id.view(-1, 50)
        tokens = ['<sos>'] + s + ['<eos>']
        word_input = [src_vocab.word_to_id(word) for word in tokens]
        if data_type == 'norm':
            output = [tgt_vocab.word_to_id(tgt_vocab.SYM_SOS)]
            output.extend([tgt_vocab.word_to_id(word) for word in t])
            output.append(tgt_vocab.word_to_id(tgt_vocab.SYM_EOS))
        else:
            output = [tgt_vocab.tag_to_id(t)]
        char_inputs.append(elmo_id)
        word_inputs.append(word_input)
        outputs.append(output)
    dataset = OurDataset(word_inputs, char_inputs, outputs, max_word, max_char)
    return dataset


def build_src_tgt_vocab(inputFile, src_vocabFile, tgt_vocabFile, label_list):
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src, tgt = pickle.load(open(inputFile, 'rb'))
    for s, t in zip(src, tgt):
        src_vocab.add_sentence(s)
        tgt_vocab.add_sentence(t)
        [src_vocab.add_word(word) for word in s]
        [tgt_vocab.add_word(word) for word in t]

    tgt_vocab.add_tags(label_list)
    save_to_pickle(src_vocabFile, src_vocab)
    save_to_pickle(tgt_vocabFile, tgt_vocab)

    return src_vocab, tgt_vocab


def build_norm_dataset(inputfile, targetfile, dictfile):
    """
    use slangdict_total.pickle ans train.csv to get normalization training dataset
    1.use slang dictionary to get normalized dataset
    2. use slang dictionary to get non-normalized dataset
    :param inputfile:
    :param targetfile:
    :param dictfile: slang dictionary file
    :return:
    """
    slangdict = pickle.load(open(dictfile, 'rb'))
    df = pd.read_csv(inputfile, names=['src', 'tgt'])
    tgt = []
    src = []
    for line in df['src']:
        words = tokenize(line.lower().strip())
        src.append([replace_symbol(x.strip()) for x in words])
        temp = words
        for w in words:
            if w in slangdict:
                temp[temp.index(w)] = slangdict[w]
        tgt.append([replace_symbol(x.strip()) for x in words])
    pickle.dump([src, tgt], open(targetfile, 'wb'))
    print('DONE !!!')


def load_embedding(filename, embedding_size, vocab):
    embeddings_index = {}
    with open(filename, encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    scale = np.sqrt(3.0 / embedding_size)
    embedding = np.random.uniform(-scale, scale, (vocab.n_words, embedding_size))

    for word, vector in embeddings_index.items():
        if word in vocab.word2id:
            embedding[vocab.word2id[word]] = vector

    return embedding


def pad_sequence(sequences, batch_first, max_word = 0, max_char = 0, padding_value = 0):
    """
    针对的是word_sentence_level的pad
    :param sequence: [batch_size, max_sent_num, max_word_num]
    :param batch_first:
    :param pad_value: 默认为0
    :return:
    """
    if max_word == 0 and max_char == 0:
        max_word = max([len(doc) for doc in sequences])
        max_char = max([max([len(sent) for sent in doc]) for doc in sequences])

    if batch_first:
        out_dims = (len(sequences), max_word, max_char)
    else:
        out_dims = (max_word, max_char, len(sequences))

    # out_tensor = sequences[0][0].data.new(*out_dims).fill_(padding_value)
    out_tensor = np.full(out_dims, padding_value)
    # print(out_tensor)
    for i, tensors in enumerate(sequences):
        tensors = tensors[:max_word]
        for j, tensor in enumerate(tensors):
            length = len(tensor)
            if length >= max_char:
                if batch_first:
                    out_tensor[i, j, :max_char, ...] = tensor[:max_char]
                else:
                    out_tensor[j, :max_char, i, ...] = tensor[:max_char]
            else:
                if batch_first:
                    out_tensor[i, j, :length, ...] = tensor
                else:
                    out_tensor[j, :length, i, ...] = tensor

    return out_tensor


if __name__ == '__main__':

    # process lexnorm data
    process_lexnorm2015('../../data/lexnorm2015/train_data.json', "../../data/train_lexnorm2015.pt")
    process_lexnorm2015('../../data/lexnorm2015/test_truth.json', "../../data/test_lexnorm2015.pt")

    # process TRAC data
    process_trac('../../data/trac/agr_en_train.csv', '../../data/agr_en_train.pt')
    process_trac('../../data/trac/agr_en_dev.csv', '../../data/agr_en_dev.pt')
    process_trac('../../data/trac/agr_en_fb_test.csv', '../../data/agr_en_fb_test.pt')
    process_trac('../../data/trac/agr_en_tw_test.csv', '../../data/agr_en_tw_test.pt')

    build_norm_dataset('../../data/trac/agr_en_train.csv', '../../data/agr_en_train_norm.pt',
                       '../../data/slangdict.pickle')

    lex_src, lex_tgt = pickle.load(open('../../data/train_lexnorm2015.pt', 'rb'))
    assert len(lex_src) == len(lex_tgt)
    print('the total number of lexnorm2015 is:', len(lex_src))
    agr_src, agr_tgt = pickle.load(open('../../data/agr_en_train_norm.pt', 'rb'))
    assert len(agr_src) == len(agr_tgt)
    print('the total number of agr_en_train is:', len(agr_src))
    lex_src.extend(agr_src)
    lex_tgt.extend(agr_tgt)
    print('the total number of dict is:', len(lex_src))
    pickle.dump([lex_src, lex_tgt], open('../../data/dict.pt', 'wb'))

