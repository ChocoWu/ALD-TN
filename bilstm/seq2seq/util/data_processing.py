import pandas as pd
import numpy as np
import json
import csv
import torch
from .tokenizer import *
import emoji
from bilstm.seq2seq.dataset.dataset import OurDataset, Vocabulary
from bilstm.seq2seq.util.utils import *
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
    result = re.sub(r'[\"\'\|\*\/]+', '', result)
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
    with open(sourcefile) as f:
        source_file = json.load(f)
        print('the total of source data is ', len(source_file))
        with open(targetfile, 'a') as f:
            csv_write = csv.writer(f)
            for data in source_file:
                output = ' '.join(data['output']).strip()
                input = ' '.join(data['input']).strip()
                output = replace_symbol(output)
                input = replace_symbol(input)
                csv_write.writerow([input, output])

    print("Lexnorm2015 data, Done!!!")


def process_kaggle(sourcefile, targetfile):
    """
    the format of data is:
    "sentence_id","token_id","class","before","after"
    :param sourcefile: source filename .csv
    :param targetfile: The name of the file to be saved, the saved file format is csv , ['source', 'target']
    :return:
    """
    df = pd.read_csv(sourcefile, header=0, names=["sentence_id", "token_id", "class", "before", "after"])
    # join the word to a sentence
    sentences = df.groupby("sentence_id")
    print("How long are the sentences like?\n", sentences['sentence_id'].count().describe())
    with open(targetfile, 'a') as f:
        csv_write = csv.writer(f)
        for sent in sentences:
            before = ' '.join([str(s) for s in sent[1]["before"]]).strip()
            after = ' '.join([str(s) for s in sent[1]["after"]]).strip()
            # remove the punctuation
            before = re.sub(RE_PUNCT, '', before)
            after = re.sub(RE_PUNCT, '', after)
            csv_write.writerow([before, after])
    print("kaggle data done!!!")


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
    df.to_csv(targetfile, columns=['encode_context', 'label'], index=False, header=False)


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
    print('the length of meraged data is ', length)
    data = pd.concat([df1, df2], axis=0)
    data.to_csv(tf1, index=False, header=False)


def load_data_with_diff_vocab(inputFile, src_vocab, tgt_vocab, max_word=200, max_char=50, type ='norm'):
    """
    use different to load data, src_vocab, tgt_vocab, src_vocab including the lexnorm2015 and aggressive dataset
    :param inputFile:
    :param src_vocab:
    :param tgt_vocab:
    :param max_word:
    :param max_char:
    :param type:
    :return:
    """
    char_inputs = []
    word_inputs = []
    outputs = []
    max_output = 0
    df = pd.read_csv(inputFile, names=['src', 'tgt'])
    for src, tgt in zip(df['src'], df['tgt']):
        elmo_id = batch_to_ids([tokenize(str(src))])
        elmo_id = elmo_id.view(-1, 50)
        word_input = [src_vocab.word_to_id(word) for word in tokenize(str(src))]
        if type == 'norm':
            output = [tgt_vocab.word_to_id(tgt_vocab.SYM_SOS)]
            output.extend([tgt_vocab.word_to_id(word) for word in tgt.strip().split(' ')])
            output.append(tgt_vocab.word_to_id(tgt_vocab.SYM_EOS))
        else:
            output = [tgt_vocab.tag_to_id(tag) for tag in tgt.strip().split()]
        char_inputs.append(elmo_id)
        word_inputs.append(word_input)
        # print(str(idx), len(output))
        # idx += 1
        outputs.append(output)
    max_output = max([len(sent) for sent in outputs])
    # print(max_output)
    outputs = list(map(lambda d: d[:max_output], outputs))
    outputs = list(map(lambda d: d + (max_output - len(d)) * [tgt_vocab.word_to_id(tgt_vocab.W_PAD)], outputs))
    word_inputs = list(map(lambda d: d[:max_word], word_inputs))
    word_inputs = list(map(lambda d: d + (max_word - len(d)) * [src_vocab.word_to_id(src_vocab.W_PAD)], word_inputs))
    char_inputs = pad_sequence(char_inputs, True, max_word, max_char)

    # char_input = [[vocab.char_to_id(char) for char in word] for word in tokenize(str(src))]

    dataset = OurDataset(word_inputs, char_inputs, outputs)
    return dataset


def load_data_with_same_vocab(inputFile, vocab, type ='norm', max_word=200, max_char=50):
    """
    use the same vocabulary to load data
    :param inputFile:
    :param vocab:
    :param type:
    :param max_word:
    :param max_char:
    :return:
    """
    char_inputs = []
    word_inputs = []
    outputs = []
    max_output = 0
    df = pd.read_csv(inputFile, names=['src', 'tgt'])
    idx = 1
    for src, tgt in zip(df['src'], df['tgt']):
        elmo_id = batch_to_ids([tokenize(str(src))])
        elmo_id = elmo_id.view(-1, 50)
        word_input = [vocab.word_to_id(word) for word in tokenize(str(src))]
        if type == 'norm':
            output = [vocab.word_to_id(word) for word in tgt.strip().split(' ')]
        else:
            output = [vocab.tag_to_id(tag) for tag in tgt.strip().split()]
        char_inputs.append(elmo_id)
        word_inputs.append(word_input)
        # print(str(idx), len(output))
        # idx += 1
        outputs.append(output)
    max_output = max([len(sent) for sent in outputs])
    # print(max_output)
    outputs = list(map(lambda d: d[:max_output], outputs))
    outputs = list(map(lambda d: d + (max_output - len(d)) * [vocab.word_to_id(vocab.W_PAD)], outputs))
    word_inputs = list(map(lambda d: d[:max_word], word_inputs))
    word_inputs = list(map(lambda d: d + (max_word - len(d)) * [vocab.word_to_id(vocab.W_PAD)], word_inputs))
    char_inputs = pad_sequence(char_inputs, True, max_word, max_char)

    # char_input = [[vocab.char_to_id(char) for char in word] for word in tokenize(str(src))]

    dataset = OurDataset(char_inputs, word_inputs, outputs)
    return dataset


def build_data(doc, labels):
    assert len(doc) == len(labels)
    tokens, labs = [], []
    for sents, label in zip(doc, labels):
        # labs.append(str(label))
        labs.append(str(label).split())
        temp = str(sents).split('|')
        res = []
        for sent in temp:
            words = sent.strip().split()
            res.append(words)
        tokens.append(res)

    return tokens, labs


def build_data_c(doc, labels):
    assert len(doc) == len(labels)
    w_tokens, c_tokens, labs = [], [], []
    for sents, label in zip(doc, labels):
        labs.append(str(label))
        s_temp = sents.split("|")
        s_res = []
        w_res = []
        for sent in s_temp:
            words = sent.strip().split()
            s_res.append(words)
            temp = []
            for word in words:
                temp.append([c for c in word])
            w_res.append(temp)
        w_tokens.append(s_res)
        c_tokens.append(w_res)
    return w_tokens, c_tokens, labs


def pad_sequence_c(sequences, max_sent=None, max_word=None, max_char=None, padding_value=0):
    """
    针对char_word_sentence_level的pad
    :param sequences: 边长的数据
    :param max_sent:
    :param max_word:
    :param max_char:
    :param padding_value:
    :return:
        batch_size, max_sent, max_word, max_char
    """
    if max_sent is None and max_word is None and max_char is None:
        max_sent = max([len(doc) for doc in sequences])
        max_word = max([max([len(sent) for sent in doc]) for doc in sequences])
        max_char = max([max([max(len(word) for word in sent) for sent in doc]) for doc in sequences])

    out_dims = (len(sequences), max_sent, max_word, max_char)
    out_tensor = np.full(out_dims, padding_value)
    for i, doc in enumerate(sequences):
        doc = doc[:max_sent]
        for j, sent in enumerate(doc):
            sent = sent[:max_word]
            for k, word in enumerate(sent):
                length = len(word)
                if length > max_char:
                    out_tensor[i, j, k, :max_char, ...] = word[:max_char]
                else:
                    out_tensor[i, j, k, :length, ...] = word[:length]

    return out_tensor


def load_data(filename, max_sent, max_word, src_vocab=None, tgt_vocab=None, model_type=None, data_type='norm'):
    df = pd.read_csv(filename, header=0, names=['content', 'label'])
    # df = pd.read_csv(filename, header=0, names=['content', 'label'])
    outputs = []
    if model_type == 'char':
        w_data, c_data, label = build_data_c(df['content'], df['label'])
        w_input = [[[src_vocab.word_to_id(word) for word in sent] for sent in doc] for doc in w_data]
        c_input = [[[[src_vocab.char_to_id(c) for c in word] for word in sent] for sent in doc] for doc in c_data]
        w_input = pad_sequence(w_input, True, max_sent, max_word)
        c_input = pad_sequence_c(c_input, max_sent, max_word, 25)
        # label = [tgt_vocab.tag_to_id(label) for label in label]
        label = [[tgt_vocab.tag_to_id(x) for x in l] for l in label]
        dataset = OurDataset(w_input, char_inputs=c_input, labels=label)
        return dataset
    elif model_type == 'elmo':
        w_data, label = build_data(df['content'], df['label'])

        character_ids = []
        for data in w_data:
            temp_tensor = torch.zeros(max_sent, max_word, 50, dtype=torch.long)
            temp_ids = batch_to_ids(data)
            t_max_sent = temp_ids.size(0)
            t_max_word = temp_ids.size(1)
            if t_max_sent > max_sent:
                if t_max_word > max_word:
                    temp_tensor = temp_ids[:max_sent, :max_word, :]
                else:
                    temp_tensor[:, :t_max_word, :] = temp_ids[:max_sent, :, :]
            else:
                if t_max_word > max_word:
                    temp_tensor[:t_max_sent, :, :] = temp_ids[:, :max_word, :]
                else:
                    temp_tensor[:t_max_sent, :t_max_word, :] = temp_ids[:, :, :]
            character_ids.append(temp_tensor)

        w_input = [[[src_vocab.word_to_id(word) for word in sent] for sent in doc] for doc in w_data]
        w_input = pad_sequence(w_input, True, max_sent, max_word)
        for l in label:
            if data_type == 'norm':
                output = [tgt_vocab.word_to_id(tgt_vocab.SYM_SOS)]
                output.extend([tgt_vocab.word_to_id(word) for word in l])
                output.append(tgt_vocab.word_to_id(tgt_vocab.SYM_EOS))
            else:
                output = [tgt_vocab.tag_to_id(tag) for tag in l]
            outputs.append(output)
        max_output = max([len(sent) for sent in outputs])
        # print(max_output)
        outputs = list(map(lambda d: d[:max_output], outputs))
        outputs = list(map(lambda d: d + (max_output - len(d)) * [tgt_vocab.word_to_id(tgt_vocab.W_PAD)], outputs))
        # label = [[tgt_vocab.tag_to_id(x) for x in l] for l in label]
        dataset = OurDataset(w_input, char_inputs=character_ids, labels=outputs)

        return dataset
    else:
        w_data, label = build_data(df['content'], df['label'])
        w_input = [[[src_vocab.word_to_id(word) for word in sent] for sent in doc] for doc in w_data]
        w_input = pad_sequence(w_input, True, max_sent, max_word)
        for l in label:
            if data_type == 'norm':
                output = [tgt_vocab.word_to_id(tgt_vocab.SYM_SOS)]
                output.extend([tgt_vocab.word_to_id(word) for word in l])
                output.append(tgt_vocab.word_to_id(tgt_vocab.SYM_EOS))
            else:
                output = [tgt_vocab.tag_to_id(tag) for tag in l]
            outputs.append(output)
        max_output = max([len(sent) for sent in outputs])
        # print(max_output)
        outputs = list(map(lambda d: d[:max_output], outputs))
        outputs = list(map(lambda d: d + (max_output - len(d)) * [tgt_vocab.word_to_id(tgt_vocab.W_PAD)], outputs))
        # label = [[tgt_vocab.tag_to_id(x) for x in l] for l in label]
        label = [[tgt_vocab.tag_to_id(x) for x in l] for l in label]
        dataset = OurDataset(w_input, char_inputs=None, labels=outputs)

        return dataset


def split_kaggle_data(filename, new_filename):
    data = pd.read_csv(filename, names=['context', 'label'])
    new_content = []
    print('the number of {} is {}'.format(filename, len(data)))
    for line in data['context']:
        sents = '|'.join(split_sentence(replace_symbol(str(line))))
        new_content.append(sents)
    df = pd.DataFrame({'content': new_content, 'label': data['label']})
    df.to_csv(new_filename, index=False, header=False)
    print('split data DONE!')


def split_sentence(text):
    try:
        sents = re.split('[.?,;!]+', text)
        i = 0
        while True:
            if i == len(sents):
                break
            sent = sents[i].strip()
            if len(sent) == 0:
                sents.remove(sents[i])
            elif len(sent) < len(sents[i]):
                sents[i] = sent
                i += 1
            else:
                i += 1
        return sents
    except:
        return ''


def build_src_tgt_vocab(inputFile, src_vocabFile, tgt_vocabFile):
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    df = pd.read_csv(inputFile, names=['src', 'tgt'])
    for src, tgt in zip(df['src'], df['tgt']):
        # print(tgt)
        src = tokenize(str(src))
        tgt = tokenize(tgt)
        src_vocab.add_sentence(src)
        tgt_vocab.add_sentence(tgt)
        [src_vocab.add_word(word) for word in src]
        [tgt_vocab.add_word(word) for word in tgt]
    tags = ['CAG', 'OAG', 'NAG']
    tgt_vocab.add_tags(tags)
    save_to_pickle(src_vocabFile, src_vocab)
    save_to_pickle(tgt_vocabFile, tgt_vocab)

    return src_vocab, tgt_vocab


def build_vocab(inputFile, vocabFile):
    vocab = Vocabulary()
    df = pd.read_csv(inputFile, names=['src', 'tgt'])
    for src, tgt in zip(df['src'], df['tgt']):
        # print(tgt)
        src = tokenize(str(src))
        tgt = tokenize(tgt)
        vocab.add_sentence(src)
        vocab.add_sentence(tgt)
        [vocab.add_word(word) for word in src]
        [vocab.add_word(word) for word in tgt]
    tags = ['CAG', 'OAG', 'NAG']
    vocab.add_tags(tags)
    save_to_pickle(vocabFile, vocab)
    return vocab


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
    inv_slangdict = {v: k for k, v in slangdict.items()}
    df = pd.read_csv(inputfile, names=['src', 'tgt'])
    # df['new_src'] = df.apply(lambda x: df['src'][x].lower(), axis=1)
    # df['new_src'] = df.apply(lambda x: replace_symbol(x.src), axis=1)
    # new_src = []
    tgt = []
    for line in df['src']:
        line = replace_symbol(line)
        words = tokenize(line.strip())
        # temp = words.copy()
        temp = words
        for w in words:
            if w in slangdict:
                temp[temp.index(w)] = slangdict[w]
            # if w in inv_slangdict and np.random.randint(0, 2) == 1:
            #     temp[temp.index(w)] = inv_slangdict[w]
        # new_src.append(' '.join(words))
        tgt.append(' '.join(temp))
    # data = pd.DataFrame({"src": list(df['new_src']), "tgt": new_src})
    data = pd.DataFrame({"src": list(df['src']), "tgt": tgt})
    data.to_csv(targetfile, index=False, header=False)
    print('DONE !!!')


if __name__ == '__main__':
    # sent = json.load(open("../../data/lexnorm2015/train_data.json"))
    # print(sent[0])
    # input = ' '.join(sent[0]['input'])
    # print(input)

    # df = pd.read_csv("../../data/kaggle/en_train.csv")
    # sentences = df.groupby("sentence_id")
    # print("How long are the sentences like?\n", sentences['sentence_id'].count().describe())

    # process_lexnorm2015('../../data/lexnorm2015/train_data.json', "../../data/train_lexnrom2015.csv")
    # process_lexnorm2015('../../data/lexnorm2015/test_truth.json', '../../data/test_lexnorm2015.csv')
    # process_kaggle("../../data/kaggle/en_train.csv", "../../data/train_kaggle.csv")

    # merage the  two source file
    # with open("../../data/train_lexnrom2015.csv", 'r') as f:
    #     for line in f:
    #         with open('../../data/slangdict.csv', 'a') as f_1:
    #             f_1.write(line)

    # split data into train data, dev data, test data
    # df = pd.read_csv('../../data/dict.csv', header=0)
    # train, test = train_test_split(df, shuffle=True, test_size=0.3)
    # # dev, test = train_test_split(test, shuffle=True, test_size=0.5)
    # train.to_csv("../../data/train.csv", header=None, index=None)
    # test.to_csv('../../data/test.csv', header=None, index=None)
    # dev.to_csv("../../data/dev.csv", header=None, index=None)

    # remove sentence that length is zero
    # df = pd.DataFrame({"key": ['green', 'red', 'blue'],
    #                    "data1": ['a', 'b', 'c'], "sorce": [33, 61, 99]})
    # data = pd.concat([df, df], ignore_index=True)
    # print(data)
    # print([len(d) > 3 for d in data['key']])
    # print(data[[len(d) > 3 for d in data['key']]])
    # remove_0_length('../../data/train.csv', '../../data/train_1.csv')

    # process TRAC data
    # process_trac('../../data/trac/agr_en_train.csv', '../../data/agr_en_train.csv')
    # process_trac('../../data/trac/agr_en_dev.csv', '../../data/agr_en_dev.csv')
    # process_trac('../../data/trac/agr_en_fb_test.csv', '../../data/agr_en_fb_test.csv')
    # process_trac('../../data/trac/agr_en_tw_test.csv', '../../data/agr_en_tw_test.csv')

    # merage data
    merage_dataset('../../data/train_lexnrom2015.csv', '../../data/agr_en_train_norm.csv', '../../data/train_new.csv')
    # merage_dataset('../../data/dev.csv', '../../data/agr_en_dev.csv', '../../data/dev_new.csv')
    # merage_dataset('../../data/test.csv', '../../data/agr_en_fb_test_p.csv', '../../data/test_1.csv')
    # merage_dataset('../../data/test.csv', '../../data/agr_en_tw_test_p.csv', '../../data/test_2.csv')

    # for build total dictionary, merage train.csv and agr_en_train.csv
    # df1 = pd.read_csv('../../data/train.csv', names=['src', 'tgt'])
    # print("the length if train.csv is ", len(df1))
    # df2 = pd.read_csv('../../data/agr_en_train.csv', names=['src', 'tgt'])
    # print("the length if agr_en_train.csv is ", len(df2))
    # df = pd.concat([df1, df2], ignore_index=True)
    # print("the length if dict.csv is ", len(df))
    # df.to_csv('../../data/dict.csv', header=None, index=None)

    # the size of dictionary
    # df = pd.read_csv('../../data/dict.csv', names=['src', 'tgt'])
    # dictionary = {}
    # for data in df['tgt']:
    #     s = tokenize(str(data))
    #     for word in s:
    #         if word in dictionary:
    #             dictionary[word] += 1
    #         else:
    #             dictionary[word] = 0
    # print(len(dictionary))  # src the total number is 321077 ; tgt the total number is 280589

    # slangdict = pickle.load(open('../../data/slangdict.pickle', 'rb'))
    # with open('../../data/slangdict.csv', 'w') as f:
    #     for k, v in slangdict.items():
    #         f.write(k + ',' + v + '\n')

    # merge train_lexnorm2015.csv and slangdict.csv
    # merage_dataset('../../data/dict.csv', '../../data/agr_en_train.csv', '../../data/new_dict.csv')

    # split_sentence('l love ,,,.,.,you, nihao , ? ???,ni ')

    # split_kaggle_data('../../data/kaggle/kaggle_train.csv', '../../data/kaggle/kaggle_train_1.csv')
    # split_kaggle_data('../../data/kaggle/kaggle_test.csv', '../../data/kaggle/kaggle_test_1.csv')

    # build_norm_dataset('../../data/agr_en_train.csv', '../../data/agr_en_train_norm.csv',
    #                    '../../data/lexnorm2015/slangdict_total.pickle')

    pass
