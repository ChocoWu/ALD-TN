#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

import pickle
import json
import pandas as pd
from transformer.transformer.util.tokenizer import *
import emoji


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
        print('the total of {} is {}'.format(source_file, len(source_file)))
        for data in source_file:
            # tgt.append(replace_symbol(' '.join(data['output']).strip()))
            # src.append(replace_symbol(' '.join(data['input']).strip()))
            tgt.append([replace_symbol(w.strip())for w in data['output']])
            src.append([replace_symbol(w.strip())for w in data['input']])
        # df = pd.DataFrame({'src': src, 'tgt': tgt})
        # df.to_csv(targetfile, header=False, index=False, encoding='utf-8')
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
    print('the total of {} is {}'.format(sourcefile, len(df)))
    data = df[[len(data) > 0 for data in df['src']]]
    print('the length of processed data: ', len(data))
    data.to_csv(targetfile, index=False, header=False, encoding='utf-8')


def process_trac(sourcefile, targetfile):
    """
    process the data from TRAC, the format of the data is csv(id, context, label)
    :param sourcefile:
    :param targetfile: The name of the file to be saved, the saved file format is csv
    :return:
    """
    df = pd.read_csv(sourcefile, names=['id', 'context', 'label'])
    print("the total of {} is {}".format(sourcefile, len(df)))
    df['encode_context'] = df.apply(lambda s: replace_symbol(str(s.context)), axis=1)
    print(df['encode_context'][0])
    print(df['label'][0])
    df.to_csv(targetfile, columns=['encode_context', 'label'], index=False, header=False, encoding='utf-8')
    src = [tokenize(s) for s in df['encode_context']]
    tgt = list(df['label'])
    # data = pd.DataFrame({'src': src, 'tgt': tgt})
    # data.to_csv(targetfile, index=False, header=False, encoding='utf-8')
    pickle.dump([src, tgt], open(targetfile, 'wb'))
    print("process trac data done")


def merage_dataset(sf1, sf2, tf1):
    """

    :param sf1: source file 1 'src' 'tgt'
    :param sf2: source file 2 'context' 'label'
    :param tf1: target file 'src' 'tgt' 'context' 'label'
    :return:
    """
    df1 = pd.read_csv(sf1, names=['src', 'tgt'])
    df2 = pd.read_csv(sf2, names=['src', 'tgt'])
    print('the data number of {} is {}'.format(sf1, len(df1)))
    print('the data number of {} is {}'.format(sf2, len(df2)))

    data = pd.concat([df1, df2], axis=0)
    print('the merged data number of {} is {}'.format(tf1, len(data)))
    data.to_csv(tf1, index=False, header=False, encoding='utf-8')
    print('merge data successfully')


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
        # src.append(replace_symbol(' '.join(words).strip()))
        temp = words
        for w in words:
            if w in slangdict:
                temp[temp.index(w)] = slangdict[w]
        tgt.append([replace_symbol(x.strip()) for x in words])
        # tgt.append(replace_symbol(' '.join(words).strip()))
    # data = pd.DataFrame({'src': src, 'tgt': tgt})
    # data.to_csv(targetfile, index=False, header=False, encoding='utf-8')
    pickle.dump([src, tgt], open(targetfile, 'wb'))
    print('DONE !!!')


if __name__ == '__main__':

    # process lexnorm data
    process_lexnorm2015('./data/lexnorm2015/train_data.json', "./data/train_lexnorm2015.pt")
    process_lexnorm2015('./data/lexnorm2015/test_truth.json', "./data/test_lexnorm2015.pt")

    # process TRAC data
    process_trac('./data/trac/agr_en_train.csv', './data/agr_en_train.pt')
    process_trac('./data/trac/agr_en_dev.csv', './data/agr_en_dev.pt')
    process_trac('./data/trac/agr_en_fb_test.csv', './data/agr_en_fb_test.pt')
    process_trac('./data/trac/agr_en_tw_test.csv', './data/agr_en_tw_test.pt')

    build_norm_dataset('./data/trac/agr_en_train.csv', './data/agr_en_train_norm.pt',
                       './data/slangdict.pickle')

    # merge lexnorm2015 and trac dataset
    # merage_dataset('./data/agr_en_train_norm.pt', './data/train_lexnorm2015.pt', './data/dict.csv')
    lex_src, lex_tgt = pickle.load(open('./data/train_lexnorm2015.pt', 'rb'))
    assert len(lex_src) == len(lex_tgt)
    print('the total number of lexnorm2015 is:', len(lex_src))
    agr_src, agr_tgt = pickle.load(open('./data/agr_en_train_norm.pt', 'rb'))
    assert len(agr_src) == len(agr_tgt)
    print('the total number of agr_en_train is:', len(agr_src))
    lex_src.extend(agr_src)
    lex_tgt.extend(agr_tgt)
    print('the total number of dict is:', len(lex_src))
    pickle.dump([lex_src, lex_tgt], open('./data/dict.pt', 'wb'))
