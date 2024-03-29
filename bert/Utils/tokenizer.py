'''
Splits up a Unicode string into a list of tokens.
Recognises:
- Abbreviations
- URLs
- Emails
- #hashtags
- @mentions
- emojis
- emoticons (limited support)
Multiple consecutive symbols are also treated as a single token.
'''
from __future__ import absolute_import, division, print_function, unicode_literals

import re

# Basic patterns.
RE_NUM = r"\b\d+(?:[\.,']\d+)?\b"
RE_PERCENTAGE = RE_NUM + "%"
RE_WORD = r'[a-zA-Z]+'
RE_WHITESPACE = r'\s+'
RE_ANY = r'.'
RE_PUNCT = r'[\.\,\"\(\)\!\?\:]+'
RE_PHONE = r'(?<![0-9])(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?![0-9])'
RE_MONEY = r"(?:[$€£¢]\d+(?:[\.,']\d+)?(?:[MmKkBb](?:n|(?:il(?:lion)?))?)?)|(?:\d+(?:[\.,']\d+)?[$€£¢])"


__short_date = r"(?:\b(?<!\d\.)(?:(?:(?:[0123]?[0-9][\.\-\/])?[0123]?[0-9][\.\-\/][12][0-9]{3})|(?:[0123]?[0-9][\.\-\/][0123]?[0-9][\.\-\/][12]?[0-9]{2,3}))(?!\.\d)\b)"
__full_date_parts = [
    # prefix
    r"(?:(?<!:)\b\'?\d{1,4},? ?)",

    # month names
    r"\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\b",

    # suffix
    r"(?:(?:,? ?\'?)?\d{1,4}(?:st|nd|rd|n?th)?\b(?:[,\/]? ?\'?\d{2,4}[a-zA-Z]*)?(?: ?- ?\d{2,4}[a-zA-Z]*)?(?!:\d{1,4})\b)",
]
__fd1 = "(?:{})".format("".join(
    [__full_date_parts[0] + "?", __full_date_parts[1], __full_date_parts[2]]))
__fd2 = "(?:{})".format("".join(
    [__full_date_parts[0], __full_date_parts[1], __full_date_parts[2] + "?"]))
__date = "(?:" + "(?:" + __fd1 + "|" + __fd2 + ")" + "|" + __short_date + ")"

RE_DATE = __date
RE_TIME = r"(?:(?:\d+)?\.?\d+(?:AM|PM|am|pm|a\.m\.|p\.m\.))|(?:(?:[0-2]?[0-9]|[2][0-3]):(?:[0-5][0-9])(?::(?:[0-5][0-9]))?(?: ?(?:AM|PM|am|pm|a\.m\.|p\.m\.))?)"

# Combined words such as 'red-haired' or 'CUSTOM_TOKEN'
RE_COMB = r'[a-zA-Z]+[-_][a-zA-Z]+'

# English-specific patterns
RE_CONTRACTIONS = RE_WORD + r'\'' + RE_WORD

TITLES = [
    r'Mr\.',
    r'Ms\.',
    r'Mrs\.',
    r'Dr\.',
    r'Prof\.',
    ]
# Ensure case insensitivity
RE_TITLES = r'|'.join([r'(?i)' + t for t in TITLES])

# Symbols have to be created as separate patterns in order to match consecutive
# identical symbols.
SYMBOLS = r'()<!?.,/\'\"-_=\\§|´ˇ°[]<>{}~$^&*;:%+\xa3€`'
RE_SYMBOL = r'|'.join([re.escape(s) + r'+' for s in SYMBOLS])

# Hash symbols and at symbols have to be defined separately in order to not
# clash with hashtags and mentions if there are multiple - i.e.
# ##hello -> ['#', '#hello'] instead of ['##', 'hello']
SPECIAL_SYMBOLS = r'|#+(?=#[a-zA-Z0-9_]+)|@+(?=@[a-zA-Z0-9_]+)|#+|@+'
RE_SYMBOL += SPECIAL_SYMBOLS

RE_ABBREVIATIONS = r'\b(?<!\.)(?:[A-Za-z]\.){2,}'

# Twitter-specific patterns
RE_HASHTAG = r'#[a-zA-Z0-9_]+'
RE_MENTION = r'@[a-zA-Z0-9_]+'

RE_URL = r'(?:https?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
RE_EMAIL = r'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b'

# Emoticons and emojis
RE_HEART = r'(?:<+/?3+)+'
EMOTICONS_START = [
    r'>:',
    r':',
    r'=',
    r';',
    ]
EMOTICONS_MID = [
    r'-',
    r',',
    r'^',
    '\'',
    '\"',
    ]
EMOTICONS_END = [
    r'D',
    r'd',
    r'p',
    r'P',
    r'v',
    r')',
    r'o',
    r'O',
    r'(',
    r'3',
    r'/',
    r'|',
    '\\',
    ]
EMOTICONS_EXTRA = [
    r'-_-',
    r'x_x',
    r'^_^',
    r'o.o',
    r'o_o',
    r'(:',
    r'):',
    r');',
    r'(;',
    r'＼(^o^)／',
    ]

RE_EMOTICON = r'|'.join([re.escape(s) for s in EMOTICONS_EXTRA])
for s in EMOTICONS_START:
    for m in EMOTICONS_MID:
        for e in EMOTICONS_END:
            RE_EMOTICON += '|{0}{1}?{2}+'.format(re.escape(s), re.escape(m), re.escape(e))

# requires ucs4 in python2.7 or python3+
# RE_EMOJI = r"""[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]"""
# safe for all python
RE_EMOJI = r"""\ud83c[\udf00-\udfff]|\ud83d[\udc00-\ude4f\ude80-\udeff]|[\u2600-\u26FF\u2700-\u27BF]"""

# List of matched token patterns, ordered from most specific to least specific.
TOKENS = [
    RE_URL,
    RE_EMAIL,
    RE_COMB,
    RE_HASHTAG,
    RE_MENTION,
    RE_HEART,
    RE_EMOTICON,
    RE_CONTRACTIONS,
    RE_TITLES,
    RE_ABBREVIATIONS,
    RE_NUM,
    RE_WORD,
    RE_SYMBOL,
    RE_EMOJI,
    RE_ANY
    ]

# List of ignored token patterns
IGNORED = [
    RE_WHITESPACE
    ]

# Final pattern
RE_PATTERN = re.compile(r'|'.join(IGNORED) + r'|(' + r'|'.join(TOKENS) + r')',
                        re.UNICODE)


def tokenize(text):
    '''Splits given input string into a list of tokens.
    # Arguments:
        text: Input string to be tokenized.
    # Returns:
        List of strings (tokens).
    '''
    result = RE_PATTERN.findall(text)

    # Remove empty strings
    result = [t for t in result if t.strip()]
    return result


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('test.csv', names=['src', 'tgt'])
    print(re.sub(RE_PUNCT, '', df['src'][0]))
    print(re.sub(RE_PUNCT, '', df['tgt'][0]))
    # text = ""Lmao nigga was selling a walker , only on 125th URL""
    # print(tokenize(text))
