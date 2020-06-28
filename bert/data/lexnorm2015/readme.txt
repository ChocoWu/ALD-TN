By using the data, You agree to use the data following Twitter and the shared task guideline.

Note the data may contain coarse/offensive languages.

If you have any questions or correction suggestions for the dataset, please post it on the google forum https://groups.google.com/forum/#!members/lexical-normalisation-for-english-tweets

Please cite the following paper:
Baldwin, Timothy, Marie-Catherine de Marneffe, Bo Han, Young-Bum Kim, Alan Ritter and Wei Xu (2015) Shared Tasks of the 2015 Workshop on Noisy User-generated Text: Twitter Lexical Normalization and Named Entity Recognition, in Proceedings of the ACL 2015 Workshop on Noisy User-generated Text (W-NUT), Beijing, China, pp. 126—135.



Description:
To make the task of text normalisation tractable, this shared task focuses on context-sensitive lexical normalisation of English Twitter messages, under the following constraints:

Non-standard words (NSWs) are normalised to one or more canonical English words based on a pre-defined lexicon. For instance, l o v e should be normalised to love (many-to-one normalisation), tmrw to tomorrow (one-to-one normalisation), and cu to see you (one-to-many normalisation). Additionally, IBM should be left untouched as it is in the lexicon and in its canonical form, and the informal lol should be expanded to laughing out loud.
Non-standard words may be either out-of-vocabulary (OOV) tokens (e.g., tmrw for tomorrow) or in-vocabulary (IV) tokens (e.g., wit for with in I will come wit you).
Only alphanumeric tokens (e.g., 2, 4eva and tmrw) and apostrophes used in contractions (e.g., yoou've) are considered for normalisation. Tokens including hyphens, single quotes and other types of contractions should be ignored.
Domain specific entities are ignored even if they are in non-standard forms, e.g., #ttyl, @nyc
It is possible for a tweet to have no non-standard tokens but still require normalisation (e.g. our example of wit above), and also for the tweet to require no normalisation whatsoever.
Proper nouns shall be left untouched, even if they are not in the given lexicon (e.g., Twitter).
All normalisations should use American spelling (e.g., tokenize rather than tokenise).
In cases where human annotators have been unable to determine whether a given token is a non-standard word or its normalised (OOV) form, we have chosen to be conservative and leave the token unchanged.
A more detailed set of annotation guidelines is provided here

For your convenience and consistency of evaluation, we have pre-tokenised tweets and provided them in JSON format. The training data file is a JSON list in which each item represents a tweet. A tweet is a JSON dict containing four fields: index (the ID for annotation), tid (tweet ID), input (a list of case-sensitive tokens to be normalised), and output (a list of normalised tokens, in lowercase). The test data for evaluation follows the same format as the training data, but it does NOT have output fields: your task is to automatically predict the output fields. Note that all tokens in the output field for a given tweet should be in lowercase.

A mock-up sample JSON object is provided below, for illustrative purposes. Jst, lol and l o v e are normalised to just, laughing out loud and love, respectively. 
{ 'tid': '971011879910802432', 'index': '1064', 'input': [ 'Jst', 'read', 'a', 'tweet', 'lol', 'and', 'l', 'o', 'v', 'e', 'it' ], 'output': [ 'just', 'read', 'a', 'tweet', 'laughing out loud', 'and', 'love', '', '', '', 'it' ] }

For evaluation, we will use the evaluation metrics of Precision, Recall and F1. Two categories of submission will be accepted:

Task 1: Constrained systems (cm) 
Participants can only use the provided training data to perform the text normalisation task, but are able to make use of any off-the-shelf tools (e.g., Twitter POS taggers). Normalisaiton lexicons and extra tweet data shall NOT be used in the constrained system.
Task 2: Unconstrained systems (um) 
Participants can use any publicly accessible data and tools to perform the text normalisation task.

