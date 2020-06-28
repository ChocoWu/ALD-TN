import pickle
from sklearn.metrics import classification_report, confusion_matrix
from seq2seq.util.utils import *
import numpy as np
import seaborn
import pandas as pd


output_vocab = load_from_pickle('./data/tgt_vocab.pt')
print('classification label:')
print('CAG:', output_vocab.tag2id['CAG'])
print('NAG:', output_vocab.tag2id['NAG'])
print('OAG:', output_vocab.tag2id['OAG'])

fb_pred, fb_gold = pickle.load(open('./experiment/best_fb_f1_423.pickle', 'rb'))
tw_pred, tw_gold = pickle.load(open('./experiment/best_tw_f1_423.pickle', 'rb'))
print('fb_pred:\n', fb_pred[:10])
print('fb_gold:\n', fb_gold[:10])
print('tw_pred:\n', tw_pred[:10])
print('tw_gold:\n', tw_gold[:10])

print('fb classification_report:\n', classification_report(fb_gold, fb_pred, digits=5))
print('tw classification_report:\n', classification_report(tw_gold, tw_pred, digits=5))

# print(type(fb_gold))
confusion_matrix = confusion_matrix(list(fb_gold), list(fb_pred))
print(confusion_matrix)
matrix_proportions = np.zeros((3, 3))
for i in range(0, 3):
    matrix_proportions[i, :] = confusion_matrix[i, :] / float(confusion_matrix[i, :].sum())
names = ['CAG', 'OAG', 'NAG']
confusion_df = pd.DataFrame(matrix_proportions, index=names, columns=names)
plt.figure(figsize=(5, 5))
seaborn.heatmap(confusion_df, annot=True, annot_kws={'size': 12}, cmap='gist_gray_r', cbar=False,
                square=True, fmt='.2f')
plt.ylabel(r'True categories', fontsize=14)
plt.xlabel(r'Predicted categories', fontsize=14)
plt.tick_params(labelsize=12)
plt.savefig('fb_confusion.jpg')

# confusion_matrix = confusion_matrix(list(tw_gold), list(tw_pred))
# print(confusion_matrix)
# matrix_proportions = np.zeros((3, 3))
# for i in range(0, 3):
#     matrix_proportions[i, :] = confusion_matrix[i, :] / float(confusion_matrix[i, :].sum())
# names = ['CAG', 'OAG', 'NAG']
# confusion_df = pd.DataFrame(matrix_proportions, index=names, columns=names)
# plt.figure(figsize=(5, 5))
# seaborn.heatmap(confusion_df, annot=True, annot_kws={'size': 12}, cmap='gist_gray_r', cbar=False,
#                 square=True, fmt='.2f')
# plt.ylabel(r'True categories', fontsize=14)
# plt.xlabel(r'Predicted categories', fontsize=14)
# plt.tick_params(labelsize=12)
# plt.show()

