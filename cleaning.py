
import re
from nltk import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(textDataframe):
    globalProfanity = textDataframe
    for index, row in globalProfanity.iterrows():
        urlRegex = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)')
        htmlTagsRegex = r'&[-a-zA-Z0-9@:%._\+~#=]{1,256};'
        text = row['text']
        text = " ".join([re.sub(htmlTagsRegex, '', word) for word in text.split(' ') if not urlRegex.search(word.lower()) and word != 'RT'])
        tokenizedText = word_tokenize(text)
        text = " ".join([word.lower() for word in tokenizedText if word.lower() not in ENGLISH_STOP_WORDS])
        globalProfanity.at[index, 'text'] = text
    return globalProfanity