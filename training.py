import pandas as pd
from pprint import pprint
from nltk import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
import re
from numpy import mean, std
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import pickle
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import KFold
from xgboost import XGBClassifier



# import nltk


if __name__ == "__main__" :
    """
        cleaning profanity_en.csv and replacing it with profanity_en_cleaned.csv
    """

    # profanity_en = pd.read_csv("./profanity_en.csv")
    # profanity_en.drop(profanity_en.columns.difference(['text', 'severity_description']), 1, inplace=True)
    # profanity_en.reset_index()
    # try:
    #     for index, row in profanity_en.iterrows():
    #         profanity_en.at[index, 'severity_description'] = 1
    # except Exception as err:
    #     pprint(err)
    # profanity_en = profanity_en.set_axis(['text', 'profanity'], axis=1, copy=True)
    # profanity_en.to_csv('profanity_en_cleaned.csv')
    # pprint(profanity_en)

    """
        cleaning twitter_profanity.csv and replacing it with twitter_profanity_cleaned.csv
    """

    # NOT_PROFANITY = 2
    # twitterProfanity = pd.read_csv('./twitter_profanity.csv')
    # twitterProfanity.drop(twitterProfanity.columns.difference(['tweet', 'class']), 1, inplace=True)

    # try:
    #     for index, row in twitterProfanity.iterrows():
    #         if row['class'] == NOT_PROFANITY:
    #             twitterProfanity.at[index, 'class'] = 0
    #         else:
    #             twitterProfanity.at[index, 'class'] = 1
    # except Exception as err:
    #     print(err)
    # twitterProfanity = twitterProfanity.set_axis(['profanity', 'text'], axis=1, copy=True)
    # twitterProfanity = twitterProfanity[['text', 'profanity']]
    # twitterProfanity.to_csv('twitter_profanity_cleaned.csv')

    """
        Merging the previously made dataFrames to one globale one. and shuffle it :D
    """
    # twitterProfanity = pd.read_csv('./twitter_profanity_cleaned.csv')
    # profanity_en = pd.read_csv('./profanity_en_cleaned.csv')
    # globalProfanity = pd.concat([twitterProfanity, profanity_en], axis=0)
    # globalProfanity = globalProfanity.sample(frac=1)
    # globalProfanity = globalProfanity.set_axis(['id', 'text', 'profanity'], axis=1, copy=True)
    # globalProfanity = globalProfanity[['text', 'profanity']]
    # globalProfanity.to_csv('profanity.csv', index=False)
    # pprint(globalProfanity)


    """
        Tokenize the sentences, and remove emoticons, Html Tags, Twitter mentiones, RT symbole...
    """
    # nltk.download('punkt')

    # globalProfanity = pd.read_csv('./profanity.csv')

    # for index, row in globalProfanity.iterrows():
    #     urlRegex = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)')
    #     htmlTagsRegex = r'&[-a-zA-Z0-9@:%._\+~#=]{1,256};'
    #     text = row['text']
    #     text = " ".join([re.sub(htmlTagsRegex, '', word) for word in text.split(' ') if not urlRegex.search(word.lower()) and word != 'RT'])
    #     tokenizedText = word_tokenize(text)
    #     text = " ".join([word.lower() for word in tokenizedText if word.lower() not in ENGLISH_STOP_WORDS])
    #     globalProfanity.at[index, 'text'] = text
    
    # globalProfanity.to_csv('profanity_clean_with_ponctuation.csv', index=False)


    """
        Moment of truth, creating a TFIDF sparse Matrix and 
        splitting the dataset to training(75%) and evaluation(25%)
        pickling all models for further use !
    """

    # profanity_clean = pd.read_csv('./profanity_clean_with_ponctuation.csv')
    # tfidf_vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents="ascii")

    # y = profanity_clean.profanity
    # X = tfidf_vectorizer.fit_transform(profanity_clean.text)

    # X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.25, random_state=1)

    # classifier = SVC(probability=True)
    # classifier.fit(X_train, y_train)

    # classifier_fd = open("SVC.pickle", "wb")
    # vectorizer_fd = open("TFIDF_vectorizer.pickle", "wb")
    # tfidf_vectorizer.fit(profanity_clean.text)

    # pickle.dump(classifier, classifier_fd)
    # pickle.dump(tfidf_vectorizer, vectorizer_fd)

    # classifier_fd.close()
    # vectorizer_fd.close()
    # print(classifier.score(X_eval, y_eval) * 100)




    """
        Testing with randome sentences
    """
    # classifier_fd = open('SVC.pickle', 'rb')
    # over_classifier_fd = open('over_SVC.pickle', 'rb')
    # vectorizer_fd = open('TFIDF_vectorizer.pickle', 'rb')
    # over_classifier = pickle.load(over_classifier_fd)
    # classifier = pickle.load(classifier_fd)
    # vectorizer = pickle.load(vectorizer_fd)
    # test = [("visit Estonia")]
    # test_vector = vectorizer.transform(test)
    # # the first is 0 and second is 1
    # print(classifier.predict_proba(test_vector))
    # print(over_classifier.predict_proba(test_vector))

    # classifier_fd.close()
    # vectorizer_fd.close()

    """
        There is a problem, i found out that we have kind of unbalanced dataset with :
        22218: entry for profanity positive entries
        4163: entry for profanity negative entries
        and this will make the model accurate with profanity+ prediction but less with profanity- ones.
    """

    # profanity_clean = pd.read_csv('./profanity_clean_with_ponctuation.csv')
    # vectorizer_fd = open('TFIDF_vectorizer.pickle', 'rb')
    # vectorizer = pickle.load(vectorizer_fd)

    # print(Counter(profanity_clean.profanity))

    # oversample = SMOTE()
    # over_X, over_y = oversample.fit_resample(vectorizer.transform(profanity_clean.text), profanity_clean.profanity)

    # print(Counter(over_y))

    # X_train, X_eval, y_train, y_eval = train_test_split(over_X, over_y, test_size=0.25, random_state=1)

    # over_classifier = SVC(probability=True)
    # over_classifier.fit(X_train, y_train)

    # over_classifier_fd = open('over_SVC.pickle', 'wb')
    # pickle.dump(over_classifier, over_classifier_fd)
    # over_classifier_fd.close()

    # classifier_fd = open('SVC.pickle', 'rb')
    # classifier= pickle.load(classifier_fd)

    # print(classifier.score(X_eval, y_eval) * 100)
    # print(over_classifier.score(X_eval, y_eval) * 100)


    """
        Model Validation, KFold cross validation resulted in 98.1% Accuracy.
    """
    # profanity_clean = pd.read_csv('./profanity_clean_with_ponctuation.csv')
    # vectorizer_fd = open('TFIDF_vectorizer.pickle', 'rb')
    # vectorizer = pickle.load(vectorizer_fd)

    # oversample = SMOTE()
    # over_X, over_y = oversample.fit_resample(vectorizer.transform(profanity_clean.text), profanity_clean.profanity)

    # X_train, X_eval, y_train, y_eval = train_test_split(over_X, over_y, test_size=0.25, random_state=1)

    # over_classifier = XGBClassifier()
    # over_classifier.fit(X_train, y_train)
    # print(over_classifier.score(X_eval, y_eval) * 100)
    # pickle.dump(over_classifier, open('over_XGBClassifier.pickle', 'wb'))
    # kf = KFold(n_splits=10, random_state=1, shuffle=True)
    # scores = cross_val_score(over_classifier, over_X, over_y, scoring='accuracy', cv=kf, n_jobs=-1)
    # print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


    """
        I tested probabily only with SVC model till the moment, testing with other models
    """
    # profanity_clean = pd.read_csv('./profanity_clean_with_ponctuation.csv')
    # vectorizer_fd = open('TFIDF_vectorizer.pickle', 'rb')
    # vectorizer = pickle.load(vectorizer_fd)

    # oversample = SMOTE()
    # over_X, over_y = oversample.fit_resample(vectorizer.transform(profanity_clean.text), profanity_clean.profanity)

    # classifier_fd = open('over_DecisionTreeClassifier.pickle', 'rb')
    # over_classifier = pickle.load(classifier_fd)

    # test = [("good day wonderful nigga")]
    # test_vector = vectorizer.transform(test)
    # # the first is 0 and second is 1
    # print(over_classifier.predict_proba(test_vector))



    """
        Testing the voting classifier, it actually give some pretty good result, 97.94% accuracy
    """
    # profanity_clean = pd.read_csv('./profanity_clean_with_ponctuation.csv')
    # vectorizer_fd = open('TFIDF_vectorizer.pickle', 'rb')
    # vectorizer = pickle.load(vectorizer_fd)
    # estimators = []
    # estimators.append(('MuNo', MultinomialNB()))
    # estimators.append(('Ber', BernoulliNB()))
    # estimators.append(('Knear', KNeighborsClassifier()))
    # estimators.append(('LR', LogisticRegression()))
    # estimators.append(('DT', DecisionTreeClassifier()))
    # estimators.append(('RF', RandomForestClassifier()))
    # estimators.append(('SVC', SVC(probability=True)))
    # estimators.append(('XGB', XGBClassifier()))

    # oversample = SMOTE()
    # over_X, over_y = oversample.fit_resample(vectorizer.transform(profanity_clean.text), profanity_clean.profanity)

    # X_train, X_eval, y_train, y_eval = train_test_split(over_X, over_y, test_size=0.25, random_state=1)
    # vot_soft = VotingClassifier(estimators=estimators, voting='soft')
    # vot_soft.fit(over_X, over_y)
    # pickle.dump(vot_soft, open('over_VotingClassifier.pickle', 'wb'))
    # y_pred = vot_soft.predict(X_eval)

    # print(accuracy_score(y_eval, y_pred))
