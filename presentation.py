import pickle
import pandas as pd
from cleaning import clean_text

def predict(text):
    classifier = pickle.load(open('./models/over_VotingClassifier.pickle', 'rb'))
    vectorizer = pickle.load(open('./models/TFIDF_vectorizer.pickle', 'rb'))
    test = {'text': [text]}
    test_dataframe = pd.DataFrame(data=test)
    clean_test = clean_text(test_dataframe)
    test_vector = vectorizer.transform(clean_test['text'])
    prediction = classifier.predict_proba(test_vector)
    print(prediction)
    print("Yes : {:.2f}%\nNo : {:.2f}%".format(prediction[0][1] * 100, prediction[0][0] * 100))


if __name__ == "__main__":
    predict("Hello sunshine i hope your doing well")