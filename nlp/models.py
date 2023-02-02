from django.db import models

import pandas as pd
import numpy as np
import re # re: regular expression
from nltk.corpus import stopwords # nltk natural language toolkit
from nltk.tokenize import WordPunctTokenizer # noktalama işaretleriyle tokenize işlemi
from nltk.stem.porter import PorterStemmer # stemming
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from rake_nltk import Rake # keyword extraction
import spacy # similarity 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import io

class Dataset(models.Model):
    
    # 1. Veri setinin okunması ----------------------------------------------------------------------------

    global df
    df = pd.read_csv('tweets.csv')
    
    df.info()
    buf = io.StringIO()
    df.info(buf=buf)
    info = buf.getvalue().split('\n')[0:-2]
    info = pd.DataFrame(info, columns=['Bilgiler'])
    blankIndex=[''] * len(info)
    info.index=blankIndex
    info = info.head(20).style.set_table_styles(
        #     [{"selector": "", "props": [("border", "3px solid grey")]},
            [{"selector": "tbody td", "props": [("border", "")]},
        #     {"selector": "th", "props": [("border", "2px solid grey")]}
            ])
    type(info)

    dataset = df.head(20).style.set_table_styles(
            [{"selector": "", "props": [("border", "3px solid grey")]},
            {"selector": "tbody td", "props": [("border", "2px solid grey")]},
            {"selector": "th", "props": [("border", "2px solid grey")]}
            ])

    # 2. Preprocessing -----------------------------------------------------------------------------------

    # verilen patternle eşleşen kelimeleri çıkaran fonksiyon
    global remove_pattern
    def remove_pattern(input_text, pattern):
        r = re.findall(pattern, input_text)
        for word in r:
            input_text = re.sub(word, "", input_text)
        return input_text

    # @username ler temizlenmesi
    # temizlenmiş tweetler datasete eklenir
    df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")

    not_user_dataset = df.head(20).style.set_table_styles(
            [{"selector": "", "props": [("border", "3px solid grey")]},
            {"selector": "tbody td", "props": [("border", "2px solid grey")]},
            {"selector": "th", "props": [("border", "2px solid grey")]}
            ])
    
    # tamamen temizlenmiş veri seti ---------
    # tokenize
    df['tweet_token'] = df['clean_tweet'].apply(lambda x: WordPunctTokenizer().tokenize((x.lower())))

    # noktalama işaretlerinin kaldırılması
    global punctuation
    punctuation = '''!()-[]{};':",<>./?@$%^&*_~'''
    df['tweet_token'] = df['tweet_token'].apply(lambda x: " ".join([token for token in x if not token in punctuation]))

    # tokenize
    df['tweet_token'] = df['tweet_token'].apply(lambda x: WordPunctTokenizer().tokenize((x.lower())))

    # stopwordlerin kaldırılması
    global stop_words
    stop_words = set(stopwords.words('english'))
    df['clean_tweet'] = df['tweet_token'].apply(lambda x: " ".join([word for word in x if not word in stop_words]))

    # hashtag lerin düzeltilmesi '# ' -> '#'
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: x.replace("# ", "#"))

    # non-ascii karakterlerin temizlenmesi
    global removeNonAscii
    def removeNonAscii(s):
        '''
        Removes non-ascii characters from database terms (as some downloaded
        information has special characters which cause errors)
        '''
        return "".join(i for i in s if ord(i) < 128)
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: removeNonAscii(x))

    clean_dataset = df.head(20).style.set_table_styles(
            [{"selector": "", "props": [("border", "3px solid grey")]},
            {"selector": "tbody td", "props": [("border", "2px solid grey")]},
            {"selector": "th", "props": [("border", "2px solid grey")]}
            ])
    
    # stemming yapılmış veri seti -----------
    # tokenize
    df['tweet_token'] = df['clean_tweet'].apply(lambda x: WordPunctTokenizer().tokenize((x.lower())))

    # stemming
    global stemmer
    stemmer = PorterStemmer()
    df['clean_tweet'] = df['tweet_token'].apply(lambda sentence: " ".join([stemmer.stem(word) for word in sentence]))

    # hashtag lerin düzeltilmesi '# ' -> '#'
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: x.replace("# ", "#"))

    stemming_dataset = df.head(20).style.set_table_styles(
            [{"selector": "", "props": [("border", "3px solid grey")]},
            {"selector": "tbody td", "props": [("border", "2px solid grey")]},
            {"selector": "th", "props": [("border", "2px solid grey")]}
            ])

    # 3. Eğitim ve Test -----------------------------------------------------------------------------------

    # feature extraction
    # max_df=0.90, min_df=2, max_features=1000, stop_words='english'
    global bow_vectorizer
    bow_vectorizer = CountVectorizer()
    bow = bow_vectorizer.fit_transform(df['clean_tweet'])

    x_train, x_test, y_train, y_test = train_test_split(bow, df['label'], random_state=42, test_size=0.15)
    global model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    # testing
    pred = model.predict(x_test)
    score_f1 = f1_score(y_test, pred)
    accuracy = accuracy_score(y_test,pred)
    
    # 4. Keyword Extraction ve Similarity -----------------------------------------------------------------------------------

    # Rake -> keyword extraction
    global model_Rake
    model_Rake = Rake()
    # spacy -> similarity
    global nlp
    nlp = spacy.load("en_core_web_lg")

    # keywords
    racist_words = 'race covert colourism colorism racism diversity inclusion black hate racist white nigger'
    sexist_words = " gender inequality misogyny patriarchy sexism sexist objectification harassment discrimination "

    global racist
    global sexist
    racist = nlp(racist_words)
    sexist = nlp(sexist_words)

    # similarity fonksiyonları
    global similarity
    def similarity(nlp_choice, text):
        model_Rake.extract_keywords_from_text(text)
        extract = model_Rake.get_ranked_phrases_with_scores()
        tweet_keyword = nlp(extract[0][1])
        return nlp_choice.similarity(tweet_keyword)

    def similarity_tweet(nlp_choice, tweet_index):  
        return similarity(nlp_choice, df['clean_tweet'][tweet_index])

    # nefret söylemi tipi tespit fonksiyonları
    # if hate == 1
    global find_hate_type
    def find_hate_type(text):
        r = similarity(racist, text)
        s = similarity(sexist, text)
        if (r < 0.5) & (s < 0.5):
        # undefined -> seçilen tweet nefret söylemi içermiyor olabilir
           return 'other'
        elif r > s:
           return 'racist'
        else:
           return 'sexist'

    # if hate == 1
    def find_hate_type_tweet(tweet_index):
        return find_hate_type(df['clean_tweet'][tweet_index])

    # 5. Veri Seti Bilgileri -----------------------------------------------------------------------------------

    # toplam, nefret söylemi içeren eve içermeyen tweet sayıları
    total_num = df['tweet'].count()
    hate = df[df.label == 1].count()
    hate_num = hate['tweet']
    not_hate = df[df.label == 0].count()
    not_hate_num = not_hate['tweet']

    # 6. Örnekler -----------------------------------------------------------------------------------

    # örnek tweet
    def example():
        df1 = df[df.label == 1].sample(1)
        example_tweet = df1['clean_tweet'].to_string().split(maxsplit=1)[1]
        example_sim_rac = similarity(racist, example_tweet)
        example_sim_sex = similarity(sexist, example_tweet)
        hate_type = find_hate_type(example_tweet)
        return example_tweet, example_sim_rac, example_sim_sex, hate_type
    
    # kullanıcı tweeti
    def user_example(tweet):
        original_tweet = tweet

        example_tweet = remove_pattern(tweet, "@[\w]*")
        example_tweet = WordPunctTokenizer().tokenize((example_tweet.lower()))
        example_tweet = " ".join([token for token in example_tweet if not token in punctuation])
        example_tweet = WordPunctTokenizer().tokenize((example_tweet.lower()))
        example_tweet= " ".join([word for word in example_tweet if not word in stop_words])
        example_tweet = example_tweet.replace("# ", "#")
        example_tweet = WordPunctTokenizer().tokenize((example_tweet.lower()))
        example_tweet = " ".join([stemmer.stem(word) for word in example_tweet])
        example_tweet = example_tweet.replace("# ", "#")

        # example_tweet = pd.Series(example_tweet)
        # bow_vectorizer = CountVectorizer()
        # bow = bow_vectorizer.fit_transform(example_tweet)

        # pred = model.predict(bow)[0]
        pred = model.predict(bow_vectorizer.transform([example_tweet]))[0]
        example_sim_rac = similarity(racist, example_tweet)
        example_sim_sex = similarity(sexist, example_tweet)

        hate_type = find_hate_type(example_tweet)
        if hate_type == 'racist' or hate_type =='sexist':
            pred = 1
        if pred == 0:
            hate_type = "undefined"

        return pred, original_tweet, example_tweet, example_sim_rac, example_sim_sex, hate_type