import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

import re
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text

def preprocessing_function(text: str) -> str:
    lastly = ["sunday", "moday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
    preprocessed_text = remove_stopwords(text)
    preprocessed_text = preprocessed_text.lower()
    preprocessed_text = re.sub(r'[^a-z\s]', '', preprocessed_text)
    word_tokens = preprocessed_text.split()
    stems = [stemmer.stem(word) for word in word_tokens]
    preprocessed_text = ' '.join(stems)
    for x in range(len(lastly)):
        preprocessed_text = preprocessed_text.replace(lastly[x], "")
    return preprocessed_text
