import re
import string
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_urls(text):
    return re.sub(r'http\S+|www\.\S+', '', text)
    
def remove_mentions(text):
    return re.sub(r'@[A-Za-z0-9_]+', '', text)

def remove_hasgtags(text):
    return re.sub(r'#[A-Za-z0-9_]+', '', text)

def remove_emoji(text):
    return emoji.replace_emoji(text, replace='')

def remove_html(text):
    return re.sub(r'&[a-zA-Z]+;', '', text)

# string.punctuation = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ 
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_number(text):
    return re.sub(r'\d+', '', text)

def clean_tweet(text):
    
    # step1: remove URL, @, #
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hasgtags(text)
    
    # step2: remove emoji, htlm, punctuation
    text = remove_emoji(text)
    text = remove_html(text)
    text = remove_punctuation(text)

    # step3: remove numbers
    text = remove_number(text)
    
    # step3: lower
    text = text.lower()

    #  step4: tokenizer e remove stopwords
    stop_words = set(stopwords.words('english'))  
    words = word_tokenize(text)
    text = ' '.join([word for word in words if word not in stop_words])
    
    return text


