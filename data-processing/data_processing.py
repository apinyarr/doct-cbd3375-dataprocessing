import boto3
from io import BytesIO
import pandas as pd
import emoji
import demoji
# Import regular expression
import re
# Import contractions
import contractions
# Import string library
import string
# Import SnowballStemmer and Word Tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk
from nltk import pos_tag
# Import Lemmatizer
from nltk.stem import WordNetLemmatizer
# Encoding the labels for isBully
from sklearn.preprocessing import LabelEncoder
# Import train_test_split
from sklearn.model_selection import train_test_split
import threading

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
bucket = 'dataset-3375-2'
file = 'cyberbullying_tweets.csv'
tweeter_df = None

def download_dataset(bucket = 'dataset-3375-2', file = 'cyberbullying_tweets.csv'):
  try:
    # os.remove('dataset.csv')
    # Get the object from S3
    s3 = boto3.resource('s3')
    with BytesIO() as data:
      s3.Bucket(bucket).download_file(file, 'dataset.csv')
  except Exception as e:
    print(e)
    print('error occurred')
    raise e
    
def load_dataframe(file = 'dataset.csv'):
  global tweeter_df 
  tweeter_df = pd.read_csv(file)
  print(tweeter_df.head(5))

def rename_feature(new_columns = {
    'tweet_text': 'text', 
    'cyberbullying_type': 'type'
}):
  global tweeter_df
  tweeter_df = tweeter_df.rename(columns = new_columns)
  print(tweeter_df.head(5))

def remove_duplicate():
  global tweeter_df
  print("Number of text before remove duplicated text is", tweeter_df.shape[0])
  tweeter_df = tweeter_df[~tweeter_df.duplicated()]
  print("Number of text after remove duplicated text is",tweeter_df.shape[0])

def remove_other_cyberbullying_type():
  global tweeter_df
  # Remove type other_cyberbullying as it impact prediction result
  tweeter_df = tweeter_df[tweeter_df['type'] != 'other_cyberbullying']
  print(tweeter_df[['type']].value_counts())

# This function return emoji
def get_emojis(text):
  return ''.join(chars for chars in emoji.distinct_emoji_list(text))

# This function convert emoji to text
def get_emoji_text(emojis):
  emoji_dict = demoji.findall(emojis)
  values = emoji_dict.values()
  return ' '.join(string for string in values)

def add_emoji_n_emojitext_feature():
  global tweeter_df
  emojis = []
  emojis_text = []
  for t in tweeter_df.text:
    emo = get_emojis(t)
    emojis.append(emo)
    emojis_text.append(get_emoji_text(emo))
  # Add emojis column to DataFram
  tweeter_df['emojis'] = emojis
  tweeter_df['emojis_text'] = emojis_text
  print(tweeter_df.head(10))

  emoji_text = len(tweeter_df[tweeter_df['emojis_text'] != ''])
  non_emoji_text = len(tweeter_df[tweeter_df['emojis_text'] == ''])
  print("Text with emoji is", emoji_text)
  print("Text without emoji is", non_emoji_text)

######### Text cleansing ##########
def remove_emoji(text):
  return emoji.replace_emoji(text, '')
#     return emoji.demojize(text, delimiters=(' ',' '))

# Regular expression to find pattern and process words
def remove_URL(text):
  return re.sub(r"((www.[^s]+)|(http\S+))","",text)

def remove_hashtag(text):
  return str(text).replace('#', '')

def remove_repeat_space(text):
  return re.sub("\s\s+", " ", text)

def remove_numeric(text):
  return re.sub('[0-9]+', '', text)

# Remove punctuation !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
def remove_punctuation(text):
#     print(f"Removing '{string.punctuation}' in text")
  table = str.maketrans('', '', string.punctuation)
  return text.translate(table)

def remove_contractions(text):
  words = text.split()
  return ' '.join([contractions.fix(word) for word in words])

############## NLP ################
# Stemming - reduce words to their root using Porter2 (snowball stemmer)
# ex, "learning" and "learner" share the root "learn"
def stemmer(text):
  # nltk.download('punkt')
  stemmer = SnowballStemmer("english")
  words = word_tokenize(text)
  return ' '.join([stemmer.stem(word) for word in words])

# Lemmatizing - reduce words to their core meaning
# ex, better => good, rocks => rock
def lemmatizer(text):
  # nltk.download('wordnet')
  # nltk.download('omw-1.4')
  # nltk.download('averaged_perceptron_tagger')
  lemmatizer = WordNetLemmatizer()
  words = word_tokenize(text)
  pt = pos_tag(words)
  lemmas = []
  for word, tag in pt:
    wordntag = tag[0].lower()
    wordntag = wordntag if wordntag in ['a', 'r', 'n', 'v'] else None
    if wordntag:
        lemma = lemmatizer.lemmatize(word, wordntag)
    else:
        lemma = word
    lemmas.append(lemma)
  return ' '.join(lemmas)

# Combine all text cleansing function
def text_cleansing(text):
  text = remove_emoji(text)
  text = remove_URL(text)
  text = remove_hashtag(text)
  text = remove_repeat_space(text)
  text = remove_numeric(text)
  text = remove_punctuation(text)
  text = remove_contractions(text)
  text = stemmer(text)
  text = lemmatizer(text)
  return text

# Perform text cleansing for the original text
def create_clean_text_feature():
  global tweeter_df
  clean_text = []
  for text in tweeter_df.text:
      clean_text.append(text_cleansing(text))
  # Add new column to the dataframe
  tweeter_df['clean_text'] = clean_text
  print(len(tweeter_df['clean_text']))
  print(tweeter_df.head(5))

def remove_duplicate_clean_text():
  # Remove duplicated cleaned text
  global tweeter_df
  # Check duplicated text
  print(tweeter_df['clean_text'].duplicated().sum())
  tweeter_df.drop_duplicates('clean_text', inplace=True)

def add_encode_type_feature():
  global tweeter_df
  labelencoder = LabelEncoder()
  # Add new encoded labels column for types
  tweeter_df['encoded_type'] = labelencoder.fit_transform(tweeter_df['type'])
  print(tweeter_df.head(5))
  # Print new column
  print(tweeter_df[['type', 'encoded_type']].value_counts())

def remove_unused_column():
  global tweeter_df
  # Drop unused column
  tweeter_df = tweeter_df.drop(['text', 'type', 'emojis', 'emojis_text'], axis=1)
  print(tweeter_df.head(5))
  tweeter_df[['encoded_type']].value_counts()

def write_dataset_to_csv():
  global tweeter_df
  output_file = 'processed_dataset.csv'
  tweeter_df.to_csv(output_file, index=False)

def upload_dataset(bucket = 'dataset-3375-2', file = 'processed_dataset.csv'):
  try:
    # Get the object from S3
    s3 = boto3.client('s3')
    response = s3.upload_file(file, bucket, 'processed_dataset.csv')
  except Exception as e:
    print(e)
    print('error occurred')
    raise e
    
thread1 = threading.Thread(target=download_dataset)
thread1.start()
thread1.join()
thread2 = threading.Thread(target=load_dataframe)
thread2.start()
thread2.join()
thread3 = threading.Thread(target=rename_feature)
thread3.start()
thread3.join()
thread4 = threading.Thread(target=remove_duplicate)
thread4.start()
thread4.join()
thread4_1 = threading.Thread(target=remove_other_cyberbullying_type)
thread4_1.start()
thread4_1.join()
thread5 = threading.Thread(target=add_emoji_n_emojitext_feature)
thread5.start()
thread5.join()
thread6 = threading.Thread(target=create_clean_text_feature)
thread6.start()
thread6.join()
thread7 = threading.Thread(target=remove_duplicate_clean_text)
thread7.start()
thread7.join()
thread8 = threading.Thread(target=add_encode_type_feature)
thread8.start()
thread8.join()
thread9 = threading.Thread(target=remove_unused_column)
thread9.start()
thread9.join()
thread9_1 = threading.Thread(target=write_dataset_to_csv)
thread9_1.start()
thread9_1.join()
thread10 = threading.Thread(target=upload_dataset)
thread10.start()
thread10.join()

# download_dataset()
# load_dataframe()
# rename_feature()
# remove_duplicate()
# remove_other_cyberbullying_type()
# add_emoji_n_emojitext_feature()
# create_clean_text_feature()
# add_encode_type_feature
# remove_duplicate_clean_text()
# write_dataset_to_csv()
# remove_unused_column()