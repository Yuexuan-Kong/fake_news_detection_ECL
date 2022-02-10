import re
import spacy
import pandas as pd
from sklearn.model_selection import train_test_split

# Global variables used in functions

EMOJI_PATTERN = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)

# --------------------------------------------------------------

def remove_retweets(text):

    return re.sub(r"RT\s@([a-zA-Z0-9_]+:)", "", text)

def count_retweets(text):

    return len(re.findall(r"RT\s@([a-zA-Z0-9_]+:)",text))

def remove_mentions(text):

    return re.sub(r"@([a-zA-Z0-9_]+)",'',text)

def count_mentions(text):

    return len(re.findall(r"@([a-zA-Z0-9_]+)",text))

def remove_urls(text):

    return re.sub(r"(http?\://|https?\://|www)\S+", '',text)

def count_urls(text):

    return len(re.findall(r"(http?\://|https?\://|www)\S+",text))

def remove_hashtags(text):

    return re.sub(r"#", "", text)

def count_hashtags(text):

    return len(re.findall(r"#",text))

def remove_additional_space(text):

    return re.sub('  *', ' ',text)

def replace_slash_chars_by_space(text):

    res = text.replace(
        "\n"," "
        ).replace(
            "\r"," "
        ).replace(
            "\t"," "
        )
    return res

def remove_underscore(text):

    return re.sub(r"_", "", text)

def remove_emojis(text):

    return EMOJI_PATTERN.sub(r"", text)

def remove_stopwords(text,list_stopwords):

    res = ' '.join(
        [word for word in text.split() if word not in list_stopwords]
        )
    return res

def to_lowercase(text):

    return text.lower()
    

def train_val_split(df,val_size=0.1,rd_state=42):

    train_df, val_df = train_test_split(df,test_size=val_size,random_state=rd_state)
    return train_df, val_df