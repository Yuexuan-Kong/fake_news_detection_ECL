import re
import spacy
import pandas as pd

# TODO : add stemming or embedding to reduce the dimensions
def remove_special_characters(df):
    df['preprocess'] = df.apply(lambda row: row['tweet'].replace("\n", " "),
                                axis=1)  # remove new line character
    df["preprocess"] = df["preprocess"].apply(lambda row: re.sub(r"RT ", "", row).lower())  
    # no RT@ and lowercase, this has to be executed first, otherwise rt in lowercase will be removed as well
    df["preprocess"] = df["preprocess"].apply(lambda row: re.sub(r"#", "", row))  # no# #
    df["preprocess"] = df["preprocess"].apply(lambda row: re.sub(r"_", "", row))  # no_ #
    df["preprocess"] = df["preprocess"].apply(
        lambda row: re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", row))  # no URLs

    emoji_pattern = re.compile("["
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
    
    df["preprocess"] = df["preprocess"].apply(lambda row: emoji_pattern.sub(r"", row))  # no emoji
    df['preprocess'] = df.apply(lambda row: re.sub('  +', ' ', row['preprocess']).strip(),
                                                axis=1)  # remove all additional spaces
    return df

def remove_stopwords(df):
    en = spacy.load("en")
    stop = en.Defaults.stop_words
    # remove characters which could have semantical meaning from stopwrods list
    for element in ["not", "no", "never", "don't", "won't", "couldn't", "neither"]:
        stop.discard(element)
    # add _ and hashtags in the stopwords list
    for element in ["_", "#"]:  # remove _ and hashtags
        stop.add(element)
    df['preprocess'] = df['preprocess'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop)]))  # remove all stopwords
    return df

def preprocess(df):
    remove_special_characters(df)
    remove_stopwords(df)
    return df