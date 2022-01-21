import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import spacy


data_path = "../data/"
data_path_ectf = data_path + "ECTF_dataset/"
data_path_claims = data_path + "Claims/"

###############
## read data ##
###############

# read all data of dataset covid_fake_new
train_df = pd.read_csv(data_path+"Constraint_Train.csv").drop(columns=["id"],axis = 1) # tweet, label
val_df = pd.read_csv(data_path+"Constraint_Val.csv").drop(columns=["id"],axis = 1)# tweet, label
test_with_label_df = pd.read_csv(data_path+"english_test_with_labels.csv").drop(columns=["id"],axis = 1) # tweet, label
all_df = pd.concat([train_df, val_df, test_with_label_df])

# read all data of dataset ECTF
fake_df = pd.read_csv(data_path_ectf+"fake.csv", index_col=[0]).drop(columns=["id"],axis = 1)
fake_df["label"] = "fake"
genuine_df = pd.read_csv(data_path_ectf+"genuine.csv", index_col=[0]).drop(columns=["id"],axis = 1)
genuine_df["label"] = "real"
all_df_ECTF = pd.concat([fake_df, genuine_df])
all_df_ECTF = all_df_ECTF.rename(columns={"text" : "tweet"})

# read claims
fake_claim_df = pd.read_csv(data_path_claims+"fake_claim.csv", index_col=[0]).drop(["fact_check_url", "news_url"],axis = 1)
real_claim_df = pd.read_csv(data_path_claims+"real_claim.csv", index_col=[0]).drop(["fact_check_url", "news_url"],axis = 1)
fake_claim_df["label"] = "fake"
real_claim_df["label"] = "real"
all_claim = pd.concat([fake_claim_df, real_claim_df])
all_claim = all_claim.rename(columns={"title" : "tweet"})






#####################
## concat all data ##
#####################
all_df_3db = pd.concat([all_df, all_df_ECTF, all_claim]).drop_duplicates()
all_df_3db.to_csv('all_data.csv')

#########################
## pre-processing data ##
#########################
en = spacy.load("en")
stop = en.Defaults.stop_words
for element in ["not", "no", "never", "don't", "won't", "couldn't", "neither"]:
    stop.discard(element)
for element in ["_", "#"]: # remove _ and hashtags
    stop.add(element)
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



all_df_3db['preprocess'] = all_df_3db.apply(lambda row: row['tweet'].replace("\n"," "), axis=1) # remove new line character
all_df_3db["preprocess"] = all_df_3db["preprocess"].apply(lambda row : re.sub(r"RT ", "", row).lower()) # no @
all_df_3db["preprocess"] = all_df_3db["preprocess"].apply(lambda row : re.sub(r"#", "", row)) # no# #
all_df_3db["preprocess"] = all_df_3db["preprocess"].apply(lambda row : re.sub(r"_", "", row)) # no_ #

all_df_3db["preprocess"] = all_df_3db["preprocess"].apply(lambda row : re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", row)) # no URL
all_df_3db["preprocess"] = all_df_3db["preprocess"].apply(lambda row : emoji_pattern.sub(r"", row)) # no emoji
all_df_3db['preprocess'] = all_df_3db.apply(lambda row: re.sub('  +', ' ',row['preprocess']).strip(), axis=1) # remove all additional spaces


all_df_3db['preprocess'] = all_df_3db['preprocess'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))# remove all stopwords

all_df_3db.to_csv(data_path+"all_preprocess.csv")

