import pandas as pd
import re


data_path = "../data/"
data_path_ectf = data_path + "ECTF_dataset/"
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

# pre-process texts of data from ECTF
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

all_df_ECTF["text"] = all_df_ECTF["text"].apply(lambda row : re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", row)) # no URL
all_df_ECTF["text"] = all_df_ECTF["text"].apply(lambda row : re.sub(r"RT ", "", row)) # no @
all_df_ECTF["text"] = all_df_ECTF["text"].apply(lambda row : emoji_pattern.sub(r"", row)) # no emoji
all_df_ECTF = all_df_ECTF.rename(columns={"text" : "tweet"})


all_df["tweet"] = all_df["tweet"].apply(lambda row : re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", row)) # no URL
all_df["tweet"] = all_df["tweet"].apply(lambda row : re.sub(r"RT ", "", row)) # no @
all_df["tweet"] = all_df["tweet"].apply(lambda row : emoji_pattern.sub(r"", row)) # no emoji
#####################
## concat all data ##
#####################
all_df_2db = pd.concat([all_df, all_df_ECTF]).drop_duplicates()
all_df_2db.reset_index(inplace = True)
all_df_2db.drop(columns = ['index'],inplace =True)
print(all_df_2db)
all_df_2db.to_csv(data_path + 'database.csv')
print("SHAPE OF DATABASE : ", "\n", all_df_2db.shape, "\n", "EXAMPLES OF DATABASE : ", "\n", all_df_2db.head(5))