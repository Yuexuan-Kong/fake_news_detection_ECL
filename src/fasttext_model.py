"""
Training script using fastText
"""
import csv
import fasttext
import numpy as np

def process_csv(original_df, output_name, output_folder,col_text='tweet',col_label='label'):
    df = original_df.copy()
    df.loc[:, 'text'] = '__label__' + df[col_label].astype(str) + ' ' + df[col_text]
    output_file = f'{output_folder}/{output_name}.txt'
    df[['text']].to_csv(output_file, index=False, header=False, 
                quoting=csv.QUOTE_NONE,  quotechar="",  escapechar="\\")
    return output_file


def train_model(file_path,**kwargs):
    model = fasttext.train_supervised(
        input=file_path, **kwargs)
    return model

def get_model_accuracy(model,file_path:str):

    results = model.test(file_path)
    accuracy = results[1]
    return accuracy

def predict_model(model,input,return_prob=False):

    input = list(map(lambda x:x.replace("\n",""),input))
    prediction = model.predict(input)
    labels = np.array(prediction[0]).ravel()
    probs = list(np.array(prediction[1]).ravel())
    labels = list(map(lambda x:int(x[-1]),labels))
    if return_prob:
        return labels,probs
    else:
        return labels