from tkinter import Label
from tkinter.font import names
from numpy import argmax
import pandas as pd
from flair.models import TARSClassifier
from flair.data import Sentence
from transformers import  pipeline



def read_test_labels():
    my_file = open("/Users/shirinwadood/Desktop/projects/Transformer/tweeteval-main/datasets/emotion/test_labels.txt", "r")
    data = my_file.read()
    label_data_into_list = data.split("\n")
    #print(data_into_list)
    my_file.close()
    return label_data_into_list


# To read the test texts from the data sets and find the pred labels
def read_test_text():
    my_file = open("/Users/shirinwadood/Desktop/projects/Transformer/tweeteval-main/datasets/emotion/test_text.txt", "r")
    data = my_file.read()
    text_data_into_list = data.split("\n")
    # print(data_into_listclear)
    my_file.close()
    return text_data_into_list


def get_prediction_information():
    classes = ["anger", "joy", "optimism",'sadness']
    sentences = [Sentence(text) for text in test_text]
    # sentence = Sentence(test_text)
    classifier.predict_zero_shot(sentences, classes)
    return sentences


if __name__ == "__main__":


    true_labels=read_test_labels()

    true_label_mapping= {"0":'anger', "1":'joy', "2":'optimism', "3":'sadness'}
    true_string_labels=[]
    for label in true_labels:
        true_string_labels.append(true_label_mapping[label])

    test_text=read_test_text()
    test_text = test_text
    classifier = TARSClassifier.load("tars-base")
    classifier_outputs = get_prediction_information()
    sentence = Sentence("I am so glad to use Flair")
    classes = ["happy", "sad"]
    classifier.predict_zero_shot(sentence, classes)
    print(sentence)
    max_score_indices=[]
    pred_labels=[]
    print(type(classifier_outputs))
    print(classifier_outputs)
    for classifier_output in classifier_outputs:
        sent_dict = classifier_output.to_dict()
        label_dict_list=sent_dict['all labels']
        label_score = 0
        label = ''

        for dict in label_dict_list:
            if dict['confidence'] > label_score:
                label = dict['value']
                label_score = dict['confidence']

        pred_labels.append(label)

    #checking the number of  predlabel matches with true labels:
    count=0
    for i,j in zip(true_string_labels,pred_labels):
            if i==j:
                count+=1
    print(count)

    coloumn_names= ['Text sentence', 'True label', 'Pred label']
    df=pd.DataFrame([test_text,true_string_labels, pred_labels])
    df=df.T
    df.columns=coloumn_names
    print(df)
    df.to_csv("./Tweet_eval_TARS", sep="~", index=False)
    print("Awesome job!, You've done it!")