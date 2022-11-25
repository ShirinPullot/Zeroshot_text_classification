import pandas as pd
from numpy import argmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline



def read_test_labels():
    my_file = open("/Users/shirinwadood/Desktop/projects/Transformer/tweeteval-main/datasets/emotion/test_labels.txt", "r")
    data = my_file.read()
    data_into_list = data.split("\n")
    print(data_into_list)
    my_file.close()
    return data_into_list
    


def read_test_text():
    my_file = open("/Users/shirinwadood/Desktop/projects/Transformer/tweeteval-main/datasets/emotion/test_text.txt", "r")
    data = my_file.read()
    data_into_list = data.split("\n")
    # print(data_into_listclear)
    my_file.close()
    return data_into_list


def get_prediction_information():
        candidate_labels = ["anger", "joy", "optimism",'sadness']
        classifier_outputs = classifier(test_text, candidate_labels)
        return classifier_outputs


if __name__ == "__main__":

    true_labels=read_test_labels()
    true_label_mapping= {"0":'anger', "1":'joy', "2":'optimism', "3":'sadness'}
    true_string_labels=[]
    for label in true_labels:
        true_string_labels.append(true_label_mapping[label])

    

    test_text=read_test_text()
    classifier = pipeline('zero-shot-classification',model='cross-encoder/nli-distilroberta-base')
    classifier_outputs = get_prediction_information()
    pred_labels=[]
    for classifier_output in classifier_outputs:
        pred_labelstring=(classifier_output['labels'][0])
        #print(pred_labelstring)
        pred_labels.append(pred_labelstring)
    #print(pred_labels)
   
    #checking the number of  predlabel matches with true labels:
    count=0
    for i,j in zip(true_string_labels,pred_labels):
        if i==j:
            count+=1
    print(count)


    df=pd.DataFrame([test_text,true_string_labels, pred_labels])
    coloumn_names= ['Text sentence', 'True label', 'Pred label']
    df=df.T
    df.columns=coloumn_names
    print(df)
    df.to_csv("./cross encoder", sep="~", index=False)
    print("Awesome job!, You've done it!")
    