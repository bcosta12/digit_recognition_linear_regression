import pandas as pd
import numpy as np
from math import log

def train_linear_regression( label_train, data_train_bi ):


def apply_linear_regression( train, priors, test_dataset_bi ):


def  extract_terms_from_dosc (test_dataset_bi ):

    results = [[0]*2 for i in range( len( test_dataset_bi ) )]
    score = [0]*10
    return results, score

def extract_vocabulary_and_count_docs( label_train, data_train_bi ):
    train = [[0]*785 for _ in range(10)]
    priors = [0]*10
    #Count Vocabulaty and Docs
    for i in range( len( label_train ) ):
        priors[label_train[i]] = priors[label_train[i]] + 1
        for j in range ( 784 ):
            train[label_train[i]][j] = train[label_train[i]][j] + data_train_bi[i][j]

    return train, priors

def read_csv_as_matrix( csv_path ):

    return pd.read_csv( csv_path ).as_matrix()

def get_label_and_data_from_train( csv_as_matrix ):
    print("get_label_and_data_from_train")
    label_train = csv_as_matrix[0:,0]
    data_train = csv_as_matrix[0:,1:]

    return label_train, data_train

def binarization( data_train ):
    print("binarization")
    return ( data_train >= 128 ).astype( int )

def export_result( results, file_name ):

    print("export_result")
    df = pd.DataFrame( data = results, columns = ['ImageId', 'Label'] )
    df.to_csv( file_name, sep = ',', index=False )

def main():

    train_dataset = read_csv_as_matrix( 'train.csv' )
    test_dataset = read_csv_as_matrix( 'test.csv' )

    label_train, data_train = get_label_and_data_from_train( train_dataset )

    #train
    data_train_bi = binarization( data_train )
    train, priors = train_linear_regression( label_train, data_train_bi )

    #predict
    test_dataset_bi = binarization( test_dataset )
    results = apply_linear_regression( train, priors, test_dataset_bi )

    #export result
    export_result( results, "result-100-110.csv" )

main()
