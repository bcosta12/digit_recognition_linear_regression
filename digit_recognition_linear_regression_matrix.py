import pandas as pd
import numpy as np
import random as rd
import math

def train_linear_regression_lms(  label_train, data_train, thetas ):
    print("Init train_linear_regression_lms")

    data_train_transpose = np.transpose( data_train )

    #theta = inv(xT * x) * xT * y, split in 3 parts
    part_1 = np.matmul( data_train_transpose, data_train )
    part_1 = (part_1 + 0.0000001*np.random.rand(784, 784)).astype(float)
    part_1_inv = np.linalg.inv( part_1 )
    part_2 = np.matmul( part_1_inv, data_train_transpose )
    part_3 = np.matmul( part_2, label_train )
    thetas = part_3
    print("Final train_linear_regression_lms")
    return thetas

def h_theta( a, b ):

    matmul =  np.matmul(a, b)
    return  np.argmax( matmul )

def apply_linear_regression_lms( thetas, test_dataset ):
    print("Init apply_linear_regression_lms")
    results, score = extract_terms_from_dosc( test_dataset )

    for i in range( len( test_dataset ) ):
        print( i, len( test_dataset ) )
        d = test_dataset[i]

        result = int( h_theta( d, thetas ) )
        print(result)
        if result > 9:
            result = 9
        if result < 0:
            result = 0
        results[i] = [i+1, result]
    print("Final apply_linear_regression_lms")
    return results

def read_csv_as_matrix( csv_path ):

    return pd.read_csv( csv_path ).as_matrix()

def extract_terms_from_dosc ( test_dataset ):

    results = [[0]*2 for i in range( len( test_dataset ) )]
    score = [0]*10
    return results, score

def get_label_and_data_from_train( csv_as_matrix ):
    print("get_label_and_data_from_train")
    label_train = csv_as_matrix[0:,0]
    data_train = csv_as_matrix[0:,1:]

    return label_train, data_train

def export_result( results, file_name ):

    print("export_result")
    df = pd.DataFrame( data = results, columns = ['ImageId', 'Label'] )
    df.to_csv( file_name, sep = ',', index=False )

def get_thetas( length ):
    thetas = [0]*length
    for i in range( length ):
        thetas[i] = rd.random()
    return  thetas

def to_vector( labels ):

    vet_labels = []
    for label in labels:
        vet = 10*[0]
        vet[label] = 1
        vet_labels.append( vet )
    return vet_labels

def to_matrix( data ):

    matrix_data = []
    for d in data:
        matrix_data.append( d )
    return matrix_data

def to_number( vet_labels ):
    numbers=[]
    for v in vet_labels:
        numbers.append( np.argmax( v ) )
    return numbers

def main():

    thetas = get_thetas( 785 )

    train_dataset = read_csv_as_matrix( 'train.csv' )
    test_dataset = read_csv_as_matrix( 'test.csv' )

    label_train, data_train = get_label_and_data_from_train( train_dataset )

    #train
    label_train = to_vector( label_train )
    data_train = to_matrix( data_train )
    thetas = train_linear_regression_lms( label_train, data_train, thetas )

    #predict
    data_train = to_matrix( data_train )
    results = apply_linear_regression_lms( thetas, test_dataset )

    #export result
    export_result( results, "result_matrix.csv" )

main()
