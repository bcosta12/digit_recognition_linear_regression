import pandas as pd
import numpy as np
import random as rd
import math

def train_linear_regression_lms(  label_train, data_train, thetas, alfa, tolerance ):
    print("Init train_linear_regression_lms")

    data_train = np.hstack( ( np.ones( len(data_train) ).reshape( -1, 1 ), data_train ) )
    data_train /= 255

    while(True):
        error = ( data_train.dot( thetas ) - label_train )
        new_thetas = thetas - alfa * error.dot( data_train )
        print( np.linalg.norm( error ) )
        print(np.linalg.norm( thetas - new_thetas ), np.linalg.norm( tolerance ) )
        if( np.linalg.norm( thetas - new_thetas ) <  tolerance ):
            break
        thetas = new_thetas

    print ( "Final train_linear_regression_lms" )
    return thetas

def h_theta( a, b ):
    dot_sum =  np.dot(a, b)
    return dot_sum

def apply_linear_regression_lms( thetas, test_dataset ):
    print("Init apply_linear_regression_lms")

    test_dataset = np.hstack( ( np.ones( len(test_dataset) ).reshape( -1, 1 ), test_dataset ) )
    test_dataset/=255
    final_result = len( test_dataset )*[0]
    results = h_theta( test_dataset, thetas )

    for i in range( len( test_dataset ) ):
        print( i, len( test_dataset ) )
        print(results[i])
        if results[i] > 9:
            results[i] = 9
        if results[i] < 0:
            results[i] = 0
        final_result[i] = [i+1, int( results[i] )]
    print("Final apply_linear_regression_lms")
    return final_result

def read_csv_as_matrix( csv_path ):
    return pd.read_csv( csv_path ).as_matrix()

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

def main():

    thetas = get_thetas( 785 )
    alfa = 3e-08
    tolerance= 1e-6


    train_dataset = read_csv_as_matrix( 'train.csv' )
    test_dataset = read_csv_as_matrix( 'test.csv' )

    label_train, data_train = get_label_and_data_from_train( train_dataset )

    #train
    thetas = train_linear_regression_lms( label_train, data_train, thetas, alfa, tolerance )

    #predict
    results = apply_linear_regression_lms( thetas, test_dataset )

    #export result
    export_result( results, 'results_scalar-'+str(alfa)+'-'+str(tolerance)+'.csv' )

main()
