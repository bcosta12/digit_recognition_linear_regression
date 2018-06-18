import pandas as pd
import numpy as np
import random as rd
import math

def train_linear_regression_lms(  label_train, data_train, thetas, alfa, tolerance ):
    print("Init train_linear_regression_lms")
    #train, priors = extract_vocabulary_and_count_docs( label_train, data_train )

    min_error = len( data_train ) * tolerance * 784

    for k in range (1000):
        print( k, 1000 )
        current_error = 0
        for i in range( len( data_train ) ):

            d = np.insert(data_train[i], 0, 1)
            done = False
            old = 10000000

            for j in range( 784 ):
                #print(j)

                #print(label_train[i], h_theta( d, thetas ), round ( label_train[i] - h_theta( d, thetas ),2 ) )
                if( abs (round( label_train[i] - h_theta( d, thetas ), 2)) < tolerance  ) :
                    #done = True
                    break
                thetas[j] += ( alfa * ( round( label_train[i] - h_theta( d, thetas ), 2 ) ) * d[j] )
                if old < h_theta( d, thetas ):
                    #done = True
                    break
                old = h_theta( d, thetas )

            current_error += abs (round( label_train[i] - h_theta( d, thetas ), 2))
            if done:
                break
        if current_error <= min_error:
            break



    print("Final train_linear_regression_lms")
    return thetas

def h_theta( a, b ):
    np_a = np.array(a)
    np_b = np.array(b)
    #dot_sum = sum( [a[i]*b[i] for i in range(len(b))]
    dot_sum =  np.dot(np_a, np_b)
    if math.isnan(dot_sum):
        dot_sum = 0
    return abs( round( dot_sum, 2 ) )

def apply_linear_regression_lms( thetas, test_dataset ):
    print("Init apply_linear_regression_lms")
    results, score = extract_terms_from_dosc( test_dataset )

    for i in range( len( test_dataset ) ):
        print( i, len( test_dataset ) )
        d = np.insert(test_dataset[i], 0, 1)
        result = int( h_theta( d, thetas ) )
        result = result % 9

        if result == 10:
            result = 9
        results[i] = [i+1, result]
    print("Final apply_linear_regression_lms")
    return results

def extract_terms_from_dosc ( test_dataset ):

    results = [[0]*2 for i in range( len( test_dataset ) )]
    score = [0]*10
    return results, score

def extract_vocabulary_and_count_docs( label_train, data_train ):
    train = [[0]*785 for _ in range(10)]
    priors = [0]*10
    #Count Vocabulary and Docs
    for i in range( len( label_train ) ):
        priors[label_train[i]] = priors[label_train[i]] + 1
        for j in range ( 784 ):
            train[label_train[i]][j] = train[label_train[i]][j] + data_train[i][j]

    return train, priors

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
    alfa = 0.01
    tolerance=0.4


    train_dataset = read_csv_as_matrix( 'train.csv' )
    test_dataset = read_csv_as_matrix( 'test.csv' )

    label_train, data_train = get_label_and_data_from_train( train_dataset )

    #train
    thetas = train_linear_regression_lms( label_train, data_train, thetas, alfa, tolerance )

    #predict
    results = apply_linear_regression_lms( thetas, test_dataset )

    #export result
    export_result( results, 'result_s-'+str(alfa)+'-'+str(tolerance)+'.csv' )

main()
