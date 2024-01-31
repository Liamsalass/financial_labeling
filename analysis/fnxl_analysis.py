import pandas as pd
import numpy as np
import argparse as ap

def fnxl_analysis(args):
    print ('Root directory: ', args.root)
    label_count = pd.read_csv(args.root + 'labelCount.csv')
    print(label_count.shape)
    print('Head:\n', label_count.head())
    print('\nMax val in last column:', label_count.iloc[:, -1].max())
    print('Min val in last column:', label_count.iloc[:, -1].min())

    print('Mean val in last column:', label_count.iloc[:, -1].mean())
    print('Median val in last column:', label_count.iloc[:, -1].median())
    print('Std val in last column:', label_count.iloc[:, -1].std())

    train = pd.read_csv(args.root + 'train_sample.csv')
    test = pd.read_csv(args.root + 'test_sample.csv')
    print('Train shape:', train.shape)
    print('Test shape:', test.shape)
    print('Train head:\n', train.head())
    print('\nTest head:\n', test.head())


  

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--root',
                        help='root directory of the project',
                        type=str,
                        default='B:/datasets/FNXL/')
    
    args = parser.parse_args()


    fnxl_analysis(args)