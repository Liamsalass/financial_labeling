import pandas as pd
import numpy as np
import argparse as ap








def main(args):
    print ('Root directory: ', args.root)
    label_count = pd.read_csv(args.root + 'labelCount.csv')
    print(label_count.head())
    print(label_count.shape)
    # print max value in each column
    print(label_count.max())
    # print min value in each column
    print(label_count.min())




if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--root',
                        help='root directory of the project',
                        type=str,
                        default='B:/datasets/FNXL/')
    
    args = parser.parse_args()


    main(args)