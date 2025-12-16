import io
import numpy as np
from sklearn import svm
import scipy as sp
import pandas as pd
import os 
import time


# взять все строки и добавить в список
# взять только последние 107 элементы с 0 и 119 с 1

list_line = []

with open ("SVM\data\data_banknote_authentication.txt") as f : 
    for line in f:
        list_line.append(line)

# read the data

def read_and_convert_to_csv(filename, mode) :
    if (os.path.exists(filename)):
        os.remove(filename)

    with open(filename, "x") as f :
        for i in range(0, len(list_line)) :
            if (mode == "direct") :
                f.write(list_line[i])
            elif (mode == "reverse") :
                f.write(list_line[(len(list_line)-1) - i])
  
# convert to csv
def read_fake_and_true_from_csv(filename, true_count, fake_count) :
    headers = ["entropy", "dispersion_coef", "assymetry_coef", "veivlet_conversion"]
    df = pd.read_csv(filename, header=None, names=headers)
    
    print(df.head(5))

if __name__ == "__main__" :
    train_csv = "train_bin.csv"
    read_and_convert_to_csv(train_csv, "reverse")

    read_fake_and_true_from_csv(train_csv, 0, 0)

train = np.genfromtxt

