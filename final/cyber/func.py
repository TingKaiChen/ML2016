
import numpy as np
import csv

nb_train = 4408587
nb_test = 606779
dim = 41+2+69+10    #len(dim_1)+len(dim_2)+len(dim_3)+others
types = {'normal' : 0, 'dos' : 1, 'u2r' : 2, 'r2l' : 3, 'probe' : 4}

def read_train(train_file):
    
    X = np.zeros((nb_train, dim))
    y = np.zeros((nb_train,))

    idx = 0
    dim_1 = {}      # len(dim_1) == 3
    dim_2 = {}      # len(dim_2) == 70
    dim_3 = {}      # len(dim_3) == 11
    dim_1c = 0
    dim_2c = 0
    dim_3c = 0

    maptype = {}
    maptypec = 0
    mapclass = read_label()

    with open(train_file, 'r') as f:
        for line in f:
            data = line.strip('.\n')
            data = data.split(',')

            for i in xrange(len(data)-1):
                if i == 1:
                    if not data[1] in dim_1:
                        dim_1[data[i]] = dim_1c
                        dim_1c += 1
                    X[idx, 1+dim_1[data[i]]] = 1
                elif i == 2:
                    if not data[2] in dim_2:
                        dim_2[data[i]] = dim_2c
                        dim_2c += 1
                    X[idx, 4+dim_2[data[i]]] = 1
                elif i == 3:
                    if not data[3] in dim_3:
                        dim_3[data[i]] = dim_3c
                        dim_3c += 1
                    X[idx, 74+dim_3[data[i]]] = 1
                else:
                    X[idx, i+81] = float(data[i])

            '''
            if not data[-1] in maptype:
                maptype[data[-1]] = maptypec
                maptypec += 1
            '''

            y[idx] = mapclass[data[-1]] #maptype[data[-1]]
            idx += 1

    return X, y.astype(int), dim_1, dim_2, dim_3, mapclass
            

def read_test(test_file, dim_1, dim_2, dim_3):
    
    X = np.zeros((nb_test, dim))
    idx = 0
    with open(test_file, 'r') as f:
        for line in f:
            data = line.strip()
            data = data.split(',')
            for i in xrange(len(data)):
                if i == 1:
                    X[idx, 1+dim_1[data[i]]] = 1
                elif i == 2:
                    if not data[i] in dim_2:
                        dim_2[data[i]] = max(dim_2.values())+1
                        # 'icmp' not exist in training data
                        dim_2[data[i]] -= 1
                    X[idx, 4+dim_2[data[i]]] = 1
                elif i == 3:
                    X[idx, 74+dim_3[data[i]]] = 1
                else:
                    X[idx, i+81] = float(data[i])

            idx += 1
    return X

def read_label():
    
    mapclass = {'normal':0}

    with open('training_attack_types.txt', 'r') as f:
        for line in f:
            data = line.strip()
            data = data.split()
            mapclass[data[0]] = types[data[1]]
    return mapclass

def write_pred(pred_file, pred):
    with open(pred_file, 'wb') as f:
        wr = csv.writer(f)
        wr.writerow(["id", "label"])
        for i in xrange(len(pred)):
            wr.writerow([i+1, int(pred[i])])

if __name__ == '__main__':
    # nb_train = 13
    print 'Reading Data'
    train_file = 'train'
    test_file = 'test.in'
    output = 'rf.csv'

    X, y, dim_1, dim_2, dim_3, maptype = read_train(train_file)

    X_test = read_test(test_file, dim_1, dim_2, dim_3)
    

