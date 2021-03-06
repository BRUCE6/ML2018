import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sys.stdout = open('output.txt', 'w')
    X_train, y_train = [], []
    X_test, y_test = [], []
    with open('spambasetrain.csv', 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            X_train.append([float(n) for n in line[:-1]])
            y_train.append(int(line[-1]))
    with open('spambasetest.csv', 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            X_test.append([float(n) for n in line[:-1]])
            y_test.append(int(line[-1]))
    
    # calculate prior
    p_spam = sum(y_train) / len(y_train)
    print('1.')
    print('P(spam)', p_spam)
    print('P(not_spame)', 1 - p_spam)
    print()

    # calculate likelihood
    X_train_spam, X_train_notspam = [], []
    for i in range(len(X_train)):
        if y_train[i]:
            X_train_spam.append(X_train[i])
        else:
            X_train_notspam.append(X_train[i])
    def calculate_likelihood(X):
        u = np.sum(X, axis = 0) / len(X)
        sigma2 = np.sum((X - u) ** 2, axis = 0) / (len(X) - 1)
        return u, sigma2
    u_spam, sigma2_spam = calculate_likelihood(X_train_spam)
    u_notspam, sigma2_notspam = calculate_likelihood(X_train_notspam)
    print('2.')
    print('p(xi|spam)')
    for i, (u, s2) in enumerate(zip(u_spam, sigma2_spam)):
        print(i + 1, u, s2)
    print('p(xi|not_spam)') 
    for i, (u, s2) in enumerate(zip(u_notspam, sigma2_notspam)):
        print(i + 1, u, s2)
    print()

    def log_gaussian(u, sigma2, x):
        return -sum(0.5 * np.log(2 * np.pi * sigma2) + (x - u) ** 2 / (2 * sigma2))

    # test error  
    print('3.')
    print('predicted class: 1 - spam, 0 - not spam')  
    error = 0
    for x, y in zip(X_test, y_test):
        post_spam = np.log(p_spam) + log_gaussian(u_spam, sigma2_spam, x)
        post_notspam = np.log(1 - p_spam) + log_gaussian(u_notspam, sigma2_notspam, x)
        predict = int(post_spam > post_notspam) 
        print(predict)
        error += predict != y
    print()
    print(4.)
    print('number of test examples classified correctly:', len(y_test) - error)
    print()
    print(5.)
    print('number of test examples classified incorrectly:', error)
    print()
    print(5.)
    print('percentage error:', error / len(y_test))
    sys.stdout.close()