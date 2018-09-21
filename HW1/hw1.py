import csv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
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
    print('P(spam)', p_spam)

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
    print(u_spam, sigma2_spam)
    print(u_notspam, sigma2_notspam)

    def log_gaussian(u, sigma2, x):
        return -sum(0.5 * np.log(2 * np.pi * sigma2) + (x - u) ** 2 / (2 * sigma2))

    # test error    
    error = 0
    for x, y in zip(X_test, y_test):
        post_spam = np.log(p_spam) + log_gaussian(u_spam, sigma2_spam, x)
        post_notspam = np.log(1 - p_spam) + log_gaussian(u_notspam, sigma2_notspam, x)
        error += int(post_spam > post_notspam) != y
    print(error / len(y_test))

    # gaussian distribution proof
    def gaussian_prob(x, u, sigma2):
        return np.e ** (- (x - u) ** 2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)
    for i in range(len(X_train_spam[0])):
        data = [x[i] for x in X_train_spam]
        print(max(data), min(data))
        hist, bins = np.histogram(data, bins=500)
        bin_centers = (bins[1:]+bins[:-1])*0.5
        plt.plot(bin_centers, hist / max(hist))
        plt.scatter(data, gaussian_prob(data, u_spam[i], sigma2_spam[i]))
        plt.show()