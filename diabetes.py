from audioop import avg
from collections import Counter
from random import random, seed, shuffle
from numpy import genfromtxt
data_path = 'diabetes.csv'
dataset = genfromtxt(data_path, delimiter=',')

Train_set = []
Val_set = []
Test_set = []

seed(123)
shuffle(dataset)
for S in dataset:
    R = random()
    if R >= 0 and R <=0.7:
        Train_set.append(S)
    elif R >= 0.7 and R <= 0.85:
        Val_set.append(S)
    else:
        Test_set.append(S)


K = 20

def run(Val_set):
    error = 0
    total = 0
    for V in Val_set:
        L = []
        for T in Train_set:
            N = len(T)

            distance = 0
            for i in range(N-1):
                distance = (V[i] - T[i]) ** 2

            train_class = T[-1]
            L.append((distance, train_class))

        L.sort()
        L = L[:K]

        total_value = 0
        for distance, val in L:
            total_value += val

        avg_value = total_value / len(L)
        true_value = V[-1]

        error += (true_value - avg_value) ** 2
        total += 1

    mean_squared_error = error/total
    return mean_squared_error

val_error = run(Val_set)
print("Validation Error: ", val_error)

test_error = run(Test_set)
print("Test Error: ", test_error)