from collections import Counter
from random import random, seed, shuffle
from numpy import genfromtxt
data_path = 'iris.csv'
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


K = 10

def run(Val_set):
    correct = 0
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

        class_count = Counter()
        for distance, _class in L:
            class_count[_class] += 1

        majority_class = class_count.most_common()[0][0]
        target_class = V[-1]

        if majority_class == target_class:
            correct += 1
        total += 1

    accuracy = (correct / total) * 100
    return accuracy

val_accuracy = run(Val_set)
print("Validation Accuracy: ", val_accuracy)

test_accuracy = run(Test_set)
print("Test Accuracy: ", test_accuracy)