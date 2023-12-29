from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV as RSCV
from sklearn.model_selection import cross_val_score
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#DO NOT MESS WITH RNG - IT WILL RUIN THE GRAPHS I HAVE COPIED AND PASTED INTO SLIDES
#Used for random, but consistent results
import random
rng = random.Random(x = 104)


def avg_cv(method, X, y, cv):
    total = cross_val_score(method, X, y, cv = cv)
    avg_score = 0
    for val in total:
        avg_score += val
    return avg_score / cv

def RF(X, y, rng):
    size = len(X[0])
    param_grid = {'n_estimators': [25, 75, 125],
              'max_features': [math.ceil((size) ** 0.5) * 2, math.ceil((size) ** 0.5), math.ceil(math.log2(size))],
              'max_depth': [3, 5, 7, 9, 11],
              'max_leaf_nodes': [20, 30, 40, 50, 60, 70]}
    #Forcing a random smaller training set because of computational power
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, test_size = 10, random_state=rng.randint(0, 200))
    model = RSCV(RandomForestClassifier(), param_grid, n_iter = 15, cv = 5).fit(X_train, y_train)
    return model

def runningMethods(X,y,cv,rng,classes):
    #Other methods
    if True:
        print(f'The average accuracy score for LDA using {cv}-fold CV: {avg_cv(LinearDiscriminantAnalysis(), X, y, cv)}')
        print(f'The average accuracy score for Naive Bayes using {cv}-fold CV: {avg_cv(GaussianNB(), X, y, cv)}')
        print(f'The average accuracy score for QDA using {cv}-fold CV: {avg_cv(QuadraticDiscriminantAnalysis(), X, y, cv)}')

        best = RF(X, y, rng=rng).best_params_
        rf = RandomForestClassifier(n_estimators=best['n_estimators'], max_features= best['max_features'],
                                    max_depth = best['max_depth'], max_leaf_nodes = best['max_leaf_nodes'], random_state=rng.randint(0, 200))
        print(f'The average accuracy score for Random Forests using {cv}-fold CV and GridSearch for the params: {avg_cv(rf, X, y, cv)}')

    #Neural networks
    if True:
        torch.manual_seed(rng.randint(1,200))
        device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        #Builds a Dataset
        class CustomDataset(Dataset):
            def __init__(self, x, y):
                super(CustomDataset, self).__init__()
                self.x = torch.tensor(x, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32)

            def __len__(self):
                return len(self.x)

            def __getitem__(self, index):
                return self.x[index], self.y[index]

        #Defines a NeuralNetwork
        class NeuralNetwork(nn.Module):
            #seq of layers
            def __init__(self, seq):
                super().__init__()
                self.seq = seq

            def forward(self, x):
                logits = self.seq(x)
                return logits
        

        #This will just be used for getting the best parameters - Forcing a random smaller training set because of computational power
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, test_size = 0.05, random_state=rng.randint(0, 200))
        
        training_data = CustomDataset(X_train, y_train)
        test_data = CustomDataset(X_test, y_test)

        best_seq = None
        best_lr = -1
        best_e = -1
        best_overall_acc = -1

        # Batch size is how much data to look at at once
        train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, pin_memory=True)
        test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, pin_memory=True)

        # Changing the number of neurons
        size = len(X[0])
        list_of_nur =[]
        if size//2 == abs(size - (classes // 2)) + 1:
            list_of_nur = [size//2, size + 1]
        else:
            list_of_nur = [size//2, size + 1, abs(size - (classes // 2)) + 1]
        for num in list_of_nur:
            dict = {0 : nn.Sequential(nn.Linear(size, 2 * num), nn.SELU(), nn.Linear(2 * num, classes), nn.Softmax(dim=1)),
                    1 : nn.Sequential(nn.Linear(size, num), nn.SELU(), nn.Linear(num, classes), nn.Softmax(dim=1)),

                    2 : nn.Sequential(nn.Linear(size, 2 * num), nn.ReLU(), nn.Linear(2 * num, classes), nn.Softmax(dim=1)),
                    3 : nn.Sequential(nn.Linear(size, num), nn.ReLU(), nn.Linear(num, classes), nn.Softmax(dim=1)),

                    4 : nn.Sequential(nn.Linear(size, 2 * num), nn.Tanh(), nn.Linear(2 * num, classes), nn.Softmax(dim=1)),
                    5 : nn.Sequential(nn.Linear(size, num), nn.Tanh(), nn.Linear(num, classes), nn.Softmax(dim=1)),

                    6 : nn.Sequential(nn.Linear(size, 2 * num), nn.Softmax(dim=1), nn.Linear(2 * num, classes), nn.Softmax(dim=1)),
                    7 : nn.Sequential(nn.Linear(size, num), nn.Softmax(dim=1), nn.Linear(num, classes), nn.Softmax(dim=1)),

                    8 : nn.Sequential(nn.Linear(size, 2 * num), nn.Sigmoid(), nn.Linear(2 * num, classes), nn.Softmax(dim=1)),
                    9 : nn.Sequential(nn.Linear(size, num), nn.Sigmoid(), nn.Linear(num, classes), nn.Softmax(dim=1)),}
            for i in range(len(dict)):
                model = NeuralNetwork(dict[i])
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)

                # Move model to device
                model.to(device)

                # Define Loss function and Optimizer - lr gives a measure of how much to change by on each pass
                loss_fn = nn.CrossEntropyLoss()
                lrs = [0.1, 0.01, 0.001]
                for lr in lrs:
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    # Training Loop - epochs is how many times to run through the data set
                    patience = 10  # Set patience for early stopping
                    counter = 0  # Counter to track how many epochs have passed without improvement
                    patience_for_slow = 10
                    counter_for_slow = 0
                    best_acc = -1
                    best_cur_e = -1
                    num_epochs = 500
                    for epoch in range(num_epochs):
                        model.train()
                        for batch in train_dataloader:
                            inputs, targets = batch
                            targets = targets.to(torch.long)
                            inputs, targets = inputs.to(device), targets.to(device)

                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = loss_fn(outputs, targets)
                            loss.backward()
                            optimizer.step()
                    
                        # Validation
                        model.eval()  # Set the model to evaluation mode
                        correct = 0
                        total = 0
                        with torch.no_grad():
                            for batch in test_dataloader:
                                inputs, targets = batch
                                targets = torch.flatten(targets)
                                inputs, targets = inputs.to(device), targets.to(device)

                                outputs = model(inputs)
                                predicted = outputs.argmax(1)
                                total += targets.size(dim=0)
                                correct += (predicted == targets).sum().item()

                        accuracy = correct / total

                        # Check for improvement
                        if abs(accuracy - best_acc) < 0.00005:
                            counter_for_slow += 1
                            if counter_for_slow >= patience_for_slow:
                                break #Stop because training imporvement very slow
                        else:
                            counter_for_slow = 0

                        if accuracy > best_acc:
                            best_acc = accuracy
                            best_cur_e = epoch + 1
                            counter = 0  # Reset counter since there's improvement
                        else:
                            counter += 1
                            if counter >= patience:
                                break  # Stop training as there's no improvement for 'patience' epochs
                    
                    if best_acc > best_overall_acc:
                        best_overall_acc = best_acc
                        best_seq = dict[i]
                        best_lr = lr
                        best_e =  best_cur_e

    
        #Repeating of everything above but now on the actual full set
        if True:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=rng.randint(0, 200))
            training_data = CustomDataset(X_train, y_train)
            test_data = CustomDataset(X_test, y_test)
            train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, pin_memory=True)
            test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, pin_memory=True)

            model = NeuralNetwork(best_seq)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            # Move model to device
            model.to(device)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=best_lr)
            num_epochs = best_e
            for epoch in range(num_epochs):
                model.train()
                for batch in train_dataloader:
                    inputs, targets = batch
                    targets = targets.to(torch.long)
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    optimizer.step()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_dataloader:
                    inputs, targets = batch
                    targets = torch.flatten(targets)
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs)
                    predicted = outputs.argmax(1)
                    total += targets.size(dim=0)
                    correct += (predicted == targets).sum().item()

            accuracy = correct / total
            print(f'Using seq={best_seq}, lr={best_lr}, and num_epochs={best_e}, we get {accuracy} accuracy using Neural Networks.\n\n')

#Just to keep the graphs consistent
rngmet = random.Random(x=rng.randint(0,200))

print('___PLEASE READ____')
print("These plots will be slightly different than those in the powerpoints. We used a set random number generator to do the other ones, but now we have editted this document a lot so it wont produce the exact same plot."
      + " But, the plot should be similar and have the same basic idea. Also, a lot of editting was done, so this file may no longer contain the optimal neural network, but it will still make a neural network nonetheless.")
print('About this code:\nThe code is broken down into blocks with True/False statements. Turn a block off by turning to False. This allows for separate methods to run.'
      + "\nAlso, feel free to mess around with train/test size. This will make the models run faster/slower with more/less accuracy respectively.")


classes = 3
features = 2
inform = 2
clusters = 1
X_vis, y_vis = make_classification(n_classes=classes, n_samples=5000, n_features=features, n_informative=inform, n_repeated=0, n_redundant=0, n_clusters_per_class=clusters, random_state=rng.randint(0,200))
plt.scatter(X_vis[:,0],X_vis[:,1], s=20, c=y_vis)
plt.title(f'{features} with {inform} informative and {clusters} clusters/class')
plt.show()
X = pd.DataFrame(X_vis)
y = pd.DataFrame(y_vis)
y = y.rename(columns={0: "Class"}).values.ravel()
runningMethods(X_vis, y_vis, 5, rngmet, classes)

classes = 2
features = 2
inform = 2
clusters = 2
X_vis, y_vis = make_classification(n_classes=classes, n_samples=5000, n_features=features, n_informative=inform, n_repeated=0, n_redundant=0, n_clusters_per_class=clusters, random_state=rng.randint(0,200))
plt.scatter(X_vis[:,0],X_vis[:,1], s=20, c=y_vis)
plt.title(f'{features} with {inform} informative and {clusters} clusters/class')
plt.show()
X = pd.DataFrame(X_vis)
y = pd.DataFrame(y_vis)
y = y.rename(columns={0: "Class"}).values.ravel()
runningMethods(X_vis, y_vis, 5, rngmet, classes)

classes = 2
features = 2
inform = 1
clusters = 1
X_vis, y_vis = make_classification(n_classes=classes, n_samples=5000, n_features=features, n_informative=inform, n_repeated=0, n_redundant=0, n_clusters_per_class=clusters, random_state=rng.randint(0,200))
plt.scatter(X_vis[:,0],X_vis[:,1], s=20, c=y_vis)
plt.title(f'{features} with {inform} informative and {clusters} clusters/class')
plt.show()
X = pd.DataFrame(X_vis)
y = pd.DataFrame(y_vis)
y = y.rename(columns={0: "Class"}).values.ravel()
runningMethods(X_vis, y_vis, 5, rngmet, classes)

classes = 4
features = 2
inform = 2
clusters = 1
X_vis, y_vis = make_classification(n_classes=classes, n_samples=5000, n_features=features, n_informative=inform, n_repeated=0, n_redundant=0, n_clusters_per_class=clusters, random_state=rng.randint(0,200))
plt.scatter(X_vis[:,0],X_vis[:,1], s=20, c=y_vis)
plt.title(f'{features} with {inform} informative and {clusters} clusters/class')
plt.show()
X = pd.DataFrame(X_vis)
y = pd.DataFrame(y_vis)
y = y.rename(columns={0: "Class"}).values.ravel()
runningMethods(X_vis, y_vis, 5, rngmet, classes)

classes = 2
features = 2
inform = 2
clusters = 2
X_vis, y_vis = make_classification(n_classes=classes, n_samples=5000, n_features=features, n_informative=inform, n_repeated=0, n_redundant=0, n_clusters_per_class=clusters, random_state=rng.randint(0,200))
plt.scatter(X_vis[:,0],X_vis[:,1], s=20, c=y_vis)
plt.title(f'{features} with {inform} informative and {clusters} clusters/class')
plt.show()
X = pd.DataFrame(X_vis)
y = pd.DataFrame(y_vis)
y = y.rename(columns={0: "Class"}).values.ravel()
runningMethods(X_vis, y_vis, 5, rngmet, classes)

classes = 8
features = 10
inform = 10
clusters = 5
X_vis, y_vis = make_classification(n_classes=classes, n_samples=100000, n_features=features, n_informative=inform, n_repeated=0, n_redundant=0, n_clusters_per_class=clusters, random_state=rng.randint(0,200))
X = pd.DataFrame(X_vis)
y = pd.DataFrame(y_vis)
y = y.rename(columns={0: "Class"})
new = pd.concat([X, y], axis=1, join='inner')
plot_this, throw_away = train_test_split(new, train_size=0.05, test_size = 10, random_state=0)
sns.pairplot(plot_this, hue = 'Class')
plt.show()
runningMethods(X_vis, y_vis, 5, rngmet, classes)

classes = 4
features = 20
inform = 15
clusters = 8
X_vis, y_vis = make_classification(n_classes=classes, n_samples=200000, n_features=features, n_informative=inform, n_repeated=0, n_redundant=0, n_clusters_per_class=clusters, random_state=rng.randint(0,200))
X = pd.DataFrame(X_vis)
y = pd.DataFrame(y_vis)
y = y.rename(columns={0: "Class"})
new = pd.concat([X, y], axis=1, join='inner')
plot_this, throw_away = train_test_split(new, train_size=0.025, test_size = 10, random_state=0)
sns.pairplot(plot_this, hue = 'Class')
plt.show()
runningMethods(X_vis, y_vis, 5, rngmet, classes)


