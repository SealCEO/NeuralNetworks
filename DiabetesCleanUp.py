import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV as RSCV
from sklearn.model_selection import cross_val_score
import math

print('___PLEASE READ____')
print("This may give different results than in the powerpoints. A lot of editing has been done when finding the best results, and these may no longer have the optimal solutions in them that we put in the powerpoint since we changed things around to test other possibilities."
      + " But, the code should still work the exact same and do everything the exact same. This will still create neural networks and test them against other methods. Just maybe not as optimal as a neural network we presented in the powerpoints.")
print('About this code:\nThe code is broken down into blocks with True/False statements. Turn a block off by turning to False. This allows for separate methods to run.'
      + "\nAlso, feel free to mess around with train/test size. This will make the models run faster/slower with more/less accuracy respectively.")

#Used for random, but consistent results
import random
rng = random.Random(x = 104)

data = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
data.astype(float)
X = data.drop(['Diabetes_binary'], axis = 1)
y = data[['Diabetes_binary']].values.ravel()
cv=5

def avg_cv(method, X, y, cv):
    total = cross_val_score(method, X, y, cv = cv)
    avg_score = 0
    for val in total:
        avg_score += val
    return avg_score / cv

def RF(X, y):
    size = len(X.columns)
    param_grid = {'n_estimators': [25, 75, 125],
              'max_features': [math.ceil((size) ** 0.5) * 2, math.ceil((size) ** 0.5), math.ceil(math.log2(size))],
              'max_depth': [3, 5, 7, 9, 11],
              'max_leaf_nodes': [20, 30, 40, 50, 60, 70]}
    #Forcing a random smaller training set because of computational power
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10000, test_size = 10, random_state=rng.randint(0, 200))
    model = RSCV(RandomForestClassifier(), param_grid, n_iter = 15, cv = 5).fit(X_train, y_train)
    return model

#Other methods
if True:
    print(f'The average accuracy score for LDA using {cv}-fold CV: {avg_cv(LinearDiscriminantAnalysis(), X, y, cv)}')
    print(f'The average accuracy score for Naive Bayes using {cv}-fold CV: {avg_cv(GaussianNB(), X, y, cv)}')
    print(f'The average accuracy score for QDA using {cv}-fold CV: {avg_cv(QuadraticDiscriminantAnalysis(), X, y, cv)}')

    best = RF(X, y).best_params_
    rf = RandomForestClassifier(n_estimators=best['n_estimators'], max_features= best['max_features'],
                                max_depth = best['max_depth'], max_leaf_nodes = best['max_leaf_nodes'], random_state=rng.randint(0, 200))
    print(f'The average accuracy score for Random Forests using {cv}-fold CV and GridSearch for the params: {avg_cv(rf, X, y, cv)}\n\n')

#Neural networks
if True:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader

    torch.manual_seed(rng.randint(1,200))
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    #Builds a Dataset
    class CustomDataset(Dataset):
        def __init__(self, x, y):
            super(CustomDataset, self).__init__()
            self.x = torch.tensor(x.values, dtype=torch.float32)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, test_size = 0.025, random_state=rng.randint(0, 200))
    training_data = CustomDataset(X_train, y_train)
    test_data = CustomDataset(X_test, y_test)

    best_acc = -1
    best_seq = None
    best_lr = -1

    # Batch size is how much data to look at at once
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Changing the number of neurons
    for num in range(15, 31):
        dict = {0 : nn.Sequential(nn.Linear(len(X.columns), num), nn.Linear(num, 2 * num), nn.ReLU(), nn.Linear(2 * num, 1), nn.Sigmoid()),
                1 : nn.Sequential(nn.Linear(len(X.columns), num), nn.ReLU(), nn.Linear(num, num), nn.ReLU(), nn.Linear(num, num), nn.ReLU(), nn.Linear(num, 1), nn.Sigmoid()),
                2 : nn.Sequential(nn.Linear(len(X.columns), num), nn.ReLU(), nn.Linear(num, num), nn.ReLU(), nn.Linear(num, 1), nn.Sigmoid()),
                3 : nn.Sequential(nn.Linear(len(X.columns), num), nn.ReLU(), nn.Linear(num, 1), nn.Sigmoid()),
                4 : nn.Sequential(nn.Linear(len(X.columns), num), nn.Tanh(), nn.Linear(num, num), nn.Tanh(), nn.Linear(num, 1), nn.Sigmoid()),
                5 : nn.Sequential(nn.Linear(len(X.columns), num), nn.Softmax(dim=1), nn.Linear(num, num), nn.Softmax(dim=1), nn.Linear(num, 1), nn.Sigmoid()),
                6 : nn.Sequential(nn.Linear(len(X.columns), num), nn.ELU(), nn.Linear(num, num), nn.ELU(), nn.Linear(num, 1), nn.Sigmoid()),
                7 : nn.Sequential(nn.Linear(len(X.columns), num), nn.SELU(), nn.Linear(num, num), nn.SELU(), nn.Linear(num, 1), nn.Sigmoid()),
                8 : nn.Sequential(nn.Linear(len(X.columns), num), nn.LeakyReLU(), nn.Linear(num, num), nn.LeakyReLU(), nn.Linear(num, 1), nn.Sigmoid()),
                9 : nn.Sequential(nn.Linear(len(X.columns), num), nn.Sigmoid(), nn.Linear(num, num), nn.Sigmoid(), nn.Linear(num, 1), nn.Sigmoid())}
        for i in range(len(dict)):
            model = NeuralNetwork(dict[i]).to(device)

            # Define Loss function and Optimizer - lr gives a measure of how much to change by on each pass
            # Using BCELoss() for binary classification entropy loss
            loss_fn = nn.BCELoss()
            lrs = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
            for lr in lrs:
                optimizer = optim.Adam(model.parameters(), lr=lr)
                # Training Loop - epochs is how many times to run through the data set
                num_epochs = 10
                for epoch in range(num_epochs):
                    model.train()
                    for batch in train_dataloader:
                        inputs, targets = batch
                        inputs, targets = inputs.to(device), targets.to(device)

                        optimizer.zero_grad()
                        outputs = torch.flatten(model(inputs))
                        loss = loss_fn(outputs, targets)
                        loss.backward()
                        optimizer.step()

                # Evaluation on test data
                model.eval()
                correct = 0
                total = 0
                rou = 0

                with torch.no_grad():
                    for batch in test_dataloader:
                        inputs, targets = batch
                        targets = torch.flatten(targets)
                        inputs, targets = inputs.to(device), targets.to(device)

                        outputs = model(inputs)
                        outputs = torch.flatten(outputs)
                        predicted = (outputs > 0.5).float()
                        total += targets.size(dim=0)
                        correct += (predicted == targets).sum().item()

                accuracy = correct / total
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_seq = dict[i]
                    best_lr = lr

    #Repeating of everything above but now on the actual full set
    if True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=rng.randint(0, 200))
        training_data = CustomDataset(X_train, y_train)
        test_data = CustomDataset(X_test, y_test)
        train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

        model = NeuralNetwork(best_seq).to(device)
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=best_lr)
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            for batch in train_dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = torch.flatten(model(inputs))
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            for batch in test_dataloader:
                inputs, targets = batch
                targets = torch.flatten(targets)
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                outputs = torch.flatten(outputs)
                predicted = (outputs > 0.5).float()
                total += targets.size(dim=0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        print(f'Using seq={best_seq} and lr={best_lr}, we get {accuracy} accuracy using Neural Networks.')
