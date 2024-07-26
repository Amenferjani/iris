import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from model import Model

torch.manual_seed(50)
model = Model()

data = pd.read_csv('iris.csv')
data["species"] = data["species"].replace('setosa',0.0)
data["species"] = data["species"].replace('versicolor', 1.0)
data["species"] = data["species"].replace('virginica', 2.0)

dataFrame = data.drop("species",axis=1)
target = data["species"]

dataFrame = dataFrame.values
target = target.values

xTrain, xTest, yTrain, yTest = train_test_split(
        dataFrame, target, test_size=0.2, random_state=42
    )

xTrain = torch.FloatTensor(xTrain)
xTest = torch.FloatTensor(xTest)

yTrain = torch.LongTensor(yTrain)
yTest = torch.LongTensor(yTest)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100
losses = []
for epoch in range(epochs):
    yPred = model.forward(xTrain)
    loss = criterion(yPred, yTrain)
    losses.append(loss)
    if epoch % 10 == 0 :
        print(f"epoch : {epoch} , loss : {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


with torch.no_grad():
    yEval = model.forward(xTest)
    loss = criterion(yEval, yTest)
    print("loss : ",loss)


correct = 0 
with torch.no_grad():
    for i , data in enumerate(xTest):
        yVal = model.forward(data)
        print(f"{i+1} , {str(yVal)} \t {yTest[i]}")
        if yVal.argmax().item()== yTest[i]:
            correct=correct+1
print("correct :",correct)


