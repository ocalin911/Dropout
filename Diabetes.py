#DROPOUT OPTIMIZATION for Diabetes data set

from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
import numpy as np

boston = load_diabetes()

x, y = boston.data, boston.target

dropouts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
mses1=[]
mses2=[]
mses3=[]
mses4=[]
mses5=[]
mses6=[]
mses7=[]
mses8=[]
mses9=[]
mses10=[]
def output(d):
    model = Sequential()
    model.add(Dense(20, input_dim=10, activation="relu")) 
    model.add(Dropout(d))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(x, y, epochs=20, batch_size=16, verbose=2)
    l = model.evaluate(x, y)
    return l


for d in dropouts:
    l=output(d)
    mses1.append(l)
print("1st loop done")

for d in dropouts:
    l=output(d)
    mses2.append(l)
print("2st loop done")

for d in dropouts:
    l=output(d)
    mses3.append(l)
print("3rd loop done")

for d in dropouts:
    l=output(d)
    mses4.append(l)
print("4th loop done")

for d in dropouts:
    l=output(d)
    mses5.append(l)
print("5th loop done")

for d in dropouts:
    l=output(d)
    mses6.append(l)
print("6th loop done")
    
for d in dropouts:
    l=output(d)
    mses7.append(l)
print("7th loop done")

for d in dropouts:
    l=output(d)
    mses8.append(l)
print("8th loop done")

for d in dropouts:
    l=output(d)
    mses9.append(l)
print("9th loop done")

for d in dropouts:
    l=output(d)
    mses10.append(l)    
print("10th loop done")

data = [mses1, mses2,mses3, mses4,mses5, mses6,mses7, mses8,mses9, mses10]
a=np.array(data)
res=np.average(a, axis =0)

plt.plot(dropouts, mses1, 'y-.')
plt.plot(dropouts, mses2, 'g-.')
plt.plot(dropouts, mses3, 'b-.')
plt.plot(dropouts, mses4, 'r-.')
plt.plot(dropouts, mses5, 'c-.')
plt.plot(dropouts, mses6, 'm-.')
plt.plot(dropouts, mses7, 'y--')
plt.plot(dropouts, mses8, 'g--')
plt.plot(dropouts, mses9, 'b--')
plt.plot(dropouts, mses10, 'c--')
plt.plot(dropouts, res, 'bs')
plt.ylabel("MSE")
plt.xlabel("dropout value")
plt.title('MSE versus dropout rate for diabetes data')
plt.show()




