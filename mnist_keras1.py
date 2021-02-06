
#Dropout Optimization Mnist data set
#The setup is for plotting loss versus dropout rate
#For plotting accuracy versus dropout rate use the lines with "#"

import tensorflow as tf
mnist = tf.keras.datasets.mnist

import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def output1(d):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dropout(d),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=4)
    res=model.evaluate(x_test, y_test)
    return res[0]
    #return res[1]


dropouts = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]

mses1=[]
for d in dropouts:
    l=output1(d)
    mses1.append(l)

mses2=[]
for d in dropouts:
    l=output1(d)
    mses2.append(l)

mses3=[]
for d in dropouts:
    l=output1(d)
    mses3.append(l)

mses4=[]
for d in dropouts:
    l=output1(d)
    mses4.append(l)

mses5=[]
for d in dropouts:
    l=output1(d)
    mses5.append(l)

mses6=[]
for d in dropouts:
    l=output1(d)
    mses6.append(l)
    
data = [mses1, mses2,mses3, mses4, mses5,mses6]
a=np.array(data)
res=np.average(a, axis =0)

plt.plot(dropouts, mses1, 'y-.')
plt.plot(dropouts, mses2, 'g-.')
plt.plot(dropouts, mses3, 'b-.')
plt.plot(dropouts, mses4, 'r-.')
plt.plot(dropouts, mses5, 'c-.')
plt.plot(dropouts, mses6, 'm-.')
plt.plot(dropouts, res, 'bs')
plt.ylabel("Loss")
#plt.ylabel("Accuracy")
plt.xlabel("dropout value")
plt.title('Loss versus dropout rate for mnist data')
#plt.title('Accuracy versus dropout rate for mnist data')
plt.show()    
#model.summary()

