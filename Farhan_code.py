#House prediction model
#by
#Farhan Hassan

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading the dataset
farhan_data = pd.read_csv('karachi_data.csv',names=["area","bed","bath","kitchen","floor","car","servant","price"]) #read the data
#print(farhan_data)
#X=farhan_data.iloc[:,0:2]
#print(X)
mean=farhan_data.mean()
print(mean)
standard=farhan_data.std()
print(standard)
farhan_data = (farhan_data - farhan_data.mean())/farhan_data.std()
print(farhan_data)
print(farhan_data.head())

#setting the matrixes
X = farhan_data.iloc[:,0:7]
ones = np.ones([X.shape[0],1])
#printing (ones)
X = np.concatenate((ones,X),axis=1)
y = farhan_data.iloc[:,7:8].values 
theta = np.zeros([1,8])
print(X)

#setting hyper parameters 
alpha = 0.01
iters = 10000

#computing cost function
def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))

#Applying Gradient Descent
def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)
    
    return theta,cost

#Gradient Descent and cost function
g,cost = gradientDescent(X,y,theta,iters,alpha)
print(g)

finalCost = computeCost(X,y,g)
print(finalCost)

count=0
length=len(X)
price=np.zeros(length)
print(length)
for i in X:
        price[count]=i @ g.T
        #if(count==0 or count==1 or count==2 or count==3 or count==4):
#            price1=(price[count]*176971.227812+197495.652174)
#            print("price = " +str(price[count]))
        count=count+1
    
fig, bx = plt.subplots()
bx.scatter(np.arange(length),price,marker='o')
bx.set_xlabel('Features')
bx.set_ylabel('Price')
bx.set_title('Data Behavior')
#plot the cost
fig, ax = plt.subplots()  
ax.plot(np.arange(iters), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')

#Giving some input value
#input1=1, input2 = 0.765414, input3 = 0.672741, input4 = 0.997934, input5 = 1.277270, input6 = 1.470150, input7 = 1.778357, input8 = 0.841726, input9 = 0.823327
#
##Values of thetas computed using gradient descent
#theta1= -4.67187006e-17   
#theta2=7.94548335e-01
#theta3= 2.56197332e-02
#theta4=1.06528876e-03
#theta5=9.18959182e-03
#theta6=-2.72787225e-02 
#theta7=6.15331103e-02
#theta8=2.02048002e-02
#theta9=-5.34679317e-02

#Predicting output price
y=(input1*theta1)+(input2*theta2)+(input3*theta3)+(input4*theta4)+(input5*theta5)+(input6*theta6)+(input7*theta7)+(input8*theta8)+(input9*theta9)
y=(y*3.200947e+07)+2.722821e+07
print("y= " + str(y))