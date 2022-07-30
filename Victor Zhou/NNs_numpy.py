# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 01:56:24 2022

@author: William
"""

#只用numpy搭建一个神经网络

import numpy as np

#定义激活函数
def sigmoid(x):
    # 激活函数：f(x)=1/(1+e^(-x))
    return 1/(1+np.exp(-x))

#定义神经元
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    #定义前馈函数
    def feedforward(self, inputs):
        #输入权重，添加偏差，然后使用激活函数
        total = np.dot(self.weights,inputs) + self.bias
        return sigmoid(total)

weights = np.array([0,1]) #w1 = 0, w2 = =1
bias = 4
n = Neuron(weights, bias)

x = np. array([2,3]) #x1 = 2, x2 = 3
print(n.feedforward(x)) #0.999

#定义神经网络
# class OurNeuralNetwork:
#     '''
#     该神经网络包含：
#     2个输入
#     一个含有两个神经元的隐藏层
#     一个含有一个神经元的输出层
#     每个神经元具有相同的权重和偏差值
#     '''
#     def __init__(self):
#         weights = np.array([0,1])
#         bias = 0
#         #神经元见上面定义
#         self.h1 = Neuron(weights, bias)
#         self.h2 = Neuron(weights, bias)
#         self.o1 = Neuron(weights, bias)
    
#     def feedforward(self,x):
#         out_h1 = self.h1.feedforward(x)
#         out_h2 = self.h2.feedforward(x)
#         #o1的输入是h1和h2的输出
#         out_o1 = self.o1.feedforward(np.array([out_h1,out_h2]))
        
#         return out_o1

# network = OurNeuralNetwork()
# x = np.array([2,3])
# print(network.feedforward(x))


#训练神经网络
#定义损失函数
def mes_loss(y_true, y_pred):
    #
    return ((y_true-y_pred)**2).mean()

y_true = np.array([1,0,0,1])
y_pred = np.array([0,0,0,0])

print(mes_loss(y_true, y_pred))

def deriv_sigmoid(x):
    #sigmoid激活函数的导数
    fx = sigmoid(x)
    return fx*(1-fx)

class NewNeuralNetwork:
    '''
    该神经网络与上面OurNeuralNetwork差不多
    '''
    def __init__(self):
        #权重
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        #偏差
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        #定义前馈函数    
    def feedforward(self, x):
        h1 = sigmoid(self.w1*x[0]+self.w2*x[1]+self.b1)
        h2 = sigmoid(self.w3*x[0]+self.w4*x[1]+self.b2)
        o1 = sigmoid(self.w5*h1+self.w6*h2+self.b3)
        return o1
    def train(self,data,all_y_trues):
        '''
        data 是(nx2)的numpy数组，n是数据集中样本的个数
        all_y_trues 是有n个元素的numpy数组

        '''
        learn_rate = 0.1
        epochs = 1000 #对整个数据集的训练次数
        
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                #计算前馈函数
                sum_h1 = self.w1*x[0]+self.w2*x[1]+self.b1
                h1 = sigmoid(sum_h1)
                
                sum_h2 = self.w3*x[0]+self.w4*x[1]+self.b2
                h2 = sigmoid(sum_h2)
                
                sum_o1 = self.w5*h1+self.w6*h2+self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1
                
                #计算偏导数
                #命名：d_L_d_w1 表示偏L/偏wx
                d_L_d_ypred = -2 * (y_true - y_pred)
                
                #神经元 o1
                d_ypred_d_w5 = h1*deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2*deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)
        
                d_ypred_d_h1 = self.w5*deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6*deriv_sigmoid(sum_o1)
        
                #神经元 h1
                d_h1_d_w1 = x[0]*deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1]*deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                #神经元 h2
                d_h2_d_w3 = x[0]*deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1]*deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)
                
                #更新权重和偏差
                #神经元 h1
                self.w1 -= learn_rate*d_L_d_ypred*d_ypred_d_h1*d_h1_d_w1
                self.w2 -= learn_rate*d_L_d_ypred*d_ypred_d_h1*d_h1_d_w2
                self.b1 -= learn_rate*d_L_d_ypred*d_ypred_d_h1*d_h1_d_b1
                
                #神经元 h2
                self.w3 -= learn_rate*d_L_d_ypred*d_ypred_d_h2*d_h2_d_w3
                self.w4 -= learn_rate*d_L_d_ypred*d_ypred_d_h2*d_h2_d_w4
                self.b2 -= learn_rate*d_L_d_ypred*d_ypred_d_h2*d_h2_d_b2
                
                #神经元 o1
                self.w5 -= learn_rate*d_L_d_ypred*d_ypred_d_w5
                self.w6 -= learn_rate*d_L_d_ypred*d_ypred_d_w6
                self.b3 -= learn_rate*d_L_d_ypred*d_ypred_d_b3
                
            #在每一次epoch后计算总损失函数
            if epoch % 100 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mes_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch,loss))

#定义数据集
data = np.array([
    [-2,-1], #Alice
    [25,6], #Bob
    [17,4], #Charlie
    [-15,-6], #Diana
     ])
all_y_trues = np.array([
    1, #Alice
    0, #Bob
    0, #Charlie
    1, #Diana
    ])

#训练我们的神经网络！
network = NewNeuralNetwork()
network.train(data, all_y_trues)

#做些预测
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M
                
                
                
                
                
                
                
                
                
                
                
                