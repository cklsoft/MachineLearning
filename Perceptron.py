#! /bin/env python
#-*- coding=utf-8 -*-
import numpy as np
class Perceptron():
    def __init__(self,data,cls):
        self.data=data
        self.b,self.m,self.n=0,len(data),len(data[0])
        self.alpha=1
        self.cls=cls
    def __calc(self,x,y):
         if (self.b+np.dot(self.w,x))*y<=0:
             return False
         return True
    def run(self,maxIter):#原式
        #print u"感知机原始形式：\n"
        self.w=np.zeros((1,self.m))
        cnt,curIter=0,0
        while curIter<maxIter:#小于最大迭代次数(最大调整次数)
            flag=False
            for i in range(self.n):#依顺序判断误分类
                if self.__calc(self.data[:,i],self.cls[i])==False:#误分类
                    self.w+=self.alpha*(self.data[:,i]*self.cls[i])#调整w,b
                    self.b+=self.alpha*self.cls[i]
                    print self.w,self.b
                    flag,cnt,curIter=True,cnt+1,curIter+1
                    break
            if flag==False:#全部例子都通过
                break
    def runDual(self,maxIter):#对偶式
        print u"感知机对偶形式：\n"
        a=np.zeros((1,self.n))#对偶系数
        cnt,b,iter=0,0,0
        Gram=np.dot(self.data.transpose(),self.data)#预处理
        while iter<maxIter:
            flag=False
            for id in range(self.n):#枚举误分类数据
                if self.cls[id]*(np.dot(a*self.cls,Gram[id,:])+b)<=0:#找到
                    a[0,id]+=self.alpha#调整
                    b+=self.alpha*self.cls[id]
                    print a,b
                    flag,cnt,iter=True,cnt+1,iter+1
                    break
            if flag==False:
                break
        w=np.zeros((1,self.m))
        for i in range(self.n):#利用训练得到的系数求出真正的系数
            w+=a[0,i]*self.cls[i]*self.data[:,i]
        np.dot(a,self.cls)
        print w,b
    def getAns(self):
        return self.w,self.b

def readData(path):
    f=open(path)
    lines=f.readlines()
    f.close()
    n,m=len(lines),0
    retCls,id=[],0
    for line in lines:
        p=line.strip()
        h=p.split()
        if id==0:
            m=len(h)-1
            retMat=np.zeros((m,n))
        retMat[:,id]=h[:-1]
        retCls.append(int(h[-1]))
        id+=1
    return retMat,retCls
if __name__=='__main__':
    data,cls=readData('perData.in')
    p=Perceptron(data,cls)
    p.run(1000)
    p.runDual(1000)