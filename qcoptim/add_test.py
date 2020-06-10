#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:16:58 2020

@author: chris
"""

import copy
from abc import ABC, abstractmethod

class myinterface(ABC):
    
    @abstractmethod
    def doit(self,x):
        raise NotImplementedError
        
    def __add__(self,obj):
        tmp_1 = copy.deepcopy(self)
        tmp_2 = copy.deepcopy(obj)
        new_obj = trivialclass()
        new_obj.mylist = tmp_1.mylist + tmp_2.mylist
        new_obj.doit = (lambda x: tmp_1.doit(x) + tmp_2.doit(x))
        return new_obj
        
class trivialclass(myinterface):
    
    def doit(self,x):
        pass
        
class myclass1(myinterface):
    
    def __init__(self,a):
        self.a = a
        self.mylist = [ i for i in range(a) ]
        
    def doit(self,x):
        return x*self.a
    
class myclass2(myinterface):
    
    def __init__(self,b):
        self.b = b
        self.mylist = [ -i for i in range(b) ]
        
    def doit(self,x):
        return x/self.b
    
A = myclass1(10)
B = myclass2(10)

print(A.mylist)
print(A.doit(2))

print(B.mylist)
print(B.doit(2))

C = A + B
#del A
del B

print(C.mylist)
print(C.doit(2))

A.a = 20

print(C.mylist)
print(C.doit(2))