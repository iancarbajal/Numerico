# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 23:29:38 2022

Codigo de la practica 1 para la clase de cálculo numerico I

@author: Ian Carbajal
"""
import numpy as np
from scipy import linalg as la

def metodo_potencia(A,q0,k=500,tol=1.0*10**-7):
    for i in range(k):
        q0=A@q0
        q0=q0*1/np.max(np.abs(q0))
        
    sigmai=q0.T@A@q0
    
    return sigmai, q0

def metodo_potencia_inv(A,q0,rho=0,k=500,tol=1.0*10**-7):
    I=np.eye(len(A))
    for i in range(k):
        q0=la.solve(A-rho*I, q0)
        q0=q0*1/np.max(np.abs(q0))
        
    sigmai=q0.T@A@q0
    
    return sigmai, q0
    
def metodo_potencia_invRayleigh(A,q0,k=500):
    I=np.eye(len(A))
    for i in range(k):
        rho=q0.T@A@q0/q0.T@q0
        q0=la.solve(A-rho*I, q0)
        q0=q0*1/np.max(np.abs(q0))
        
    sigmai=q0.T@A@q0
    
    return sigmai, q0

def MQR_simple(A,k):
    for i in range(0,k):
        Q,R=np.linalg.qr(A)
        A=R@Q
    return A

def MQR_dynamic(A,tol = 1.0*10**-7, ctol=500):
    A=la.hessenberg(A)
    n=len(A)
    lam=[0]*(n)
    while(n>1):
        I=np.eye(n)
        cont=0
        while((np.max(np.abs(A[-1,:n-1])) > tol*np.abs(A[-1,-1])) and cont<ctol):
            S=A[-1,-1]
            Q,R=la.qr(A-S*I)
            A=R@Q + S*I
            cont+=1
        
        if(cont<ctol):
            lam[n-1]=A[-1,-1]
            n-=1
            A=A[:n,:n]  
        else:
            disc=((A[n-2,n-2]-A[n-1,n-1])**2*A[n-1,n-2]*A[n-2,n-1])
            lam[n-1]=((A[n-2,n-2]+A[n-1,n-1]+np.emath.sqrt(disc))/2)
            lam[n-2]=((A[n-2,n-2]+A[n-1,n-1]-np.emath.sqrt(disc))/2)
            n-=2

    lam[0]=A[0,0]
    
    return lam

A=np.array([3,2,4,2,0,2,4,2,3]).reshape(3,3)
q0=np.array([1,1,1])

MQR_dynamic(A)