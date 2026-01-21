##==== import ====##
import time
import os
import sys
import random
from random import sample
import math
import scipy
from scipy import stats
from scipy.stats import rankdata
import sklearn
import numpy as np
import pandas as pd
import pickle
from importlib import resources
#---------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
import seaborn as sns; sns.set()
#----------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,precision_score,recall_score

from pathlib import Path


def bcdc3(data_all):
    #-------read pickle data --------
    data_path = Path(__file__).resolve().parent / "bgm_m_object.pickle"
    with data_path.open("rb") as f:
        bgm_m = pickle.load(f)
    Z_m = bgm_m.score_samples(data_all) ; ll_m=np.exp(Z_m)
    data_path1 = Path(__file__).resolve().parent / "bgm_a_object.pickle"
    with data_path1.open("rb") as f:
        bgm_a = pickle.load(f)
    Z_m1 = bgm_a.score_samples(data_all) ; ll_m1=np.exp(Z_m1)
    data_path2 = Path(__file__).resolve().parent / "bgm_c_object.pickle"
    with data_path2.open("rb") as f:
        bgm_c = pickle.load(f)
    Z_c = bgm_c.score_samples(data_all) ; ll_c=np.exp(Z_c)
    #----------------------bayes prob---------------------------------------
    prior_m=0.3267076255872786; prior_m1=0.3415251174557282; prior_c=0.3317672569569931
    
    ll_tot=prior_m*ll_m+prior_m1*ll_m1+prior_c*ll_c
    prob_m=prior_m*ll_m/ll_tot
    prob_m1=prior_m1*ll_m1/ll_tot
    prob_c=prior_c*ll_c/ll_tot
    #---------------labeling------------------------
    label=np.zeros(len(ll_tot))
    a=prob_m.reshape(len(ll_tot),1)
    b=prob_m1.reshape(len(ll_tot),1)
    c=prob_c.reshape(len(ll_tot),1)
    d=np.concatenate([a,b,c], axis=1)
    for iii in range(len(ll_tot)):
        label[iii]=np.argmax(d[iii])
    ent=np.concatenate((data_all,label.reshape(len(data_all[:,0]),1)), axis=1)
    ent=pd.DataFrame(ent)
    #----------------------------------------------------------------
    return {'label': label}
    
def proj_cv(numb,k,i,j,m,n,l,q, cv,si):
    if numb == 2:
        pjc=[[cv[si[k]][i][i],cv[si[k]][i][j]],[cv[si[k]][j][i],cv[si[k]][j][j]]]
        return pjc
    elif numb == 3:
        pjc=[[cv[si[k]][i][i],cv[si[k]][i][j],cv[si[k]][i][m]],\
         [cv[si[k]][j][i],cv[si[k]][j][j],cv[si[k]][j][m]],\
         [cv[si[k]][m][i],cv[si[k]][m][j],cv[si[k]][m][m]]]
        return pjc
    elif numb==4:
        pjc=[[cv[si[k]][i][i],cv[si[k]][i][j],cv[si[k]][i][m],cv[si[k]][i][n]],\
         [cv[si[k]][j][i],cv[si[k]][j][j],cv[si[k]][j][m],cv[si[k]][j][n]],\
         [cv[si[k]][m][i],cv[si[k]][m][j],cv[si[k]][m][m],cv[si[k]][m][n]],\
         [cv[si[k]][n][i],cv[si[k]][n][j],cv[si[k]][n][m],cv[si[k]][n][n]]]
        return pjc
    elif numb==5:
        pjc=[[cv[si[k]][i][i],cv[si[k]][i][j],cv[si[k]][i][m],cv[si[k]][i][n],cv[si[k]][i][l]],\
         [cv[si[k]][j][i],cv[si[k]][j][j],cv[si[k]][j][m],cv[si[k]][j][n],cv[si[k]][j][l]],\
         [cv[si[k]][m][i],cv[si[k]][m][j],cv[si[k]][m][m],cv[si[k]][m][n],cv[si[k]][m][l]],\
         [cv[si[k]][n][i],cv[si[k]][n][j],cv[si[k]][n][m],cv[si[k]][n][n],cv[si[k]][n][l]],\
          [cv[si[k]][l][i],cv[si[k]][l][j],cv[si[k]][l][m],cv[si[k]][l][n],cv[si[k]][l][l]]]
        return pjc
    elif numb==6:
        pjc=[[cv[si[k]][i][i],cv[si[k]][i][j],cv[si[k]][i][m],cv[si[k]][i][n],cv[si[k]][i][l],cv[si[k]][i][q]],\
         [cv[si[k]][j][i],cv[si[k]][j][j],cv[si[k]][j][m],cv[si[k]][j][n],cv[si[k]][j][l],cv[si[k]][j][q]],\
         [cv[si[k]][m][i],cv[si[k]][m][j],cv[si[k]][m][m],cv[si[k]][m][n],cv[si[k]][m][l],cv[si[k]][m][q]],\
         [cv[si[k]][n][i],cv[si[k]][n][j],cv[si[k]][n][m],cv[si[k]][n][n],cv[si[k]][n][l],cv[si[k]][n][q]],\
          [cv[si[k]][l][i],cv[si[k]][l][j],cv[si[k]][l][m],cv[si[k]][l][n],cv[si[k]][l][l],cv[si[k]][l][q]],\
        [cv[si[k]][q][i],cv[si[k]][q][j],cv[si[k]][q][m],cv[si[k]][q][n],cv[si[k]][q][l],cv[si[k]][q][q]]]
        return pjc
def proj_m(numb,k,i,j,m,n,l,q, mm, si):
    if numb == 2:
        pjm=[mm[si[k]][i],mm[si[k]][j]]
        return pjm
    elif numb == 3:
        pjm=[mm[si[k]][i],mm[si[k]][j],mm[si[k]][m]]
        return pjm
    elif numb == 4:
        pjm=[mm[si[k]][i],mm[si[k]][j],mm[si[k]][m],mm[si[k]][n]]
        return pjm
    elif numb == 5:
        pjm=[mm[si[k]][i],mm[si[k]][j],mm[si[k]][m],mm[si[k]][n],mm[si[k]][l]]
        return pjm
    elif numb == 6:
        pjm=[mm[si[k]][i],mm[si[k]][j],mm[si[k]][m],mm[si[k]][n],mm[si[k]][l],mm[si[k]][q]]
        return pjm
def obsdat(numb,k,i,j,m,n,l,q, obs):
    if numb == 2:
        obdat=np.concatenate((np.reshape(obs[:,i], (len(obs),1)),np.reshape(obs[:,j], (len(obs),1))), axis=1)
        return obdat
    elif numb == 3:
        obdat=np.concatenate((np.reshape(obs[:,i], (len(obs),1)),np.reshape(obs[:,j], (len(obs),1)),\
                              np.reshape(obs[:,m], (len(obs),1))), axis=1)
        return obdat
    elif numb == 4:
        obdat=np.concatenate((np.reshape(obs[:,i], (len(obs),1)),np.reshape(obs[:,j], (len(obs),1)),\
                              np.reshape(obs[:,m], (len(obs),1)),np.reshape(obs[:,n], (len(obs),1))), axis=1)
        return obdat
    elif numb == 5:
        obdat=np.concatenate((np.reshape(obs[:,i], (len(obs),1)),np.reshape(obs[:,j], (len(obs),1)),\
                              np.reshape(obs[:,m], (len(obs),1)),np.reshape(obs[:,n], (len(obs),1)),\
                             np.reshape(obs[:,l], (len(obs),1))), axis=1)
        return obdat
    elif numb == 6:
        obdat=np.concatenate((np.reshape(obs[:,i], (len(obs),1)),np.reshape(obs[:,j], (len(obs),1)),\
                              np.reshape(obs[:,m], (len(obs),1)),np.reshape(obs[:,n], (len(obs),1)),\
                             np.reshape(obs[:,l], (len(obs),1)),np.reshape(obs[:,q], (len(obs),1))), axis=1)
        return obdat
def bcdc3proj(numb,i, j, m, n, l, q, obs, clid):
    #-------read pickle data --------
    data_path = Path(__file__).resolve().parent / "bgm_m_object.pickle"
    with data_path.open("rb") as f:
        bgm_m = pickle.load(f)
    covariances=bgm_m.covariances_ ; mean_values=bgm_m.means_ ; a=bgm_m.weights_ 
    sorted_index=np.where(a > 1./100)[0] ; numb_of_gaussian=np.shape(sorted_index)[0]
    data_path1 = Path(__file__).resolve().parent / "bgm_a_object.pickle"
    with data_path1.open("rb") as f:
        bgm_a = pickle.load(f)
    data_path2 = Path(__file__).resolve().parent / "bgm_c_object.pickle"
    with data_path2.open("rb") as f:
        bgm_c = pickle.load(f)
    covariances1=bgm_c.covariances_ ; mean_values1=bgm_c.means_ ; a1=bgm_c.weights_
    sorted_index1=np.where(a1 > 1./50)[0] ; numb_of_gaussian1=np.shape(sorted_index1)[0]
    #-----------------------------------------------------#
    if (numb > 6)|(numb<2):
        print("wrong number error")
 
    #-----------------------------------------------------#
    relax_cl=[]; recent_cl=[]; ancient_cl=[]; 
    ##======fit bayes-gauss-mixture model for each sample======##
    #-------------------merger sample1--------------------
    gm = GaussianMixture(n_components=100, random_state=0, verbose=2)    
    means=[] ;  Cov=[] ;mm=bgm_m.means_ ; cv=bgm_m.covariances_ ; si=sorted_index

    for k in range(numb_of_gaussian):
        #print(sorted_index[k])
        pjm=proj_m(numb,k,i,j,m,n,l,q, mm, si)
        means.append(pjm)

        pjc= proj_cv(numb,k,i,j,m,n,l,q, cv,si)
        Cov.append(pjc)

    
    M=np.asarray(means)
    C=np.asarray(Cov)

    gm.means_=M
    gm.covariances_=C
    gm.weights_=bgm_m.weights_[sorted_index]
    gm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(C)).transpose((0, 2, 1))
    obdat=obsdat(numb,k,i,j,m,n,l,q, obs)
    Z_m = gm.score_samples(obdat) #log-likelihood
    ll_m=np.exp(Z_m)

    #-------------------merger sample2--------------------
    gm = GaussianMixture(n_components=100, random_state=0, verbose=2)    
    means=[] ;  Cov=[] ;mm=bgm_a.means_ ; cv=bgm_a.covariances_ ; si=sorted_index
    
    for k in range(numb_of_gaussian):
        #print(sorted_index[k])
        pjm=proj_m(numb,k,i,j,m,n,l,q, mm, si)
        means.append(pjm)

        pjc=proj_cv(numb,k,i,j,m,n,l,q, cv,si)
        Cov.append(pjc)

    M=np.asarray(means)
    C=np.asarray(Cov)

    gm.means_=M
    gm.covariances_=C
    gm.weights_=bgm_a.weights_[sorted_index]
    gm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(C)).transpose((0, 2, 1))

    obdat=obsdat(numb,k,i,j,m,n,l,q, obs)
    Z_m1 = gm.score_samples(obdat)
    ll_m1=np.exp(Z_m1)
    #--------------------relaxed sample-----------------------
    gm = GaussianMixture(n_components=50, random_state=0, verbose=2)    
    means=[] ;  Cov=[] ;mm=bgm_c.means_ ; cv=bgm_c.covariances_ ; si=sorted_index1
    
    for k in range(numb_of_gaussian1):
        #print(sorted_index[k])
        pjm=proj_m(numb,k,i,j,m,n,l,q, mm, si)
        means.append(pjm)

        pjc= proj_cv(numb,k,i,j,m,n,l,q, cv,si)
        Cov.append(pjc)

    M=np.asarray(means)
    C=np.asarray(Cov)

    gm.means_=M
    gm.covariances_=C
    gm.weights_=bgm_c.weights_[sorted_index1]
    gm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(C)).transpose((0, 2, 1))

    obdat=obsdat(numb,k,i,j,m,n,l,q, obs)
    Z_c = gm.score_samples(obdat)
    ll_c=np.exp(Z_c)

    #----------------------bayes prob---------------------------------------
    prior_m=0.3267076255872786; prior_m1=0.3415251174557282; prior_c=0.3317672569569931
    ll_tot=prior_m*ll_m+prior_m1*ll_m1+prior_c*ll_c
    prob_m=prior_m*ll_m/ll_tot
    prob_m1=prior_m1*ll_m1/ll_tot
    prob_c=prior_c*ll_c/ll_tot
    #---------------labeling------------------------
    label=np.zeros(len(obs))
    a=prob_m.reshape(len(obs),1)
    b=prob_m1.reshape(len(obs),1)
    c=prob_c.reshape(len(obs),1)
    d=np.concatenate([a,b,c], axis=1)
    for iii in range(len(obs)):
        label[iii]=np.argmax(d[iii])
    indp=np.where(label == 0) ;  indp1=np.where(label==1) ; indp2=np.where(label==2)
    recent_cl.append(clid[indp]) ;ancient_cl.append(clid[indp1]); relax_cl.append(clid[indp2])
    recent_cl1=np.array(recent_cl).reshape(len(recent_cl[0]), 1)
    ancient_cl1=np.array(ancient_cl).reshape(len(ancient_cl[0]), 1)
    relax_cl1=np.array(relax_cl).reshape(len(relax_cl[0]), 1)
    recent_prob=d[indp] ; ancient_prob=d[indp1] ; relax_prob=d[indp2]
    #---------------------------------------------
    recent=np.concatenate((recent_cl1,recent_prob),axis=1)
    ancient=np.concatenate((ancient_cl1,ancient_prob),axis=1)
    relax=np.concatenate((relax_cl1,relax_prob),axis=1)
    
    #---------------------------------------------
    return {'recent': recent,'ancient':ancient,  'relax': relax}
