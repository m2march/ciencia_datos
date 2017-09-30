# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division
import pandas as pd
import numpy as np
import scipy.io as sio
import scipy as sp
import os
import seaborn as sns
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
import scipy.signal as scs
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from ipywidgets import FloatProgress
from IPython.display import display
from MatrixProvider import MatrixProvider
import time
import pickle
import glob
from IPython.display import Image

from numpy import *

import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)

symbs = [''.join([str(c) for c in x]) 
         for x in [[0, 1, 2],[0, 2, 1],[1, 0, 2],[1, 2, 0],[2, 1, 0],[2, 0, 1]]]
valores = list('abcdef')
symbs_dict = dict(zip(symbs, valores))


def serie_a_simbolos(samples):
    """Convierte la serie de samples a una serie de símbolos a-f."""
    simbolos = [] 
    for i in range(len(samples) - 3):
        orden = np.argsort(samples[i:i+3])
        orden_str = ''.join([str(c) for c in orden])
        simbolos.append(symbs_dict[orden_str])
    return simbolos

def informacion_mutua_epoch(electrodo_samples):
    """Calcula la información mutua para un epoch.
    
    Args:
        electrodo_samples matriz de 256 electrodos por 201 samples
    """
    cant_valores = len(valores)
    # Diccionario para contabilizar la probabilidad de cada símbolo individual
    prob_symb = dict(zip(valores, zeros(len(valores))))
    # Diccionario para todas las tuplas (v1, v2) para contabilizar la probabilidad conjunta
    # Las claves del diccionario tienen los dos elementos ordeandos ya que es lo mismo (a, b) que (b, a)
    prob_symb_symb = dict([((valores[i], valores[j]), 0)  
                           for i in range(cant_valores) 
                           for j in range(i, cant_valores)])
    electrodos_simbolos = [serie_a_simbolos(electrodo_samples[i, :])
                           for i in range(electrodo_samples.shape[0])]
    us = []
    vs = []
    for i in range(len(electrodos_simbolos)):
        for j in range(i+1, len(electrodos_simbolos)):
            us.extend(electrodos_simbolos[i])
            vs.extend(electrodos_simbolos[j])
            
    return mutual_info_score(us, vs)

def informacion_mutua_sujeto(subject_matrix):
    epochs_info = []
    for i in range(subject_matrix.shape[0]):
        logging.debug('>    Epoch: {}'.format(i))
        epochs_info.append(informacion_mutua_epoch(subject_matrix[i, :, :]))
    return np.mean(epochs_info)

#Imean_c_filename = 'pickles/Imean_c.pkl'
#if not os.path.isfile(Imean_c_filename):
#    Imean_c = np.zeros(10)
#    for idx, path in enumerate(glob.glob('dataset/S*.mat')):
#        data = sio.loadmat(path)['data']
#        Imean_c[idx] = informacion_mutua_sujeto(data)
#
#    with open(Imean_c_filename, 'wb') as f:
#        pickle.dump(Imean_c, f)
#else:
#    with open(Imean_c_filename, 'rb') as f:
#        Imean_c = pickle.load(f)

Imean_p_filename = 'pickles/Imean_p.pkl'
if not os.path.isfile(Imean_p_filename):
    Imean_p = np.zeros(10)
    for idx, path in enumerate(glob.glob('dataset/P*.mat')):
        logging.debug('> MI for {}'.format(path))
        data = sio.loadmat(path)['data']
        Imean_p[idx] = informacion_mutua_sujeto(data)

    with open(Imean_p_filename, 'wb') as f:
        pickle.dump(Imean_p, f)
else:
    with open(Imean_p_filename, 'rb') as f:
        Imean_p = pickle.load(f)
