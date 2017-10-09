# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division
from numpy import *
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
import scipy.io as sio
import glob
import numpy as np
import os
import pandas as pd
import pickle
import datetime
import sys

import logging
logger = logging.getLogger('hi')
logger_file = logging.FileHandler(filename='hi.log', mode='w')
logger_stderr = logging.StreamHandler(sys.stderr)
logger_file.setLevel(logging.DEBUG)
logger_stderr.setLevel(logging.INFO)
logger.addHandler(logger_file)
logger.addHandler(logger_stderr)
logger.setLevel(logging.DEBUG)

pickle_dir = 'pickles'
INTER_ENTROPY = 'inter_entropy.pkl'
SUBJECT_FILES = 'subjects_files.pkl'

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
    """
    Calcula la información mutua para un epoch.
    
    Args:
        electrodo_samples matriz de 256 electrodos por 201 samples
    """
    cant_valores = len(valores)
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
    """
    Calcula métricas de información inter-electrodo para el sujeto.

    Las métricas a calcular son promedio y desvío standard de la información 
    mutua inter-electrodo para los trials del sujeto parémetro.

    Args:
        subject_name nombre del sujeto
        subject_matrix matriz de epochs x electrodos x samples

    Returns:
        tupla con
            * nombre del sujeto
            * promedio de la información mutua inter-electrodo
            * desvío standard de la información mutua inter-electrodo
    """
    epochs_info = []
    for epoch in list(range(subject_matrix.shape[0]))[:3]:
        logger.debug('>    Epoch: {}'.format(epoch))
        epochs_info.append(informacion_mutua_epoch(
            subject_matrix[epoch, :, :]))
    return (np.mean(epochs_info), np.std(epochs_info))

if __name__ == '__main__':
    logger.info('Detailed information is being kept in hi.log')
    start_time = datetime.datetime.today()

    with open(os.path.join(pickle_dir, SUBJECT_FILES), 'rb') as f:
        matrix_files = pickle.load(f)  # dataset/P01.mat ...
        matrix_names = [os.path.basename(n).replace('.mat', '') 
                        for n in matrix_files]  # P01.mat ...
    logger.debug('Subject files loaded')

    subject_metrics = []
    for mf, name in list(zip(matrix_files, matrix_names)):
        logger.debug('> Subject: {}'.format(name))
        subject_metrics.append(
            informacion_mutua_sujeto(sio.loadmat(mf)['data']))

    fd = pd.DataFrame.from_records(subject_metrics, index=matrix_names,
                                   columns=['mean', 'std'])

    with open(os.path.join(pickle_dir, INTER_ENTROPY), 'wb') as f:
        pickle.dump(fd, f, protocol=2)

    end_time =  datetime.datetime.today()

    logger.info('Tiempo total: {}'.format(end_time - start_time))
