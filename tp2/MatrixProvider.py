# -*- coding: utf-8 -*-
"""Módulo con clase para obtener las matrices.

El proveedor da una interface cómoda para obtener matrices. Además se encarga
de mostrar barra de progreso y contar el tiempo de trabajo.

Se puede recorrer usando:

for idx, nombre, path, tipo in MatrixProvider(n=6, unico=None):
    ...

Donde:
    idx es el número de ejemplo recorrido
    nombre es el nombre de la matriz (ej: P05.mat)
    path es la ruta para abrir el archivo (ej: dataset/P05.mat)
    tipo es el tipo del sujeto. Tiene valor 'P' o 'S'
"""

from __future__ import print_function, division
import os
import time
import glob
import collections
import random
from ipywidgets import FloatProgress
from IPython.display import display

class MatrixProvider:

    MatrixData = collections.namedtuple('MatrixData', ['name', 'path', 'tipo'])
    DATASET_DIR = 'dataset'
    all_matrices = glob.glob(os.path.join(DATASET_DIR, '*.mat'))

    def _matrices_data_by_type(type, all_matrices, dataset_dir, matrix_data):
        names = [os.path.basename(path) for path in all_matrices]
        return [matrix_data(name, path, type)
                for name, path in zip(names, all_matrices) 
                if name.startswith(type)]


    # Diccionario con todas los nombres de agrupadas por tipo
    MATRICES = {
        'P': _matrices_data_by_type('P', all_matrices, DATASET_DIR, MatrixData),
        'S': _matrices_data_by_type('S', all_matrices, DATASET_DIR, MatrixData)
    }  


    def __init__(self, n=6, balanceado=True, unico=None, mats=None):
        """
        Crea un proveedor que devuelve 'n' directorios de matrices.

        Se puede recorrer usando:

        for idx, nombre, path, tipo in MatrixProvider(n=6, unico=None):
            ...

        Donde:
            idx es el número de ejemplo recorrido
            nombre es el nombre de la matriz (ej: P05.mat)
            path es la ruta para abrir el archivo (ej: dataset/P05.mat)
            tipo es el tipo del sujeto. Tiene valor 'P' o 'S'

        Parámetros:
            n Cantidad de sujetos a usar
            balaneado Si los sujetos deben estar balanceados entre P y S
            unico Si vale P o S, solo usa sujetos de ese tipo
            mats Lista fija de archivos a usar 
        """
        self.n = n if n > 0 else len(self.all_matrices)
        if mats is not None:
            self.pool = []
            for m in mats:
                name = os.path.basename(m)
                tipo = name[0]
                self.pool.append(self.MatrixData(name, m, tipo))
        else:
            self.balanceado = balanceado
            self.unico = unico
           
            if unico is None:
                if self.balanceado:
                    half = n // 2
                    pool = random.sample(self.MATRICES['P'], half)
                    pool.extend(random.sample(self.MATRICES['S'], n - half))
                else:
                    pool = random.sample(MATRICES['P'] + MATRICES['S'], n)
            else:
                pool = random.sample(MATRICES[unico], n)
    
            self.pool = pool
        
        self.pbar = FloatProgress(min=0, max=self.n)
        display(self.pbar)
        self.starttime = time.time()

    def __iter__(self):
        self.iter = iter(self.pool)
        self.idx = 0
        return self

    def __next__(self):
        self.pbar.value += 1
        try:
            next_val = self.iter.next()
            idx = self.idx
            self.idx += 1
            return (idx, next_val[0], next_val[1], next_val[2])
        except StopIteration as si:
            self.stoptime = time.time()
            print('Tiempo total: {:.2f}s'.format(self.stoptime -
                                                 self.starttime))
            raise si

    def next(self):
        return self.__next__()
