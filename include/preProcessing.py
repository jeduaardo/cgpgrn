import pandas as pd
import numpy as np
from csaps import csaps
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from os.path import exists
import gc
from time import perf_counter
from scipy import stats
import os
import multiprocessing as mp
from multiprocessing import Pool, freeze_support
from itertools import repeat

class Pseudotime:
    def __init__(self, n_pseudotimes, pseudotime_file, pseudotimes = [], ordered_pseudotimes = []):
        self.pseudotime_file = pseudotime_file
        self.n_pseudotimes = n_pseudotimes
        self.pseudotimes = pseudotimes
        self.ordered_pseudotimes = ordered_pseudotimes
        self.pseudotimesToProcess = []
        self.cells = []
        for n_pseudotime in range(n_pseudotimes):
            pseudotimes.append({})
            ordered_pseudotimes.append([])
    
    def processPseudotime(self, pseudotime):
        allCells = list(self.pseudotime_file.T.columns)
        self.cells = allCells
        for cell in allCells:
            currentPseudotime = self.pseudotime_file.T[cell][pseudotime]
            if str(currentPseudotime) != 'nan':
                if currentPseudotime not in self.pseudotimes[pseudotime].keys():
                    self.pseudotimes[pseudotime][currentPseudotime] = [cell]
                else:
                    self.pseudotimes[pseudotime][currentPseudotime].append(cell)
        allPseudotimes = list(self.pseudotimes[pseudotime].keys())
        self.ordered_pseudotimes = list(self.pseudotimes[pseudotime].keys())
        self.ordered_pseudotimes.sort()
        for pt in allPseudotimes:
            if len(self.pseudotimes[pseudotime][pt]) > 1:
                self.pseudotimesToProcess.append(pt)


class ExpressionData:
    def __init__(self, n_genes, expression_file, pseudotime_object, suffix, genes=[], cells=[]):
        self.n_genes = n_genes
        self.expression_file = expression_file
        self.genes = genes
        self.cells = cells
        self.pseudotime_object = pseudotime_object
        self.suffix = suffix
        self.processedPTs = {}

    def processPTs(self, n_pseudotime):
        for ptValue in self.pseudotime_object.pseudotimesToProcess:
            cellsToProcess = self.pseudotime_object.pseudotimes[n_pseudotime][ptValue]
            processedCell = []
            for cell in cellsToProcess:
                if cell == cellsToProcess[0]:
                    processedCell = list(self.expression_file[cell])
                else:
                    currentCellExpressionData = list(self.expression_file[cell])
                    for i in range(len(currentCellExpressionData)):
                        if currentCellExpressionData[i] != 0.0:
                            processedCell[i] = (processedCell[i] + currentCellExpressionData[i])/2
            self.processedPTs[ptValue] = processedCell        


    def calculateSmooth(self, expressionData, timePoints):
        rs = ShuffleSplit(n_splits = 10, train_size = 0.1, test_size = 0.9, random_state=1345540)
        smooth_errors = {}
        for train, test in rs.split(expressionData):
            train.sort()
            test.sort()
            vector_train = [[],[]]
            vector_test = [[],[]]
            for element in train:
                vector_train[0].append(timePoints[element])
                vector_train[1].append(expressionData[element])
            for element in test:
                vector_test[0].append(timePoints[element])
                vector_test[1].append(expressionData[element])

            smooth_values = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
            for smooths in smooth_values:
                ys = csaps(vector_train[0], vector_train[1], timePoints, smooth=smooths)
                dif_total = 0
                for element in range(len(vector_test[0])):
                    calculated = ys[element]
                    real = vector_test[1][element]
                    dif_total += pow(real - calculated, 2)
                if smooths not in smooth_errors.keys():
                    smooth_errors[smooths] = dif_total
                else:
                    smooth_errors[smooths] += dif_total

            best_smooth = list(smooth_errors.keys())[np.argmin(list(smooth_errors.values()))]

        return best_smooth
    
    def generateSplineCurve(self, expressionData, timePoints, smoothValue, timePointsClean, geneIndex, currentPt):
        currentGeneName = self.expression_file.T.columns[geneIndex]
        spline_expressionData = []
        ys = csaps(timePointsClean, expressionData, timePoints, smooth=smoothValue)


        currentDict2 = {}
        currentDict2[currentGeneName] = ys
        df2 = pd.DataFrame(data = currentDict2, index=timePoints)
                
        gc.collect()
        return currentDict2
  

    def processData(self, n_pseudotime, geneIndex):
        currentGeneName = self.expression_file.T.columns[geneIndex]
        currentCellExpressionData = []
        currentTimePoints = []
        for i in range(len(self.pseudotime_object.ordered_pseudotimes)):
            currentCells = self.pseudotime_object.pseudotimes[n_pseudotime][self.pseudotime_object.ordered_pseudotimes[i]]
            if len(currentCells) == 1:
                currentED = self.expression_file[currentCells[0]][geneIndex]
                if currentED != 0:
                    currentCellExpressionData.append(self.expression_file[currentCells[0]][geneIndex])
                    currentTimePoints.append(self.pseudotime_object.ordered_pseudotimes[i])
                
            else:
                ptValue = self.pseudotime_object.ordered_pseudotimes[i]
                currentED = self.processedPTs[ptValue][geneIndex]
                if currentED != 0:          
                    currentCellExpressionData.append(self.processedPTs[ptValue][geneIndex])
                    currentTimePoints.append(ptValue)

        if len(currentTimePoints) < 2:
            print("Not enought expression values for gene: ", currentGeneName) ##aqui colocar > 1
        else:
            best_smooth = self.calculateSmooth(currentCellExpressionData, currentTimePoints) #descomentar aqui para voltar a determinar o melhor smooth
            currentDF = self.generateSplineCurve(currentCellExpressionData, self.pseudotime_object.ordered_pseudotimes, best_smooth, currentTimePoints, geneIndex, n_pseudotime)

        gc.collect()
        return currentDF
       



def preProcessData(pseudotimeFile, expressionDataFile, suffix):
    '''
    pseudotimeFile: a CSV file containing the cell name and the pseudotime value. For multiple pseudotimes, one file for each pseudotime is required.
    expressionDataFile: a CSV file containing the expression values for each gene and each cell. Genes must be rows and columns must be time points/experimental condition.
    suffix: a str that diferentiates the current experiment
    '''

    allFileNames = []

    print("Initializing...")
    print("Reading pseudotime file...")
    pt_File = pd.read_csv(pseudotimeFile, index_col=0)
    nPseudoTimes = len(pt_File.columns)

    for pts in range(nPseudoTimes):

    
        pseudotime = Pseudotime(1, pt_File)
        print("Processing pseudotimes...")
        pseudotime.processPseudotime(pts)


        print("Reading expression data file...")
        ExprFile = pd.read_csv(expressionDataFile, index_col=0)


        ExprData = ExpressionData(len(ExprFile.T.columns), ExprFile, pseudotime, suffix)

        print("Processing expression data...")
        #Só tem um pseudotime, então o parâmetro nesse caso é só o 0 mesmo
        ExprData.processPTs(pts)


        currentPercentage = -1
        #Para cada gene no conjunto de dados, realiza o processamento (ExprData.processData(n_pseudotime, gene)), n_pseudotime = 0 sempre nesse caso.
        t1_start = perf_counter()
            
        
        with Pool(mp.cpu_count()) as pool:
            M = pool.starmap(ExprData.processData, zip(repeat(pts), [i for i in range(len(ExprFile.T.columns))]))

        fileName = 'splineData_' + ExprData.suffix + "_pt" + str(pts) + '.csv'



        #Generate output file
        
        for l_splineData in M:
            df2 = pd.DataFrame(data = l_splineData, index=ExprData.pseudotime_object.ordered_pseudotimes)
            if os.path.exists(fileName):
                #df2.T.to_csv('splineData.csv', mode='a', header=False)
                df2.T.to_csv(fileName, mode='a', header=False)
            else:
                df2.T.to_csv(fileName, mode='a')
                #df2.T.to_csv('splineData.csv', mode='a')            
        

        t1_stop = perf_counter()
        print("Elapsed time:", t1_stop-t1_start) 

        
        allFileNames.append(fileName)
        
    return allFileNames


