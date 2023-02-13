import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau
import os
from sklearn import preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering
import copy
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
# Davies Bouldin score for K means
from sklearn.metrics import davies_bouldin_score
import random as r
import multiprocessing as mp
from multiprocessing import Pool, freeze_support
from itertools import repeat
from time import perf_counter
import utils as Utils

r.seed(543126345)

def main(problemName, correlationThreshold, splineFileName, nClusters, method, currentPT, Log):
    '''
    Main procedure for choosing the group gene selection technique
    Inputs:
    problemName - name of the problem
    correlationThreshold - correlation threshold value
    splineFileName - name of the spline file
    nClusters - maximum numer of clusters to be considered through the clustering step
    method - group gene selecion technique
    currentPT - current pseudotime suffix
    Log - Log Object
    '''
    PROBLEM_NAME = problemName
    if correlationThreshold != 'None':
        CORRELATION_THRESHOLD = float(correlationThreshold)
    else:
        CORRELATION_THRESHOLD = correlationThreshold
    N_CLUSTERS = nClusters
    METHOD = method
    SPLINE_FILENAME = splineFileName
    CURRENT_PT = currentPT

    splineFile = pd.read_csv(SPLINE_FILENAME, index_col=0)
    allGenes = splineFile.T.columns
    chunksize = 2000
    if chunksize < len(allGenes):
        chunksize = len(allGenes)

    if method == 'Spearman':
        dirName = spearmanCorrelation(splineFile, splineFileName, allGenes, chunksize, CORRELATION_THRESHOLD, PROBLEM_NAME, CURRENT_PT, Log)
        bestNClusters = 'None'
    elif method == 'Pearson':
        dirName = pearsonCorrelation(splineFile, splineFileName, allGenes, chunksize, CORRELATION_THRESHOLD, PROBLEM_NAME, CURRENT_PT, Log)
        bestNClusters = 'None'
    elif method == 'KendallTau':
        dirName = kendallTauCorrelation(splineFile, splineFileName, allGenes, chunksize, CORRELATION_THRESHOLD, PROBLEM_NAME, CURRENT_PT, Log)
        bestNClusters = 'None'
    elif method == 'KMeans':
        dirName, bestNClusters = KMeansClustering(splineFile, splineFileName, allGenes, chunksize, PROBLEM_NAME, int(N_CLUSTERS), CURRENT_PT, Log)
    elif method == 'Agglomerative':
        dirName, bestNClusters = AggClustering(splineFile, splineFileName, allGenes, chunksize, PROBLEM_NAME, int(N_CLUSTERS), CURRENT_PT, Log)
    elif method == 'None':
        dirName = noClustering(splineFile, splineFileName, allGenes, PROBLEM_NAME, CURRENT_PT, chunksize, Log)
        bestNClusters = 'None'

        
    return dirName, bestNClusters


def parallelSpearman(splineFile, splineFileName, allGenes, chunksize, CORRELATION_THRESHOLD, PROBLEM_NAME, CURRENT_PT, dirName, currentGene):

    foundCorrelations = []
    for otherGene in allGenes:
        if abs(spearmanr(splineFile.T[currentGene], splineFile.T[otherGene])[0]) >= CORRELATION_THRESHOLD:
               foundCorrelations.append(otherGene)

    currentFileName = os.path.join(dirName, 'geneNames_' + currentGene + "_" + CURRENT_PT + ".txt")
    outputFile = open(currentFileName, 'w')
    for gene in foundCorrelations:
        outputFile.write(gene)
        outputFile.write("\n")
    outputFile.close()

    CSV_FileName = os.path.join(dirName, 'spline_' + PROBLEM_NAME + '_' + currentGene + "_" + CURRENT_PT + '.csv')
    for chunk in pd.read_csv(splineFileName, chunksize=chunksize, index_col=0):
        allTimePoints = chunk.columns
        for gene in foundCorrelations:
            if gene in chunk.T.columns:
               currentDict = {}
               currentDict[gene] = chunk.T[gene].copy()
               currentDF = pd.DataFrame(data = currentDict, index=allTimePoints)
               if os.path.exists(CSV_FileName):
                   currentDF.T.to_csv(CSV_FileName, mode='a', header=False)
               else:
                   currentDF.T.to_csv(CSV_FileName, mode='a')
                   
            

def spearmanCorrelation(splineFile, splineFileName, allGenes, chunksize, CORRELATION_THRESHOLD, PROBLEM_NAME, CURRENT_PT, Log):

    #dirName = 'spearman'
    dirName = Utils.dict_directories['Spearman']
    Utils.mkdir(dirName)
    '''
    if os.path.exists(dirName) == False:
        os.mkdir(dirName)
    '''

    print("Starting Spearman...")
    Log.register('message', "Starting Spearman...")

    with Pool(mp.cpu_count()) as pool:
        pool.starmap(parallelSpearman, zip(repeat(splineFile), repeat(splineFileName), repeat(allGenes), repeat(chunksize), repeat(CORRELATION_THRESHOLD), repeat(PROBLEM_NAME), repeat(CURRENT_PT), repeat(dirName), allGenes))
      
    print("Spearman successfully performed.")
    Log.register('message', "Spearman successfully performed.")
    return dirName


def parallelPearson(splineFile, splineFileName, allGenes, chunksize, CORRELATION_THRESHOLD, PROBLEM_NAME, CURRENT_PT, dirName, currentGene):

    foundCorrelations = []
    for otherGene in allGenes:
        if abs(pearsonr(splineFile.T[currentGene], splineFile.T[otherGene])[0]) >= CORRELATION_THRESHOLD:
               foundCorrelations.append(otherGene)

    currentFileName = os.path.join(dirName, 'geneNames_' + currentGene + "_" + CURRENT_PT + ".txt")
    outputFile = open(currentFileName, 'w')
    for gene in foundCorrelations:
        outputFile.write(gene)
        outputFile.write("\n")
    outputFile.close()

    CSV_FileName = os.path.join(dirName, 'spline_' + PROBLEM_NAME + '_' + currentGene + "_" + CURRENT_PT + '.csv')
    for chunk in pd.read_csv(splineFileName, chunksize=chunksize, index_col=0):
        allTimePoints = chunk.columns
        for gene in foundCorrelations:
            if gene in chunk.T.columns:
               currentDict = {}
               currentDict[gene] = chunk.T[gene].copy()
               currentDF = pd.DataFrame(data = currentDict, index=allTimePoints)
               if os.path.exists(CSV_FileName):
                   currentDF.T.to_csv(CSV_FileName, mode='a', header=False)
               else:
                   currentDF.T.to_csv(CSV_FileName, mode='a')

def pearsonCorrelation(splineFile, splineFileName, allGenes, chunksize, CORRELATION_THRESHOLD, PROBLEM_NAME, CURRENT_PT, Log):
    #dirName = 'pearson'
    dirName = Utils.dict_directories['Pearson']
    Utils.mkdir(dirName)
    '''
    if os.path.exists(dirName) == False:
        os.mkdir(dirName)
    '''

    print("Starting Pearson...")
    Log.register('message', 'Starting Pearson...')

    with Pool(mp.cpu_count()) as pool:
        pool.starmap(parallelPearson, zip(repeat(splineFile), repeat(splineFileName), repeat(allGenes), repeat(chunksize), repeat(CORRELATION_THRESHOLD), repeat(PROBLEM_NAME), repeat(CURRENT_PT), repeat(dirName), allGenes))
      
    print("Pearson successfully performed.")
    Log.register('message', 'Pearson successfully performed.')
    return dirName                   

def parallelKendallTau(splineFile, splineFileName, allGenes, chunksize, CORRELATION_THRESHOLD, PROBLEM_NAME, CURRENT_PT, dirName, currentGene):

    foundCorrelations = []
    for otherGene in allGenes:
        if abs(kendalltau(splineFile.T[currentGene], splineFile.T[otherGene])[0]) >= CORRELATION_THRESHOLD:
               foundCorrelations.append(otherGene)

    currentFileName = os.path.join(dirName, 'geneNames_' + currentGene + "_" + CURRENT_PT + ".txt")
    outputFile = open(currentFileName, 'w')
    for gene in foundCorrelations:
        outputFile.write(gene)
        outputFile.write("\n")
    outputFile.close()

    CSV_FileName = os.path.join(dirName, 'spline_' + PROBLEM_NAME + '_' + currentGene + "_" + CURRENT_PT + '.csv')
    for chunk in pd.read_csv(splineFileName, chunksize=chunksize, index_col=0):
        allTimePoints = chunk.columns
        for gene in foundCorrelations:
            if gene in chunk.T.columns:
               currentDict = {}
               currentDict[gene] = chunk.T[gene].copy()
               currentDF = pd.DataFrame(data = currentDict, index=allTimePoints)
               if os.path.exists(CSV_FileName):
                   currentDF.T.to_csv(CSV_FileName, mode='a', header=False)
               else:
                   currentDF.T.to_csv(CSV_FileName, mode='a')

def kendallTauCorrelation(splineFile, splineFileName, allGenes, chunksize, CORRELATION_THRESHOLD, PROBLEM_NAME, CURRENT_PT, Log):
    #dirName = 'kendallTau'
    dirName = Utils.dict_directories['KendallTau']
    Utils.mkdir(dirName)
    '''
    if os.path.exists(dirName) == False:
        os.mkdir(dirName)
    '''

    print("Starting Kendall Tau...")
    Log.register('message', 'Starting Kendall Tau...')

    with Pool(mp.cpu_count()) as pool:
        pool.starmap(parallelKendallTau, zip(repeat(splineFile), repeat(splineFileName), repeat(allGenes), repeat(chunksize), repeat(CORRELATION_THRESHOLD), repeat(PROBLEM_NAME), repeat(CURRENT_PT), repeat(dirName), allGenes))
     
    print("Kendall Tau successfully performed.")
    Log.register('message', 'Kendall Tau successfully performed.')
    return dirName                       


def KMeansClustering(splineFile, splineFileName, allGenes, chunksize, PROBLEM_NAME, max_clusters, CURRENT_PT, Log):

    #dirName = 'kmeans'
    dirName = Utils.dict_directories['KMeans']
    Utils.mkdir(dirName)
    '''
    if os.path.exists(dirName) == False:
        os.mkdir(dirName)
    '''
        
    print("Starting KMeans Clustering...")
    Log.register('message', 'Starting KMeans Clustering...')
    
    allKMeansModels = []
    allSilhouettes = []
    for i in range(2, max_clusters+1):
        allKMeansModels.append(KMeans(n_clusters=i, random_state=0))

    allExpressionUniform = []
    for gene in allGenes:
        allExpressionUniform.append(list(splineFile.T[gene]))
    #minmax_scaler = preprocessing.MinMaxScaler()
    #allExpressionUniform = minmax_scaler.fit_transform(allExpressionUniform)
        
    print("Calculating the best number of clusters...")
    Log.register('message', 'Calculating the best number of clusters...')
    for models in allKMeansModels:
        allSilhouettes.append(silhouette_score(allExpressionUniform, models.fit_predict(allExpressionUniform)))

    best_NClusters = np.argmax(allSilhouettes) + 2
    print("Best number of clusters: ", best_NClusters)
    Log.register('info', 'Best number of clusters: ' + str(best_NClusters))

    print("Predicting the labels for n_clusters =", best_NClusters)
    Log.register('message', 'Predicting the labels for n_clusters = ' + str(best_NClusters))
    finalKMeansModel = KMeans(n_clusters=best_NClusters)
    predictedValues = finalKMeansModel.fit_predict(allExpressionUniform)
    
    genesPerCluster = []
    for i in range(best_NClusters):
        genesPerCluster.append([])
    
        
    for i in range(len(predictedValues)):
        currentCluster = predictedValues[i]
        genesPerCluster[currentCluster].append(allGenes[i])

    for i in range(len(genesPerCluster)):
        dirNameCluster = os.path.join(os.getcwd(), dirName)
        dirNameCluster = os.path.join(dirNameCluster, "Cluster" + str(i))
        if os.path.exists(dirNameCluster) == False:
            os.mkdir(dirNameCluster)
        cluster = genesPerCluster[i]

        print("Generating files for cluster ", str(i))
        Log.register('message', 'Generating files for cluster ' + str(i))

        currentFileName = os.path.join(dirNameCluster, "geneNames_" + str(i) + "_" + CURRENT_PT + ".txt")
        outputFile = open(currentFileName, 'w')
        for gene in cluster:
            outputFile.write(gene)
            outputFile.write("\n")
            CSV_FileName = os.path.join(dirNameCluster, 'spline_' + PROBLEM_NAME + '_' + str(i) + "_" + CURRENT_PT + '.csv')
            for chunk in pd.read_csv(splineFileName, chunksize=chunksize, index_col=0):
                allTimePoints = chunk.columns
                currentDict = {}
                currentDict[gene] = chunk.T[gene].copy()
                currentDF = pd.DataFrame(data = currentDict, index=allTimePoints)
                if os.path.exists(CSV_FileName):
                    currentDF.T.to_csv(CSV_FileName, mode='a', header=False)
                else:
                    currentDF.T.to_csv(CSV_FileName, mode='a')
              
        
        
        outputFile.close()        
        
    print("KMeans successfully performed.")
    Log.register('message', 'KMeans successfully performed.')
    return dirName, best_NClusters






def noClustering(splineFile, splineFileName, allGenes, PROBLEM_NAME, CURRENT_PT, chunksize, Log):

    dirName = 'noClustering'
    Utils.mkdir(dirName)
    '''
    if os.path.exists(dirName) == False:
        os.mkdir(dirName)
    '''
        
    print("No clustering method chosen.")
    Log.register('message', 'No clustering method chosen.')

    currentFileName = os.path.join(dirName, 'geneNames_' + PROBLEM_NAME + "_" + CURRENT_PT + ".txt")
    outputFile = open(currentFileName, 'w')
    for gene in allGenes:
        outputFile.write(gene)
        outputFile.write("\n")
    outputFile.close()
        
    CSV_FileName = os.path.join(dirName, 'spline_' + PROBLEM_NAME + "_" + CURRENT_PT + '.csv')
    for chunk in pd.read_csv(splineFileName, chunksize=chunksize, index_col=0):
        allTimePoints = chunk.columns
        #print(allTimePoints)
        for gene in chunk.T.columns:
            currentDict = {}
            currentDict[gene] = chunk.T[gene].copy()
            currentDF = pd.DataFrame(data = currentDict, index=allTimePoints)
            if os.path.exists(CSV_FileName):
                currentDF.T.to_csv(CSV_FileName, mode='a', header=False)
            else:
                currentDF.T.to_csv(CSV_FileName, mode='a')


    print("End of no clustering procedure.")
    Log.register('message', 'End of no clustering procedure.')

    return dirName  



def AggClustering(splineFile, splineFileName, allGenes, chunksize, PROBLEM_NAME, max_clusters, CURRENT_PT, Log):
    
    #dirName = 'aggclustering'
    dirName = Utils.dict_directories['Agglomerative']
    Utils.mkdir(dirName)
    '''
    if os.path.exists(dirName) == False:
        os.mkdir(dirName)
    '''
        
    print("Starting Agglomerative Clustering...")
    Log.register('message', 'Starting Agglomerative Clustering...')
    
    
    allKMeansModels = []
    allSilhouettes = []


    configurations = [['ward', 'euclidean']]
    linkages = ['complete', 'average', 'single']
    affinities = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    for linkage in linkages:
        for affinity in affinities:
            vlocal = []
            vlocal.append(linkage)
            vlocal.append(affinity)
            configurations.append(vlocal)
    #print(configurations)

    for configuration in configurations:
        for i in range(2, max_clusters+1):
            allKMeansModels.append(AgglomerativeClustering(n_clusters=i, linkage=configuration[0], affinity=configuration[1]))
                                

    allExpressionUniform = []
    for gene in allGenes:
        allExpressionUniform.append(list(splineFile.T[gene]))
    minmax_scaler = preprocessing.MinMaxScaler()
    allExpressionUniform = minmax_scaler.fit_transform(allExpressionUniform)
        
    print("Calculating the best number of clusters...")
    Log.register('message', 'Calculating the best number of clusters...')


    allScores = []
    for models in allKMeansModels:
        allScores.append(davies_bouldin_score(allExpressionUniform, models.fit_predict(allExpressionUniform)))
    print(allScores)


    bestScoreValue = allScores[np.argmin(allScores)]
    allBestScores = []
    for i in range(len(allScores)):
        if allScores[i] == bestScoreValue:
            allBestScores.append(i)


    allBestModels = []
    for score in allBestScores:
        allBestModels.append(allKMeansModels[score])


    bestModel = allBestModels[r.randint(0, len(allBestModels)-1)]

    best_NClusters = bestModel.n_clusters


    print("Best Model: ", bestModel)
    Log.register('info', 'Best model: ' + str(bestModel))
    print("Best number of clusters: ", best_NClusters)
    Log.register('info', 'Best number of clusters: ' + str(best_NClusters))

    print("Predicting the labels...")
    Log.register('message', 'Predicting the labels...')
    finalKMeansModel = bestModel
    predictedValues = finalKMeansModel.fit_predict(allExpressionUniform)
    
    genesPerCluster = []
    for i in range(best_NClusters):
        genesPerCluster.append([])
    
        
    for i in range(len(predictedValues)):
        currentCluster = predictedValues[i]
        genesPerCluster[currentCluster].append(allGenes[i])


    for i in range(len(genesPerCluster)):
        dirNameCluster = os.path.join(os.getcwd(), dirName)
        dirNameCluster = os.path.join(dirNameCluster, "Cluster" + str(i))
        
        if os.path.exists(dirNameCluster) == False:
            os.mkdir(dirNameCluster)
        cluster = genesPerCluster[i]

        print("Generating files for cluster ", str(i))
        Log.register('message', 'Generating files for cluster ' + str(i))

        currentFileName = os.path.join(dirNameCluster, "geneNames_" + str(i) + "_" + CURRENT_PT + ".txt")
        outputFile = open(currentFileName, 'w')
        for gene in cluster:
            outputFile.write(gene)
            outputFile.write("\n")
            CSV_FileName = os.path.join(dirNameCluster, 'spline_' + PROBLEM_NAME + '_' + str(i) + "_" + CURRENT_PT + '.csv')
            for chunk in pd.read_csv(splineFileName, chunksize=chunksize, index_col=0):
                allTimePoints = chunk.columns
                currentDict = {}
                currentDict[gene] = chunk.T[gene].copy()
                currentDF = pd.DataFrame(data = currentDict, index=allTimePoints)
                if os.path.exists(CSV_FileName):
                    currentDF.T.to_csv(CSV_FileName, mode='a', header=False)
                else:
                    currentDF.T.to_csv(CSV_FileName, mode='a')
              
        
        
        outputFile.close()        
        
    print("Agglomerative Clustering successfully performed.")
    Log.register('message', 'Agglomerative Clustering successfully performed.')



    return dirName, best_NClusters
