import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau
import os
from sklearn import preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering
import copy
from sklearn.metrics import silhouette_score
import numpy as np
import argparse
import pandas as pd
from itertools import product, permutations, combinations, combinations_with_replacement
import numpy as np
import glob
import sys
from warnings import simplefilter
import subprocess
from multiprocessing import Pool, freeze_support
from time import perf_counter
import math


sys.path.insert(0, os.curdir + '/include')

import preProcessing as PP
import clusterData as CD
import discretizeData as DD
import generateOutputs as GO
import utils as Utils
import makeFile as MKFile
import objects as objs

simplefilter(action='ignore', category=FutureWarning)


def get_parser() -> argparse.ArgumentParser:
    '''
    :return: an argparse ArgumentParser object for parsing command
        line parameters
    '''
    parser = argparse.ArgumentParser(description='Complete pre-processing and script generator for CGPGRN framework.', epilog = 'Example usage to process data: python completePreProcessing.py -cm KMeans -nc 10 -e 500nTF-ExpressionData.csv -p PseudoTime.csv -pn hHep -s 500nTF -d BiKMeans -r 5')

    parser.add_argument('-cm', dest = 'clusterMethod', type = str,
                        default = 'None',
                        help='Cluster method used for grouping genes. \n\n Correlation Approaches (must define correlation threshold -t): Spearman, Person and KendallTau. \n Clustering Approaches (must define the number of clusters -nc): KMeans, Agglomerative \n'
                        )
    
    parser.add_argument('-nc', dest = 'n_clusters', type = str,
                        default = 'None',
                        help='The maximum number of clusters to be tested with clustering methods. \n')

    parser.add_argument('-t', dest = 'correlation_threshold', type = str,
                        default = 'None',
                        help='The correlation coefficient threshold for correlation methods. \n')

    parser.add_argument('-e', dest = 'expressionDataFile', type = str,
                        default = 'ExpressionData.csv',
                        help='Path to expression data file. Required. \n')

    parser.add_argument('-p', dest = 'pseudotimeFile', type = str,
                        default = 'PseudoTime.csv',
                        help='Path to pseudotime file. Required. \n')

    parser.add_argument('-pn', '--problemName', dest = 'problemName', type = str,
                        default = 'None',
                        help='Problem Name. Required. \n')

    parser.add_argument('-s', '--suffix', dest = 'suffix', type = str,
                        default = 'None',
                        help='Suffix for generated files. Required. \n')        

    parser.add_argument('-d', '--discretization', dest = 'discretizationApproach', type = str,
                        default = 'None',
                        help='Discretization approach for spline data. Required. \n\n Discretization approaches: Mean, Median, TSD, EFD, BiKMeans  \n')  

    parser.add_argument('-r', '--independentRuns', dest = 'independentRuns', type = int,
                        default = 1,
                        help='Number of independent runs of CGP.  \n')  

    parser.add_argument('-fd', dest = 'fullDiscretization', action='store_true', default = False,
                        help='Discretize every spline generated file. If False, only the main spline file will be discretized. Default = False.\n')

    parser.add_argument('-sf', '--splineFile', dest = 'argSplineFile', type = str,
                        default = 'None',
                        help='Spline File. If you pass a spline file this script will not pre-process the data and the spline file passed will be used instead.  \n')  

    parser.add_argument('-sl', dest = 'splineList', action='store_true', default = False,
                        help='When True, -sf must be a txt file with each spline file per line. Default = False.\n')
 

    parser.add_argument('-run', dest = 'runCGP', action='store_true', default = False,
                        help='Automatically run CGPGRN inference procedure after preprocessing. Default = False.\n')


    parser.add_argument('-kd', dest = 'keepData', action='store_true', default = False,
                        help='Keep Source and Spline Data after the complete pipeline execution. Default = False.\n')

    parser.add_argument('-fullTT', dest = 'fullTT', action='store_true', default = False,
                        help='Use full truth tables for clustering methods. Default = False.\n')

    parser.add_argument('-gsl', dest = 'generateSplineList', action='store_true', default = False,
                        help='Generate spline list file if the number of spline files are greater than one. Default = False.\n')

    parser.add_argument('-cgpn', '--nodes', dest = 'cgpNodes', type = str,
                        default = 'None',
                        help='The number of nodes to encode CGPGRN inference algorithm genotype.\n')

    parser.add_argument('-cgpg', '--gens', dest = 'cgpGens', type = str,
                        default = 'None',
                        help='The maximum number of generations for CGPGRN inference algorithm evolutionary search.\n')  
    
        

    return parser                        


def parse_arguments():
    '''
    Initialize a parser and use it to parse the command line arguments
    :return: parsed dictionary of command line arguments
    '''
    parser = get_parser()
    opts = parser.parse_args()

    return opts


if __name__ == '__main__':


    t0_General = perf_counter()
    
    
    correlationMethods = Utils.correlationMethods    
    clusteringMethods = Utils.clusteringMethods    
    possibleDiscretizationApproaches = Utils.possibleDiscretizationApproaches


    opts = parse_arguments()

    logFileName = "LogFile_" + vars(opts)['problemName'] + ".txt"
    Log = Utils.Log(vars(opts), logFileName)
    Log.register('initial')

    Utils.verifyArgs(opts, Log)

    CGPGRN = objs.CGPGRN(opts, Log)
    splineDataNames = CGPGRN.performSplineStep()
    #print(splineDataNames)



    
    problemName = opts.problemName
    suffix = opts.suffix
    clusterMethod = opts.clusterMethod
    numberOfClusters = opts.n_clusters
    correlationThreshold = opts.correlation_threshold
    numberOfClusters = opts.n_clusters
    correlationThreshold = opts.correlation_threshold
    expressionDataFile = opts.expressionDataFile
    pseudotimeFile = opts.pseudotimeFile
    argSplineFile = opts.argSplineFile
    discretizationApproach = opts.discretizationApproach
    independentRuns = opts.independentRuns
    fullDiscretization = opts.fullDiscretization
    splineList = opts.splineList
    runCGP = opts.runCGP
    keepData = opts.keepData
    fullTT = opts.fullTT
    generateSplineList = opts.generateSplineList
    cgpNodes = opts.cgpNodes
    cgpGens = opts.cgpGens

    #print(fullTT)
    #print(type(fullTT))

    #exit()


    all_SPFiles = []

    t0_spline = perf_counter()

    '''
    if argSplineFile == 'None':
        if Utils.verifyNumberOfClusters([expressionDataFile], numberOfClusters) == True:
            print("Generating spline file...")
            Log.register('message', "Generating spline file...")
            splineDataNames = PP.preProcessData(pseudotimeFile, expressionDataFile, suffix)
        else:
            print("The maximum number of clusters must be lower than the total number of genes.")
            Log.register('error', "The maximum number of clusters must be lower than the total number of genes.")
            Log.register('info', "You set the number of clusters as " + str(numberOfClusters))            
            exit()        
    else:
        if splineList == False:
            verifyClusterConstraint = Utils.verifyNumberOfClusters([argSplineFile], numberOfClusters)
            if verifyClusterConstraint == True:
                print("Using spline file passed as an argument...")
                Log.register('message', "Using spline file passed as an argument...")
                splineDataNames = [argSplineFile]
            elif verifyClusterConstraint == 'nf':
                print("Spline file not found.")
                Log.register('error', "Spline file not found.")
                exit()
            else:
                print("The maximum number of clusters must be lower than the total number of genes.")
                Log.register('error', "The maximum number of clusters must be lower than the total number of genes.")
                exit()
        else:
            splineDataNames = Utils.getSplineFilesNames(argSplineFile)
            verifyClusterConstraint = Utils.verifyNumberOfClusters(splineDataNames, numberOfClusters)          
            if verifyClusterConstraint == False:
                print("The maximum number of clusters must be lower than the total number of genes.")
                Log.register('error', "The maximum number of clusters must be lower than the total number of genes.")
                exit()
            elif verifyClusterConstraint == 'nf':
                print("Spline file not found.")
                Log.register('error', "Spline file not found.")
                exit()
    '''

    tf_spline = perf_counter()

    
    
    Log.register('time', str(tf_spline - t0_spline))
    Log.register('message', "End of spline step.")
    print("End of spline step.")


    all_SPFiles.append(tf_spline - t0_spline)

    mainDirectory = os.getcwd()

    all_discretizedFiles = []
    all_genesNamesFiles = []
    all_directories = []
    all_dirExecutions = []
    additional_discretizedFiles = []

    all_CDTimes = []
    all_DDTimes = []

    #if generateSplineList == True:
    #    Utils.generateSplineList(splineDataNames, Log, problemName)

    print("Processing spline data...")
    Log.register('message', "Processing spline data...")


    for splineDataName in splineDataNames:
        splitName = splineDataName.split("_")
        currentPseudoTime = splitName[len(splitName)-1].replace('.csv', '')

        print("Current pseudotime: ", currentPseudoTime)
        Log.register('message', "Current pseudotime " + str(currentPseudoTime))
        print("Starting clustering step for pseudotime " + str(currentPseudoTime))
        Log.register('message', "Starting clustering step for pseudotime " + str(currentPseudoTime))

        t0_CD = perf_counter()
        #dirName, bestNClusters = CD.main(problemName, correlationThreshold, splineDataName, numberOfClusters, clusterMethod, currentPseudoTime, Log)
        dirName, bestNClusters = CGPGRN.performClusteringStep(splineDataName, currentPseudoTime)
        tf_CD = perf_counter()

        print("End clustering step for pseudotime " + str(currentPseudoTime))
        Log.register('message', "End clustering step for pseudotime " + str(currentPseudoTime))
        Log.register('time', str(tf_CD - t0_CD))

        all_CDTimes.append(tf_CD-t0_CD)

        
        discretizedFiles = []
        genesNamesFiles = []
        directories = []


        t0_DD = perf_counter()
        print("Discretizing data using " + str(discretizationApproach) + " for pseudotime " + str(currentPseudoTime))
        Log.register('message', "Discretizing data using " + str(discretizationApproach) + " for pseudotime " + str(currentPseudoTime))


        
        it_genesNamesFiles, it_directories, it_additional_discretizedFiles, it_discretizedFiles = CGPGRN.performDiscretizationStep(dirName, bestNClusters, currentPseudoTime, splineDataName)

        for it in it_genesNamesFiles:
            genesNamesFiles.append(it)
        for it in it_directories:
            directories.append(it)
        for it in it_additional_discretizedFiles:
            additional_discretizedFiles.append(it)
        for it in it_discretizedFiles:
            discretizedFiles.append(it)
        '''
        

        
        if fullDiscretization == True:
            Log.register('message', "Using full discretization...")
            if clusterMethod in clusteringMethods:
                localClusterDir = Utils.dict_directories[clusterMethod]
                    
                for cluster in range(bestNClusters):
                    print("Discretizing data for cluster " + str(cluster))
                    Log.register('message', "Discretizing data for cluster " + str(cluster))

                    currentDir1 = os.path.join(os.getcwd(), localClusterDir)

                    currentDir = currentDir1 + "/Cluster" + str(cluster)
                    
                    currentFile = currentDir + "/spline_" + problemName + "_" + str(cluster) + "_" + currentPseudoTime + ".csv"
                    currentGenesNamesFiles = currentDir + "/geneNames_" + str(cluster) + "_" + currentPseudoTime + ".txt"
                    genesNamesFiles.append(currentGenesNamesFiles)
                    directories.append(currentDir)                    
                    currentOutFile = currentDir + Utils.dict_discretizationPrefixes[discretizationApproach] + problemName + "_" + str(cluster) + "_" + currentPseudoTime + ".csv"
                    if discretizationApproach == 'BiKMeans':
                        if Utils.verifyNumberOfClusters([localClusterDir+"/Cluster"+str(cluster) + "/spline_"+problemName+"_"+str(cluster)+"_"+currentPseudoTime+".csv"], 3) != True:
                            print("The spline data does not contain the minimum number of needed genes (3). Using not full discretization instead.")
                            Log.register('warning', "The spline data does not contain the minimum number of needed genes (3). Using not full discretization instead.")
                            Log.register('info', "Spline File Name: " + str(currentFile))
                            #exit()

                            currentTempFile = splineDataName
                            currentTempOutFile = Utils.dict_discretizationPrefixes[discretizationApproach] + problemName + "_full_" + currentPseudoTime + ".csv"
                            if not os.path.exists(currentTempOutFile):
                                DD.discretizationProcedure(discretizationApproach, currentTempOutFile, currentTempFile, problemName, currentPseudoTime)
                                #additional_discretizedFiles.append(os.path.join(os.getcwd(), currentTempOutFile))
                                additional_discretizedFiles.append(currentTempOutFile)



                            currentOutFile2 = currentDir + Utils.dict_discretizationPrefixes[discretizationApproach] + problemName + "_" + str(cluster) + "_" + currentPseudoTime + ".csv"
                               
                            DD.generateNotFullDiscretizationData(currentTempOutFile, currentOutFile2, currentGenesNamesFiles)

                            discretizedFiles.append(currentOutFile2)
                        else:

                            DD.discretizationProcedure(discretizationApproach, currentOutFile, currentFile, problemName, currentPseudoTime)

                            discretizedFiles.append(currentOutFile)
                    else:
                        DD.discretizationProcedure(discretizationApproach, currentOutFile, currentFile, problemName, currentPseudoTime)

                        discretizedFiles.append(currentOutFile)
                    

                
                               
            elif clusterMethod in correlationMethods:

                currentDir = dirName + "/"

                readSplineData = pd.read_csv(splineDataName, index_col=0)
                allGenes = readSplineData.T.columns
                for gene in allGenes:
                    Log.register('message', "Discretizing data for gene " + str(gene))
                    currentFile = currentDir + "spline_" + problemName + "_" + str(gene) + "_" + currentPseudoTime + ".csv"
                    currentGenesNamesFiles = currentDir + "geneNames_" + str(gene) + "_" + currentPseudoTime + ".txt"
                    genesNamesFiles.append(currentGenesNamesFiles)
                    directories.append(currentDir)
                    currentOutFile = currentDir + Utils.dict_discretizationPrefixes[discretizationApproach] + problemName + "_" + str(gene) + "_" + currentPseudoTime + ".csv"
                    DD.discretizationProcedure(discretizationApproach, currentOutFile, currentFile, problemName, currentPseudoTime)
  
                    discretizedFiles.append(currentOutFile)
            else:

                currentDir = dirName + "/"

                readSplineData = pd.read_csv(splineDataName, index_col=0)
                allGenes = readSplineData.T.columns

                currentFile = currentDir + "spline_" + problemName + "_" + currentPseudoTime + ".csv"
                
                currentGenesNamesFiles = currentDir + "geneNames_" + problemName + "_" + currentPseudoTime + ".txt"
                genesNamesFiles.append(currentGenesNamesFiles)
                directories.append(currentDir)
                currentOutFile = currentDir + Utils.dict_discretizationPrefixes[discretizationApproach] + problemName + "_" + currentPseudoTime + ".csv"
                DD.discretizationProcedure(discretizationApproach, currentOutFile, currentFile, problemName, currentPseudoTime)
     
                discretizedFiles.append(currentOutFile)            
                
                    
                   
        else:
            #print("nao eh full")
            Log.register('message', "Using not full discretization...")
            if clusterMethod in clusteringMethods:
                localClusterDir = Utils.dict_directories[clusterMethod]

                currentFile = splineDataName
                currentOutFile = Utils.dict_discretizationPrefixes[discretizationApproach] + problemName + "_full_" + currentPseudoTime + ".csv"
                DD.discretizationProcedure(discretizationApproach, currentOutFile, currentFile, problemName, currentPseudoTime)
     
                

                for cluster in range(bestNClusters):

                    print("Discretizing data for cluster " + str(cluster))
                    Log.register('message', "Discretizing data for cluster " + str(cluster))
                    
                    currentDir1 = os.path.join(os.getcwd(), localClusterDir)
                    currentDir = currentDir1 + "/Cluster" + str(cluster)
                    currentGenesNamesFiles = currentDir + "/geneNames_" + str(cluster) + "_" + currentPseudoTime + ".txt"
                    genesNamesFiles.append(currentGenesNamesFiles)
                    directories.append(currentDir)
                    currentOutFile2 = currentDir + Utils.dict_discretizationPrefixes[discretizationApproach] + problemName + "_" + str(cluster) + "_" + currentPseudoTime + ".csv"
                       
                    DD.generateNotFullDiscretizationData(currentOutFile, currentOutFile2, currentGenesNamesFiles)

                    discretizedFiles.append(currentOutFile2)
             
            elif clusterMethod in correlationMethods:

                currentFile = splineDataName
                currentOutFile = Utils.dict_discretizationPrefixes[discretizationApproach] + problemName + "_full_" + currentPseudoTime + ".csv"
                DD.discretizationProcedure(discretizationApproach, currentOutFile, currentFile, problemName, currentPseudoTime)

                currentDir = dirName + "/"

                readSplineData = pd.read_csv(splineDataName, index_col=0)
                allGenes = readSplineData.T.columns
                for gene in allGenes:
                    Log.register('message', "Discretizing data for gene " + str(gene))
                    currentFile = currentDir + "spline_" + problemName + "_" + str(gene) + "_" + currentPseudoTime + ".csv"
                    currentGenesNamesFiles = currentDir + "geneNames_" + str(gene) + "_" + currentPseudoTime + ".txt"
                    genesNamesFiles.append(currentGenesNamesFiles)
                    directories.append(currentDir)
                    currentOutFile2 = currentDir + Utils.dict_discretizationPrefixes[discretizationApproach] + problemName + "_" + str(gene) + "_" + currentPseudoTime + ".csv"

                    DD.generateNotFullDiscretizationData(currentOutFile, currentOutFile2, currentGenesNamesFiles)


                    
                    discretizedFiles.append(currentOutFile2)


            else:

                currentDir = dirName + "/"

                readSplineData = pd.read_csv(splineDataName, index_col=0)
                allGenes = readSplineData.T.columns

                currentFile = currentDir + "spline_" + problemName + "_" + currentPseudoTime + ".csv"
                
                currentGenesNamesFiles = currentDir + "geneNames_" + problemName + "_" + currentPseudoTime + ".txt"
                genesNamesFiles.append(currentGenesNamesFiles)
                directories.append(currentDir)
                currentOutFile = currentDir + Utils.dict_discretizationPrefixes[discretizationApproach] + problemName + "_" + currentPseudoTime + ".csv"
                DD.discretizationProcedure(discretizationApproach, currentOutFile, currentFile, problemName, currentPseudoTime)
      

                discretizedFiles.append(currentOutFile)
        '''


        print("Data discretization successfully performed.")
        Log.register('message', "Data discretization sucessfully performed.")
    
        tf_DD = perf_counter()
        Log.register('time', str(tf_DD-t0_DD))
        all_DDTimes.append(tf_DD-t0_DD)

        for DFs in discretizedFiles:
            all_discretizedFiles.append(DFs)
        for GNFs in genesNamesFiles:
            all_genesNamesFiles.append(GNFs)
        for DIRS in directories:
            all_directories.append(DIRS)


        print("Generating truth tables...")
        Log.register('message', 'Generating truth tables...')
        

        all_GOTimes = []
        

        for i in range(len(directories)):
           Log.register('message', "Generating truth tables for directory " + str(i+1) + "/" + str(len(directories)))
           t0_GO = perf_counter()
           truthTableDirectory = directories[i] + "/truth_tables" + "_" + currentPseudoTime + "/"
           if not(os.path.exists(truthTableDirectory)):
               os.mkdir(truthTableDirectory)
           if clusterMethod not in correlationMethods and clusterMethod != 'None':
               if discretizationApproach == 'BiKMeans':
                   #if fullTT == False:
                   #GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 1, truthTableDirectory, 1)
                   if fullTT == True:
                       Log.register('message', "Using full truth table...")
                       GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 1, truthTableDirectory, 0, 1) ##O zero significa que não é um output por gene
                   else:
                       Log.register('message', "Using one output per gene...")
                       GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 1, truthTableDirectory, 1, 1)
               else:
                   #GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 0, truthTableDirectory, 1)
                   if fullTT == True:
                       Log.register('message', "Using full truth table...")
                       GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 0, truthTableDirectory, 0, 1)
                   else:
                       Log.register('message', "Using one output per gene...")
                       GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 1, truthTableDirectory, 1, 1)
           elif clusterMethod in correlationMethods:
               currentTargetGene = genesNamesFiles[i].split("_")
               currentTargetGene = currentTargetGene[1]
               currentTargetGene = currentTargetGene.replace('.txt', '')
               targetGenes = []
               targetGenes.append(currentTargetGene)
               if discretizationApproach == 'BiKMeans':
                   #GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 1, truthTableDirectory, 0, targetGenes)
                   GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 1, truthTableDirectory, 1, 0, targetGenes)
               else:
                   #GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 0, truthTableDirectory, 0, targetGenes)
                   GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 0, truthTableDirectory, 1, 0, targetGenes)
           else:
               print("ENTROU AQUI") #clusterMethod = 'None' aqui
               if discretizationApproach == 'BiKMeans':
                   if fullTT == True:
                       Log.register('message', "Using full truth table...")
                       GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 1, truthTableDirectory, 0, 1)
                   else:
                       Log.register('message', "Using one output per gene...")
                       GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 1, truthTableDirectory, 1, 1)
               else:
                   #GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 0, truthTableDirectory, 1)
                   if fullTT == True:
                       Log.register('message', "Using full truth table...")
                       GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 0, truthTableDirectory, 0, 1)
                   else:
                       Log.register('message', "Using one output per gene...")
                       GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 1, truthTableDirectory, 1, 1)

           tf_GO = perf_counter()
           Log.register('time', str(tf_GO - t0_GO))
           all_GOTimes.append(tf_GO - t0_GO)

        '''    
        if os.path.exists('executions_parallel') == False:
            os.mkdir('executions_parallel')
        if os.path.exists('time_counting') == False:
            os.mkdir('time_counting')
        '''
        Utils.mkdir('executions_parallel')
        Utils.mkdir('time_counting')

        all_BashTimes = []
        all_geneNamesDirFULLTT = []
        all_geneNamesDir = []

        t0_Bash = perf_counter()

        print("Generating bash scripts...")
        Log.register('message', "Generating bash scripts...")
        bashScripts = []

        if clusterMethod in correlationMethods:
            for i in range(independentRuns):
                currentExecution = 'exe_' + str(i+1) + "_" + currentPseudoTime
                all_dirExecutions.append(currentExecution)
                Utils.mkdir('executions_parallel/' + currentExecution)
                Utils.mkdir('time_counting/' + currentExecution)
                '''
                if os.path.exists('executions_parallel/' + currentExecution) == False:
                    os.mkdir('executions_parallel/' + currentExecution)
                if os.path.exists('time_counting/' + currentExecution) == False:
                    os.mkdir('time_counting/' + currentExecution)
                '''
                currentOutputFileName = problemName + '_exe' + str(i+1) + "_" + currentPseudoTime + ".sh"
                bashScripts.append(currentOutputFileName)
                openOutFile = open(currentOutputFileName, 'w')
                for j in range(len(genesNamesFiles)):
                    currentGeneFileName = genesNamesFiles[j]
                    currentSeed = (i * len(genesNamesFiles)) + j
                    currentGeneDirectory = currentGeneFileName
                    currentGene = currentGeneFileName.split('_')[1].replace('.txt', '')
                    currentTruthTableDirectory = dirName + "/truth_tables" + "_" + currentPseudoTime
                    all_geneNamesDir.append(currentGeneDirectory)
                    if os.name == 'nt':
                        outputString = "./progW " + currentGeneDirectory + " " + currentExecution + " " + problemName + " " + currentGene + " " + str(currentSeed) + " " + currentTruthTableDirectory + " 0 " + "\n"
                    else:
                        outputString = "./progL " + currentGeneDirectory + " " + currentExecution + " " + problemName + " " + currentGene + " " + str(currentSeed) + " " + currentTruthTableDirectory + " 0 " + "\n"
                    openOutFile.write(outputString)
                openOutFile.close()
        elif clusterMethod in clusteringMethods:#== 'KMeans' or clusterMethod == 'Agglomerative':
            for i in range(bestNClusters):
                currentClusterDir = dirName + '/Cluster' + str(i)
                currentGeneNamesDir = currentClusterDir + '/geneNames_' + str(i) + "_" + currentPseudoTime + '.txt'
                os.chdir(currentClusterDir + '/truth_tables' + "_" + currentPseudoTime)
                allCurrentTruthTables = glob.glob('*.txt')
                os.chdir('../../..')
                for j in range(independentRuns):
                    currentExecution = 'exe_' + str(j+1) + '_Cluster' + str(i) + "_" + currentPseudoTime
                    all_dirExecutions.append(currentExecution)
                    Utils.mkdir('executions_parallel/' + currentExecution)
                    Utils.mkdir('time_counting/' + currentExecution)
                    '''
                    if os.path.exists('executions_parallel/' + currentExecution)  == False:
                        os.mkdir('executions_parallel/' + currentExecution)
                    if os.path.exists('time_counting/' + currentExecution) == False:
                        os.mkdir('time_counting/' + currentExecution)
                    '''                        
                    currentOutputFileName = problemName + '_' + currentExecution + ".sh"
                    bashScripts.append(currentOutputFileName)
                    allGenesInCurrentCluster = []
                    all_geneNamesDir.append(currentGeneNamesDir)
                    openGeneFile = open(currentGeneNamesDir, "r")
                    for line in openGeneFile:
                        allGenesInCurrentCluster.append(line.strip())
                        
                    openOutFile = open(currentOutputFileName, 'w')
                    if fullTT == False:
                        for k in range(len(allGenesInCurrentCluster)):
                            currentSeed = (j * len(allGenesInCurrentCluster)) + k
                            currentGene = allGenesInCurrentCluster[k]
                            currentTruthTableDirectory = dirName + '/Cluster' + str(i) + '/truth_tables' + "_" + currentPseudoTime
                            if os.name == 'nt':
                                outputString = "./progW " + currentGeneNamesDir + " " + currentExecution + " " + problemName + " " + currentGene + " " + str(currentSeed) + " " + currentTruthTableDirectory + " 0 " + "\n"
                            else:
                                outputString = "./progL " + currentGeneNamesDir + " " + currentExecution + " " + problemName + " " + currentGene + " " + str(currentSeed) + " " + currentTruthTableDirectory + " 0 " + "\n"                    
                            
                            openOutFile.write(outputString)
                    else:
                        all_geneNamesDirFULLTT.append(currentGeneNamesDir)
                        currentSeed = j + independentRuns*i
                        currentGene = 'None'
                        currentTruthTableDirectory = dirName + '/Cluster' + str(i) + '/truth_tables' + "_" + currentPseudoTime
                        if os.name == 'nt':
                            outputString = "./progW " + currentGeneNamesDir + " " + currentExecution + " " + problemName + " " + currentGene + " " + str(currentSeed) + " " + currentTruthTableDirectory + " 1 " + "\n"
                        else:
                            outputString = "./progL " + currentGeneNamesDir + " " + currentExecution + " " + problemName + " " + currentGene + " " + str(currentSeed) + " " + currentTruthTableDirectory + " 1 " + "\n"                    
                        
                        openOutFile.write(outputString)                        
                    openOutFile.close()
        else:
            for i in range(independentRuns):
                currentExecution = 'exe_' + str(i+1) + "_" + currentPseudoTime
                Utils.mkdir('executions_parallel/' + currentExecution)
                Utils.mkdir('time_counting/' + currentExecution)
                '''
                if os.path.exists('executions_parallel/' + currentExecution) == False:
                    os.mkdir('executions_parallel/' + currentExecution)
                if os.path.exists('time_counting/' + currentExecution) == False:
                    os.mkdir('time_counting/' + currentExecution)
                '''
                all_dirExecutions.append(currentExecution)
                currentOutputFileName = problemName + '_exe' + str(i+1) + "_" + currentPseudoTime + ".sh"
                bashScripts.append(currentOutputFileName)
                openOutFile = open(currentOutputFileName, 'w')
                if clusterMethod != 'None':
                    if fullTT == False:
                        for j in range(len(genesNamesFiles)):
                            currentGeneFileName = genesNamesFiles[j]
                            currentSeed = (i * len(genesNamesFiles)) + j
                            
                            currentGeneDirectory = currentGeneFileName
                            currentGene = currentGeneFileName.split('_')[1].replace('.txt', '')
                            currentTruthTableDirectory = dirName + "/truth_tables" + "_" + currentPseudoTime
                            all_geneNamesDir.append(currentGeneDirectory)
                            if os.name == 'nt':
                                outputString = "./progW " + currentGeneDirectory + " " + currentExecution + " " + problemName + " " + currentGene + " " + str(currentSeed) + " " + currentTruthTableDirectory + " 0 " + "\n"
                            else:
                                outputString = "./progL " + currentGeneDirectory + " " + currentExecution + " " + problemName + " " + currentGene + " " + str(currentSeed) + " " + currentTruthTableDirectory + " 0 " + "\n"
                            openOutFile.write(outputString)
                    else:
                        for j in range(len(genesNamesFiles)):
                            currentGeneFileName = genesNamesFiles[j]
                            currentSeed = (i * len(genesNamesFiles)) + j
                            currentGeneDirectory = currentGeneFileName
                            currentGene = 'None'
                            all_geneNamesDir.append(currentGeneDirectory)
                            #all_geneNamesDirFULLTT.append(currentGeneNamesDir)
                            currentTruthTableDirectory = dirName + "/truth_tables" + "_" + currentPseudoTime
                            if os.name == 'nt':
                                outputString = "./progW " + currentGeneDirectory + " " + currentExecution + " " + problemName + " " + currentGene + " " + str(currentSeed) + " " + currentTruthTableDirectory + " 1 " + "\n"
                            else:
                                outputString = "./progL " + currentGeneDirectory + " " + currentExecution + " " + problemName + " " + currentGene + " " + str(currentSeed) + " " + currentTruthTableDirectory + " 1 " + "\n"
                            openOutFile.write(outputString)
                else:
                    if fullTT == False:
                        for j in range(len(genesNamesFiles)):
                            currentGeneFileName = genesNamesFiles[j]
                            currentSeed = (i * len(genesNamesFiles)) + j
                            
                            currentGeneDirectory = currentGeneFileName
                            #currentGene = currentGeneFileName.split('_')[1].replace('.txt', '')
                            currentLocalGenesOpen = open(currentGeneFileName, 'r')
                            for line in currentLocalGenesOpen:
                                currentGene = line.strip()
                                currentTruthTableDirectory = dirName + "/truth_tables" + "_" + currentPseudoTime
                                all_geneNamesDir.append(currentGeneDirectory)
                                if os.name == 'nt':
                                    outputString = "./progW " + currentGeneDirectory + " " + currentExecution + " " + problemName + " " + currentGene + " " + str(currentSeed) + " " + currentTruthTableDirectory + " 0 " + "\n"
                                else:
                                    outputString = "./progL " + currentGeneDirectory + " " + currentExecution + " " + problemName + " " + currentGene + " " + str(currentSeed) + " " + currentTruthTableDirectory + " 0 " + "\n"
                                openOutFile.write(outputString)
                    else:
                        for j in range(len(genesNamesFiles)):
                            currentGeneFileName = genesNamesFiles[j]
                            currentSeed = (i * len(genesNamesFiles)) + j
                            currentGeneDirectory = currentGeneFileName
                            currentGene = 'None'
                            all_geneNamesDir.append(currentGeneDirectory)
                            all_geneNamesDirFULLTT.append(currentGeneDirectory)
                            currentTruthTableDirectory = dirName + "/truth_tables" + "_" + currentPseudoTime
                            if os.name == 'nt':
                                outputString = "./progW " + currentGeneDirectory + " " + currentExecution + " " + problemName + " " + currentGene + " " + str(currentSeed) + " " + currentTruthTableDirectory + " 1 " + "\n"
                            else:
                                outputString = "./progL " + currentGeneDirectory + " " + currentExecution + " " + problemName + " " + currentGene + " " + str(currentSeed) + " " + currentTruthTableDirectory + " 1 " + "\n"
                            openOutFile.write(outputString)                    
                        
                openOutFile.close()


        #all_geneNamesDir = []
        allMaxInputs = Utils.calculateMaxInputs(all_geneNamesDir)
                    
        allCounts = []      
        if len(all_geneNamesDirFULLTT) != 0:            
            for localDir in all_geneNamesDirFULLTT:
                currentCount = 0
                localOpen = open(localDir, "r")
                for line in localOpen:
                    currentCount += 1
                allCounts.append(currentCount)
                localOpen.close()

        allShellScriptsFileName = problemName + "_full.sh"


        Utils.generateFullBashScript(allShellScriptsFileName, fullTT, bashScripts, allCounts, allMaxInputs, opts) #Complete generation of the fullbashscript file
        '''
        outputFullShellScripts = open(allShellScriptsFileName, 'a')
        if fullTT == False:
            if os.name == 'nt':
                currentMaxOutputs = allMaxInputs
                
                #if currentMaxOutputs <= 200:
                #    currentMaxNodes = 500
                #else:
                #    currentMaxNodes = 500 + math.ceil(currentMaxOutputs // 200) * 500
                #currentMaxEval = 50000 + math.ceil(currentMaxOutputs // 20) * 50000
                
                currentMaxNodes, currentMaxEval = Utils.getNodesAndGenerations(fullTT, opts.cgpNodes, opts.cgpGens, currentMaxOutputs)
                currentString = "python include/makeFile.py -o " + "1" + " -n " + str(currentMaxNodes) + " -e " + str(currentMaxEval) + "\n"
                outputFullShellScripts.write(currentString)                
                for script in bashScripts:
                    currentString = 'bash ' + script + ";\n"
                    outputFullShellScripts.write(currentString)
            else:
                currentMaxOutputs = allMaxInputs
                
                #if currentMaxOutputs <= 200:
                #    currentMaxNodes = 500
                #else:
                #    currentMaxNodes = 500 + math.ceil(currentMaxOutputs // 200) * 500
                #currentMaxEval = 50000 + math.ceil(currentMaxOutputs // 20) * 50000
                
                currentMaxNodes, currentMaxEval = Utils.getNodesAndGenerations(fullTT, opts.cgpNodes, opts.cgpGens, currentMaxOutputs)
                currentString = "python3 include/makeFile.py -o " + "1" + " -n " + str(currentMaxNodes) + " -e " + str(currentMaxEval) + "\n"
                outputFullShellScripts.write(currentString)                  
                for script in bashScripts:                  
                    currentString = 'bash ' + script + " ;\n"
                    outputFullShellScripts.write(currentString)
            outputFullShellScripts.close()
        else:
            if os.name == 'nt':
                for scriptNumber in range(len(bashScripts)):
                    script = bashScripts[scriptNumber]
                    currentMaxOutputs = allCounts[scriptNumber]
                    
                    #if currentMaxOutputs <= 200:
                    #    currentMaxNodes = 500
                    #else:
                    #    currentMaxNodes = 500 + math.ceil(currentMaxOutputs // 200) * 500
                    #currentMaxEval = 1000000 + math.ceil(currentMaxOutputs // 400) * 1000000
                    
                    currentMaxNodes, currentMaxEval = Utils.getNodesAndGenerations(fullTT, opts.cgpNodes, opts.cgpGens, currentMaxOutputs)
                    currentString = "python include/makeFile.py -o " + str(currentMaxOutputs) + " -n " + str(currentMaxNodes) + " -e " + str(currentMaxEval) + "\n"
                    outputFullShellScripts.write(currentString)
                    currentString = 'bash ' + script + ";\n"
                    outputFullShellScripts.write(currentString)
            else:
                for scriptNumber in range(len(bashScripts)):
                    script = bashScripts[scriptNumber]
                    currentMaxOutputs = allCounts[scriptNumber]
                    
                    #if currentMaxOutputs <= 200:
                    #    currentMaxNodes = 500
                    #else:
                    #    currentMaxNodes = 500 + math.ceil(currentMaxOutputs // 200) * 500
                    #currentMaxEval = 1000000 + math.ceil(currentMaxOutputs // 400) * 1000000
                    
                    currentMaxNodes, currentMaxEval = Utils.getNodesAndGenerations(fullTT, opts.cgpNodes, opts.cgpGens, currentMaxOutputs)
                    currentString = "python3 include/makeFile.py -o " + str(currentMaxOutputs) + " -n " + str(currentMaxNodes) + " -e " + str(currentMaxEval) + "\n"
                    outputFullShellScripts.write(currentString)
                    currentString = 'bash ' + script + " ;\n"
                    outputFullShellScripts.write(currentString)
            outputFullShellScripts.close()            
        '''
    
    outputFullShellScripts = open(allShellScriptsFileName, 'a')
    if os.name == 'nt':
        outputFullShellScripts.write("python include/postProcess.py")
    else:
        outputFullShellScripts.write("python3 include/postProcess.py")
    outputFullShellScripts.close()


    Log.register('message', "Bash scripts sucessfully generated.")
    tf_Bash = perf_counter()
    all_BashTimes.append(tf_Bash - t0_Bash)
    Log.register('time', str(tf_Bash - t0_Bash))

    tf_General = perf_counter()

    Log.register('final')
    Log.register('timef', str(tf_General - t0_General))

    parametersFileName = "CGPGRN_parameters.txt"
    parametersFile = open(parametersFileName, "w")
    parametersFile.write(str(all_discretizedFiles) + "\n")
    parametersFile.write(str(all_genesNamesFiles) + "\n")
    parametersFile.write(str(all_directories) + "\n")
    parametersFile.write(str(all_dirExecutions) + "\n")
    parametersFile.write("Additional Discretization Files: " + str(additional_discretizedFiles) + "\n")
    parametersFile.write(str(opts) + "\n")
    parametersFile.close()

    timeFileName = "CGPGRN_times.txt"
    timeFile = open(timeFileName, "w")
    timeFile.write("Spline Files: " + str(all_SPFiles) + " - Total: " + str(sum(all_SPFiles)) + "\n")
    timeFile.write("Clustering Step: " + str(all_CDTimes) + " - Total: " + str(sum(all_CDTimes)) + "\n")
    timeFile.write("Discretization: " + str(all_DDTimes) + " - Total: " + str(sum(all_DDTimes)) + "\n")
    timeFile.write("Truth Table Generation: " + str(all_GOTimes) + " - Total: " + str(sum(all_GOTimes)) + "\n")
    timeFile.write("Bash Generation: " + str(all_BashTimes) + " - Total: " + str(sum(all_BashTimes)) + "\n")
    timeFile.write("Total Elapsed Time: " + str(tf_General - t0_General) + "\n")
    timeFile.close()



    print(opts)


    if runCGP:

        if os.name != 'nt':
            os.system('chmod +x *.sh')
            os.system('chmod +x progL')
            shellFile = "./" + allShellScriptsFileName
            subprocess.call(shellFile, shell=True)
        else:
            os.popen(allShellScriptsFileName)



