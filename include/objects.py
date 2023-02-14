import os
import pandas as pd
import utils as Utils
import preProcessing as PP
import clusterData as CD
import discretizeData as DD


class CGPGRN:
    def __init__(self, arguments, Log):
        '''
        Object constructor
        '''
        self.arguments = arguments
        self.Log = Log
        
        self.genesNamesFiles = []
        self.directories = []
        self.additional_discretizedFiles = []
        self.discretizedFiles = []        

    def performSplineStep(self):
        '''
        Performs the spline step. Considers and verifies the given spline files if passed as argument.
        Outputs:
        splineDataNames - all spline data filenames
        '''

        std_messages = {'maxclusters_error': "The maximum number of clusters must be lower than the total number of genes."}

        print("Starting spline step...")
        self.Log.register('message', 'Starting spline step...')
        
        #Verification of spline files, spline list and generation of spline data
        if self.arguments.argSplineFile == 'None':
            if Utils.verifyNumberOfClusters([self.arguments.expressionDataFile], self.arguments.n_clusters) == True:
                print("Generating spline file...")
                self.Log.register('message', "Generating spline file...")
                splineDataNames = PP.preProcessData(self.arguments.pseudotimeFile, self.arguments.expressionDataFile, self.arguments.suffix)
            else:
                print(std_messages['maxclusters_error'])
                self.Log.register('error', std_messages['maxclusters_error'])
                self.Log.register('info', "You set the number of clusters as " + str(self.arguments.n_clusters))
        else:
            if self.arguments.splineList == False:
                verifyClusterConstraint = Utils.verifyNumberOfClusters([self.arguments.argSplineFile], self.arguments.n_clusters)
                if verifyClusterConstraint == True:
                    print("Using spline file passed as an argument...")
                    self.Log.register('message', "Using spline file passed as an argument")
                    splineDataNames = [self.arguments.argSplineFile]
                elif verifyClusterConstraint == 'nf':
                    print("Spline file not found.")
                    self.Log.register('error', "Spline file not found.")
                    exit()
                else:
                    print(std_messages['maxclusters_error'])
                    self.Log.register('error', std_messages['maxclusters_error'])
                    exit()
            else:
                print("Using spline list file passed as an argument...")
                self.Log.register('message', "Using spline list file passed as an argument...")                
                splineDataNames = Utils.getSplineFilesNames(self.arguments.argSplineFile)
                verifyClusterConstraint = Utils.verifyNumberOfClusters(splineDataNames, self.arguments.n_clusters)
                if verifyClusterConstraint == False:
                    print(std_messages['maxclusters_error'])
                    self.Log.register('error', std_messages['maxclusters_error'])
                    exit()
                elif verifyClusterConstraint == 'nf':
                    print("Spline file not found.")
                    self.Log.register('error', "Spline file not found.")
                    exit()

        #Generation of spline list, if -gsl is True
        if self.arguments.generateSplineList == True:
            Utils.generateSplineList(splineDataNames, self.Log, self.arguments.problemName)

        
        return splineDataNames


    def performClusteringStep(self, splineDataName, currentPseudoTime):
        dirName, bestNClusters = CD.main(self.arguments.problemName, self.arguments.correlation_threshold, splineDataName, self.arguments.n_clusters, self.arguments.clusterMethod, currentPseudoTime, self.Log)
        return dirName, bestNClusters


    def performDiscretizationStep(self, dirName, bestNClusters, currentPseudoTime, splineDataName):
        print("DISCRETIZATION")

        genesNamesFiles = []
        directories = []
        additional_discretizedFiles = []
        discretizedFiles = []

        if self.arguments.fullDiscretization == True:
            self.Log.register('message', "Using full discretization...")

            if self.arguments.clusterMethod in Utils.clusteringMethods: #Using a clustering method
                localClusterDir = Utils.dict_directories[self.arguments.clusterMethod]

                for cluster in range(bestNClusters):
                    print("Discretizing data for cluster " + str(cluster))
                    self.Log.register('message', "Discretizing data for cluster " + str(cluster))

                    currentDir1 = os.path.join(os.getcwd(), localClusterDir)
                    currentDir = currentDir1 + "/Cluster" + str(cluster)

                    currentFile = currentDir + "/spline_" + self.arguments.problemName + "_" + str(cluster) + "_" + currentPseudoTime + ".csv"
                    currentGenesNamesFiles = currentDir + "/geneNames_" + str(cluster) + "_" + currentPseudoTime + ".txt"

                    genesNamesFiles.append(currentGenesNamesFiles)
                    directories.append(currentDir)

                    currentOutFile = currentDir + Utils.dict_discretizationPrefixes[self.arguments.discretizationApproach] + self.arguments.problemName + "_" + str(cluster) + "_" + currentPseudoTime + ".csv"

                    if self.arguments.discretizationApproach == 'BiKMeans':
                        if Utils.verifyNumberOfClusters([localClusterDir+"/Cluster"+str(cluster) + "/spline_" + self.arguments.problemName + "_" + str(cluster) + "_" + currentPseudoTime + ".csv"], 3) != True:
                            print("The spline data does not contain the minimum number of needed genes (3). Using not full discretization instead.")
                            self.Log.register('warning', "The spline data does not contain the minimum number of needed genes (3). Using not full discretization instead.")
                            self.Log.register('info', "Spline File Name: " + str(currentFile))

                            currentTempFile = splineDataName
                            currentTempOutFile = Utils.dict_discretizationPrefixes[self.arguments.discretizationApproach] + self.arguments.problemName + "_full_" + currentPseudoTime + ".csv"
                            if not os.path.exists(currentTempOutFile):
                                DD.discretizationProcedure(self.arguments.discretizationApproach, currentTempOutFile, currentTempFile, self.arguments.problemName, currentPseudoTime)
                                additional_discretizedFiles.append(currentTempOutFile)

                            currentOutFile2 = currentDir + Utils.dict_discretizationPrefixes[self.arguments.discretizationApproach] + self.arguments.problemName + "_" + str(cluster) + "_" + currentPseudoTime + ".csv"

                            DD.generateNotFullDiscretizationData(currentTempOutFile, currentOutFile2, currentGenesNamesFiles)

                            discretizedFiles.append(currentOutFile2)

                        else:

                            DD.discretizationProcedure(self.arguments.discretizationApproach, currentOutFile, currentFile, self.arguments.problemName, currentPseudoTime)
                            discretizedFiles.append(currentOutFile)
                    else:
                        DD.discretizationProcedure(self.arguments.discretizationApproach, currentOutFile, currentFile, self.arguments.problemName, currentPseudoTime)
                        discretizedFiles.append(currentOutFile)

            elif self.arguments.clusterMethod in Utils.correlationMethods:
                print("eh metodo de correlação")

                currentDir = dirName + "/"

                readSplineData = pd.read_csv(splineDataName, index_col=0)
                allGenes = readSplineData.T.columns

                for gene in allGenes:
                    self.Log.register('message', "Discretizing data for gene " + str(gene))
                    currentFile = currentDir + "spline_" + self.arguments.problemName + "_" + str(gene) + "_" + currentPseudoTime + ".csv"
                    currentGenesNamesFiles = currentDir + "geneNames_" + str(gene) + "_" + currentPseudoTime + ".txt"
                    genesNamesFiles.append(currentGenesNamesFiles)
                    directories.append(currentDir)
                    currentOutFile = currentDir + Utils.dict_discretizationPrefixes[self.arguments.discretizationApproach] + self.arguments.problemName + "_" + str(gene) + "_" + currentPseudoTime + ".csv"
                    DD.discretizationProcedure(self.arguments.discretizationApproach, currentOutFile, currentFile, self.arguments.problemName, currentPseudoTime)

                    discretizedFiles.append(currentOutFile)

            else:

                print("nao eh nem clustering nem correlation")

                currentDir = dirName + "/"

                readSplineData = pd.read_csv(splineDataName, index_col=0)
                allGenes = readSplineData.T.columns

                currentFile = currentDir + "spline_" + self.arguments.problemName + "_" + currentPseudoTime + ".csv"
                currentGenesNamesFiles = currentDir + "genesNames_" + self.arguments.problemName + "_" + currentPseudoTime + ".txt"
                genesNamesFiles.append(currentGenesNamesFiles)
                directories.append(currentDir)
                currentOutFile = currentDir + Utils.dict_discretizationPrefixes[self.arguments.discretizationApproach] + self.arguments.problemName + "_" + currentPseudoTime + ".csv"
                DD.discretizationProcedure(self.arguments.discretizationApproach, currentOutFile, currentFile, self.arguments.problemName, currentPseudoTime)

                discretizedFiles.append(currentOutFile)

        else:
            print("não eh full TT")

            self.Log.register('message', "Using not full discretization...")
            if self.arguments.clusterMethod in Utils.clusteringMethods:
                localClusterDir = Utils.dict_directories[self.arguments.clusterMethod]

                currentFile = splineDataName
                currentOutFile = Utils.dict_discretizationPrefixes[self.arguments.discretizationApproach] + self.arguments.problemName + "_full_" + currentPseudoTime + ".csv"
                DD.discretizationProcedure(self.arguments.discretizationApproach, currentOutFile, currentFile, self.arguments.problemName, currentPseudoTime)

                for cluster in range(bestNClusters):
                    print("Discretizing data for cluster " + str(cluster))
                    self.Log.register('message', "Discretizing data for cluster " + str(cluster))

                    currentDir1 = os.path.join(os.getcwd(), localClusterDir)
                    currentDir = currentDir1 + "/Cluster" + str(cluster)
                    currentGenesNamesFiles = currentDir + "/geneNames_" + str(cluster) + "_" + currentPseudoTime + ".txt"
                    genesNamesFiles.append(currentGenesNamesFiles)
                    directories.append(currentDir)
                    currentOutFile2 = currentDir + Utils.dict_discretizationPrefixes[self.arguments.discretizationApproach] + self.arguments.problemName + "_" + str(cluster) + "_" + currentPseudoTime + ".csv"

                    DD.generateNotFullDiscretizationData(currentOutFile, currentOutFile2, currentGenesNamesFiles)

                    discretizedFiles.append(currentOutFile2)

            elif self.arguments.clusterMethod in Utils.correlationMethods:

                currentFile = splineDataName
                currentOutFile = Utils.dict_discretizationPrefixes[self.arguments.discretizationApproach] + self.arguments.problemName + "_full_" + currentPseudoTime + ".csv"
                DD.discretizationProcedure(self.arguments.discretizationApproach, currentOutFile, currentFile, self.arguments.problemName, currentPseudoTime)

                currentDir = dirName + "/"

                readSplineData = pd.read_csv(splineDataName, index_col=0)
                allGenes = readSplineData.T.columns
                for gene in allGenes:
                    self.Log.register('message', "Discretizing data for gene " + str(gene))
                    currentFile = currentDir + "spline_" + self.arguments.problemName + "_" + str(gene) + "_" + currentPseudoTime + ".csv"
                    currentGenesNamesFiles = currentDir + "/geneNames_" + str(gene) + "_" + currentPseudoTime + ".txt"
                    genesNamesFiles.append(currentGenesNamesFiles)
                    directories.append(currentDir)
                    currentOutFile2 = currentDir + Utils.dict_discretizationPrefixes[self.arguments.discretizationApproach] + self.arguments.problemName + "_" + str(gene) + "_" + currentPseudoTime + ".csv"

                    DD.generateNotFullDiscretizationData(currentOutFile, currentOutFile2, currentGenesNamesFiles)

                    discretizedFiles.append(currentOutFile2)

            else:

                currentDir = dirName + "/"

                readSplineData = pd.read_csv(splineDataName, index_col=0)
                allGenes = readSplineData.T.columns

                currentFile = currentDir + "spline_" + self.arguments.problemName + "_" + currentPseudoTime + ".csv"
                currentGenesNamesFiles = currentDir + "geneNames_" + self.arguments.problemName + "_" + currentPseudoTime + ".txt"
                genesNamesFiles.append(currentGenesNamesFiles)
                directories.append(currentDir)
                currentOutFile = currentDir + Utils.dict_discretizationPrefixes[self.arguments.discretizationApproach] + self.arguments.problemName + "_" + currentPseudoTime + ".csv"
                DD.discretizationProcedure(self.arguments.discretizationApproach, currentOutFile, currentFile, self.arguments.problemName, currentPseudoTime)

                discretizedFiles.append(currentOutFile)

        return genesNamesFiles, directories, additional_discretizedFiles, discretizedFiles
            
                                

                
                
        
        
