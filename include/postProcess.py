import time
import os
import shutil
import numpy as np
import shutil
import glob
import utils as Utils

executionsPerPseudotime = {}
allExecutions = []
problemName = ''

allArgs = {}
allDirs = {}


def getDiscretizationPrefix(discretizationApproach):
    if discretizationApproach == 'BiKMeans':
        return 'BKM'
    elif discretizationApproach == 'Mean':
        return 'Mean'
    elif discretizationApproach == 'Median':
        return 'Median'
    elif discretizationApproach == 'EFD':
        return 'EFD'
    elif discretizationApproach == 'TSD':
        return 'TSD'
    else:
        return 'None'

def getArguments():
    local_allExecutions = []
    local_allArgs = {}
    local_allDirs = {}
    local_allPseudotimes = []
    local_generalSplineFiles = []
    local_discretizationFiles = []
    local_shellScripts = []
    
    logFile = open('CGPGRN_parameters.txt', "r")

    for line in logFile:
        if line.find('exe') != -1:
            splitting = line.split(",")
            for execution in splitting:
                execution = execution.replace("'", "")
                execution = execution.replace('[', '')
                execution = execution.replace(']', '')
                execution = execution.strip()
                local_allExecutions.append(execution)
                pt = execution.split("_")
                pt = pt[len(pt)-1].strip()
                if pt not in local_allPseudotimes:
                    local_allPseudotimes.append(pt)

                
            local_allArgs['exes'] = local_allExecutions
            local_allArgs['pts'] = local_allPseudotimes
            
            
        if line.find('problemName') != -1:
            splitting = line.split(",")
            for arg in splitting:
                if arg.find('problemName') != -1:
                    local_problemName = arg.split("'")[1]
                    local_problemName = local_problemName.replace("'", "")
                    local_problemName = local_problemName.strip()
                    local_allArgs['problemName'] = local_problemName
        if line.find('clusterMethod') != -1:
            splitting = line.split(",")
            for arg in splitting:
                if arg.find('clusterMethod') != -1:
                    local_clusterMethod = arg.split("'")[1]
                    local_clusterMethod = local_clusterMethod.replace("'", "")
                    local_clusterMethod = local_clusterMethod.strip()
                    local_allArgs['clusterMethod'] = local_clusterMethod
                    if local_clusterMethod != 'None':
                        local_allDirs['clusterMethod'] = Utils.dict_directories[local_clusterMethod]
                    else:
                        local_allDirs['clusterMethod'] = 'noClustering'
        if line.find('n_clusters') != -1:
            splitting = line.split(",")
            for arg in splitting:
                if arg.find('n_clusters') != -1:
                    local_nClusters = arg.split("'")[1]
                    local_nClusters = local_nClusters.replace("'", "")
                    local_nClusters = local_nClusters.strip()
                    local_allArgs['nClusters'] = local_nClusters
        if line.find('correlation_threshold') != -1:
            splitting = line.split(",")
            for arg in splitting:
                if arg.find('correlation_threshold') != -1:
                    local_correlationThreshold = arg.split("'")[1]
                    local_correlationThreshold = local_correlationThreshold.replace("'", "")
                    local_correlationThreshold = local_correlationThreshold.strip()
                    local_allArgs['correlationThreshold'] = local_correlationThreshold
        if line.find('expressionDataFile') != -1:
            splitting = line.split(",")
            for arg in splitting:
                if arg.find('expressionDataFile') != -1:
                    local_expressionDataFile = arg.split("'")[1]
                    local_expressionDataFile = local_expressionDataFile.replace("'", "")
                    local_expressionDataFile = local_expressionDataFile.strip()
                    local_allArgs['expressionDataFile'] = local_expressionDataFile
        if line.find('pseudotimeFile') != -1:
            splitting = line.split(",")
            for arg in splitting:
                if arg.find('pseudotimeFile') != -1:
                    local_pseudotimeFile = arg.split("'")[1]
                    local_pseudotimeFile = local_pseudotimeFile.replace("'", "")
                    local_pseudotimeFile = local_pseudotimeFile.strip()
                    local_allArgs['pseudotimeFile'] = local_pseudotimeFile
        if line.find('suffix') != -1:
            splitting = line.split(",")
            for arg in splitting:
                if arg.find('suffix') != -1:
                    local_suffix = arg.split("'")[1]
                    local_suffix = local_suffix.replace("'", "")
                    local_suffix = local_suffix.strip()
                    local_allArgs['suffix'] = local_suffix
        if line.find('discretizationApproach') != -1:
            splitting = line.split(",")
            for arg in splitting:
                if arg.find('discretizationApproach') != -1:
                    local_discretizationApproach = arg.split("'")[1]
                    local_discretizationApproach = local_discretizationApproach.replace("'", "")
                    local_discretizationApproach = local_discretizationApproach.strip()
                    local_allArgs['discretizationApproach'] = local_discretizationApproach
        if line.find('independentRuns') != -1:
            splitting = line.split(",")
            for arg in splitting:
                if arg.find('independentRuns') != -1:
                    local_independentRuns = arg.split("=")[1]
                    local_independentRuns = local_independentRuns.replace("'", "")
                    local_independentRuns = local_independentRuns.strip()
                    local_allArgs['independentRuns'] = local_independentRuns
        if line.find('fullDiscretization') != -1:
            splitting = line.split(",")
            for arg in splitting:
                if arg.find('fullDiscretization') != -1:
                    local_fullDiscretization = arg.split("=")[1]
                    local_fullDiscretization = local_fullDiscretization.replace("'", "")
                    local_fullDiscretization = local_fullDiscretization.strip()
                    local_allArgs['fullDiscretization'] = local_fullDiscretization
        if line.find('argSplineFile') != -1:
            splitting = line.split(",")
            for arg in splitting:
                if arg.find('argSplineFile') != -1:
                    local_argSplineFile = arg.split("=")[1]
                    local_argSplineFile = local_argSplineFile.replace("'", "")
                    local_argSplineFile = local_argSplineFile.strip()
                    local_allArgs['argSplineFile'] = local_argSplineFile
        if line.find('splineList') != -1:
            splitting = line.split(",")
            for arg in splitting:
                if arg.find('splineList') != -1:
                    local_splineList = arg.split("=")[1]
                    local_splineList = local_splineList.replace(")", "")
                    local_splineList = local_splineList.strip()
                    local_allArgs['splineList'] = local_splineList
        if line.find('keepData') != -1:
            splitting = line.split(",")
            for arg in splitting:
                if arg.find('keepData') != -1:
                    local_keepData = arg.split("=")[1]
                    local_keepData = local_keepData.replace(")", "")
                    local_keepData = local_keepData.strip()
                    local_allArgs['keepData'] = local_keepData
        if line.find('Additional Discretization Files: ') != -1:
            splitting = line.split(":")
            splitting2 = splitting[1].split(",")
            local_additionalDF = []
            for additional in splitting2:
                currentAdditional = additional.replace("[", "")
                currentAdditional = currentAdditional.replace("]", "")
                currentAdditional = currentAdditional.strip()
                local_additionalDF.append(currentAdditional)
            local_allArgs['additionalDF'] = local_additionalDF

    if local_allArgs['argSplineFile'] == 'None':
        for n_pt in local_allArgs['pts']:
            local_generalSplineFiles.append('splineData_' + local_allArgs['suffix'] + "_" + n_pt + ".csv")
    else:
        if local_allArgs['splineList'] == 'False':
            local_generalSplineFiles.append(local_allArgs['argSplineFile'])
        else:
            localFile = open(local_allArgs['argSplineFile'])
            for line in localFile:
                local_generalSplineFiles.append(line.strip())

    local_allDirs['gSplineFiles'] = local_generalSplineFiles

    #discretizationPrefix = getDiscretizationPrefix(local_allArgs['discretizationApproach'])

    discretizationPrefix = Utils.dict_discretizationPrefixes[local_allArgs['discretizationApproach']].replace("_", "")
    
    if local_allArgs['fullDiscretization'] == 'False':
        for n_pt in local_allArgs['pts']:
            local_discretizationFiles.append(discretizationPrefix + "_" + local_allArgs['problemName'] + "_full_" + n_pt + ".csv")
    local_allDirs['discretizationFiles'] = local_discretizationFiles            

    for exe in local_allArgs['exes']:
        local_shellScripts.append(local_allArgs['problemName'] + "_" + exe + ".sh")
    local_shellScripts.append(local_allArgs['problemName'] + "_full.sh")

    local_allDirs['shellScripts'] = local_shellScripts

    return local_allArgs, local_allDirs



def moveData(destination):



    
    for key in allDirs.keys():
        if key == 'gSplineFiles':
            localDestination = destination + "/splineData"
        elif key == 'discretizationFiles':
            localDestination = destination + "/discretizedData"
        elif key == 'shellScripts':
            localDestination = destination + "/shellScripts"
        else:
            localDestination = destination

        if not os.path.exists(localDestination):
            os.mkdir(localDestination)
            
        if type(allDirs[key]) == list:
            if key == 'gSplineFiles' and allArgs['keepData'] == 'True':
                for item in allDirs[key]:
                    if os.path.exists(item):
                        shutil.copy(item, localDestination)
            else:
                for item in allDirs[key]:
                    if os.path.exists(item):
                        shutil.move(item, localDestination)
        else:
            if key == 'gSplineFiles' and allArgs['keepData'] == 'True':
                if os.path.exists(allDirs[key]):
                    shutil.copy(allDirs[key], localDestination)
            else:
                if os.path.exists(allDirs[key]):
                    shutil.move(allDirs[key], localDestination)


    if len(allArgs['additionalDF']) != 0 and '' not in allArgs['additionalDF']:
        for additional in allArgs['additionalDF']:
            local = additional.replace("'", "")
            local_dir = os.path.join(os.getcwd(), local)
            shutil.move(local_dir, destination + "/discretizedData")



    if not os.path.exists(destination + "/sourceData"):
        os.mkdir(destination + "/sourceData")


    if allArgs['keepData'] == 'True':
        if allArgs['splineList'] == 'True':
            shutil.copy(allArgs['argSplineFile'], destination + "/splineData")
        shutil.copy(allArgs['expressionDataFile'], destination + "/sourceData")
        shutil.copy(allArgs['pseudotimeFile'], destination + "/sourceData")
        logFileName = 'LogFile_' + allArgs['problemName'] + ".txt"
        logFile = open(logFileName, 'r')
        splinelist = 'None'
        for line in logFile:
            if line.find('Spline List Filename:') != -1:
                position = line.find('Spline List Filename:')
                splinelist = line[position:].split(":")[1].strip()
                break
        logFile.close()
        if splinelist != 'None':
            shutil.copy(splinelist, destination + "/splineData")
                
    else:
        if allArgs['splineList'] == 'True':
            shutil.move(allArgs['argSplineFile'], destination + "/splineData")
            
        shutil.move(allArgs['expressionDataFile'], destination + "/sourceData")
        shutil.move(allArgs['pseudotimeFile'], destination + "/sourceData")

        logFileName = 'LogFile_' + allArgs['problemName'] + ".txt"
        logFile = open(logFileName, 'r')
        splinelist = 'None'
        for line in logFile:
            if line.find('Spline List Filename:') != -1:
                position = line.find('Spline List Filename:')
                splinelist = line[position:].split(":")[1].strip()
                break
        logFile.close()
        if splinelist != 'None':
            shutil.move(splinelist, destination + "/splineData")
    
    shutil.move('executions_parallel', destination)
    shutil.move('time_counting', destination)



    shutil.move('CGPGRN_parameters.txt', destination)
    shutil.move('CGPGRN_times.txt', destination)
    shutil.move('LogFile_' + allArgs['problemName'] + ".txt", destination)
    shutil.move("timeStatistics_" + allArgs['problemName'] + ".txt", destination)




    diverseSHFiles = glob.glob('*.sh')
    for shFile in diverseSHFiles:
        shutil.move(shFile, destination + "/shellScripts")


allArgs, allDirs = getArguments()



def generateTimeStatistics():
    timeIter = 0
    timeIterTotal = 0
    timeKernel = 0
    totalTime = 0
    preProcessingTime = 0
    
    timeDir = "time_counting"
    executions = allArgs['exes']

    allFiles = []
    
    for execution in executions:
        currentDir = os.path.join(timeDir, execution)
        for file in os.listdir(currentDir):
            if file.endswith(".txt"):
                allFiles.append(os.path.join(currentDir, file))
                currentOpen = open(os.path.join(currentDir, file), "r")
                for line in currentOpen:
                    if line.find("timeIter") != -1:
                        split = line.split(":")[1].strip()
                        timeIter += float(split)
                    if line.find("timeIterTotal") != -1:
                        split = line.split(":")[1].strip()
                        timeIterTotal += float(split)
                    if line.find("timeKernel") != -1:
                        split = line.split(":")[1].strip()
                        timeKernel += float(split)
                    if line.find("Total time") != -1:
                        split = line.split(":")[1].strip()
                        totalTime += float(split)
                currentOpen.close()

    logfileName = "LogFile_" + allArgs['problemName'] + ".txt"
    logfileOpen = open(logfileName, "r")
    for line in logfileOpen:
        if line.find("Total elapsed time") != -1:
            split = line[line.find("Total elapsed time"):].split(":")[1].replace("s", "")
            preProcessingTime += float(split)
    logfileOpen.close()

    fullCGPGRNTime = preProcessingTime + totalTime

    outputFileName = "timeStatistics_" + allArgs['problemName'] + ".txt"
    outputFileOpen = open(outputFileName, "w")

    outputFileOpen.write("Number of files: " + str(len(allFiles)) + "\n")
    
    outputFileOpen.write("timeName\tMean (s)\tTotal (s)\n")
    outputFileOpen.write("timeIter\t" + str(round(timeIter/len(allFiles), 2)) + "\t" + str(round(timeIter, 2)) + "\n")
    outputFileOpen.write("timeIterTotal\t" + str(round(timeIterTotal/len(allFiles), 2)) + "\t" + str(round(timeIterTotal, 2)) + "\n")
    outputFileOpen.write("timeKernel\t" + str(round(timeKernel/len(allFiles), 2)) + "\t" + str(round(timeKernel, 2)) + "\n")
    outputFileOpen.write("totalTime\t" + str(round(totalTime/len(allFiles), 2)) + "\t" + str(round(totalTime, 2)) + "\n")

    outputFileOpen.write("\nPre-processing time: " + str(preProcessingTime) + "s\n")
    outputFileOpen.write("\nTotal elapsed time for CGPGRN framework: " + str(round(fullCGPGRNTime, 2)) + "s")

    outputFileOpen.close()





def generateRankedEdgesPerExecution(finalDirectory):

    nRuns = allArgs['independentRuns']
    nPTs = allArgs['pts']
    CM = allArgs['clusterMethod']
    EXES = allArgs['exes']
    allExes = []
    for i in range(int(nRuns)):
        allExes.append("exe_" + str(i+1))



    if CM in Utils.clusteringMethods:
        foundClusters = []
        for i in range(int(allArgs['nClusters'])):
            currentSearch = 'Cluster' + str(i)
            for value in allArgs['exes']:
                if currentSearch in value:
                    if currentSearch not in foundClusters:
                        foundClusters.append(currentSearch)

        for exe in allExes:
            currentDirsExecution = []
            for pt in nPTs:
                for cluster in foundClusters:
                    currentDirsExecution.append(exe + "_" + cluster + "_" + pt)
            URE_fileName = 'unified_rankedEdges_' + allArgs['problemName'] + "_" + exe + ".csv"
            #URE_file = open(URE_fileName, "w")
            URE_file = open(finalDirectory + "/" + URE_fileName, "w")
            for currentDir in currentDirsExecution:
                currentFileName = "executions_parallel/" + currentDir + "/" + "rankedEdges_" + allArgs['problemName'] + ".csv"
                if os.path.exists(currentFileName):
                    localOpen = open(currentFileName, "r")
                    for line in localOpen:
                        URE_file.write(line)
                    localOpen.close()
            URE_file.close()
    else:
        for exe in allExes:
            currentDirsExecution = []
            for pt in nPTs:
                currentDirsExecution.append(exe + "_" + pt)
            URE_fileName = 'unified_rankedEdges_' + allArgs['problemName'] + "_" + exe + ".csv"
            #URE_file = open(URE_fileName, "w")
            URE_file = open(finalDirectory + "/" + URE_fileName, "w")
            for currentDir in currentDirsExecution:
                currentFileName = "executions_parallel/" + currentDir + "/" + "rankedEdges_" + allArgs['problemName'] + ".csv"
                if os.path.exists(currentFileName):
                    localOpen = open(currentFileName, "r")
                    for line in localOpen:
                        URE_file.write(line)
                    localOpen.close()
            URE_file.close()            




    for exe in allExes:
        currentURE_fileName = 'unified_rankedEdges_' + allArgs['problemName'] + "_" + exe + ".csv"
        #currentURE_file = open(currentURE_fileName, "r")
        currentURE_file = open(finalDirectory + "/" + currentURE_fileName, "r")
        #if os.path.exists(currentURE_fileName):

        EDGES = {}

        for line in currentURE_file:
            localEdges = []
            splitting = line.split("\t")
            genes = splitting[0].strip() + "\t" + splitting[1].strip()
            regulationStrength = splitting[2].replace("\n", "")
            regulationStrength = regulationStrength.replace("'", "")
            regulationStrength = regulationStrength.strip()
            if genes in EDGES.keys():
                if float(regulationStrength) > EDGES[genes]:
                    EDGES[genes] = float(regulationStrength)
            else:
                EDGES[genes] = float(regulationStrength)

        URE_file.close()



        uniqueValues = list(np.unique(list(EDGES.values())))
        uniqueValues.sort(reverse=True)

        FRE_fileName = 'rankedEdges_' + allArgs['problemName'] + "_" + exe + ".csv"

        FRE_file = open(finalDirectory + "/" +FRE_fileName, "w")
        #FRE_file = open(FRE_fileName, "w")
        FRE_file.write("Gene1\tGene2\tEdgeWeight\n")

        for value in uniqueValues:
            for regulation in EDGES.keys():
                if EDGES[regulation] == value:
                    currentString = str(regulation) + "\t" + str(value) + "\n"
                    FRE_file.write(currentString)

        FRE_file.close()        


        #else:
        #    print("File not found.")
        #    print(currentURE_fileName)
            
          


generateTimeStatistics()





        


finalDirectory = "finalResults_" + allArgs['problemName']
if not(os.path.exists(finalDirectory)):
    os.mkdir(finalDirectory)
else:
    i = 1
    local_finalDirectory = finalDirectory + "_" + str(i)
    while(os.path.exists(local_finalDirectory)):
        local_finalDirectory = local_finalDirectory.replace(str(i), str(i+1))
        i += 1
    finalDirectory = local_finalDirectory
    os.mkdir(finalDirectory)
        


generateRankedEdgesPerExecution(finalDirectory)


URE_fileName = 'unified_rankedEdges_' + allArgs['problemName'] + ".csv"

URE_file = open(finalDirectory + "/" + URE_fileName, "w")
for execution in allArgs['exes']:
    currentFileName = "executions_parallel/" + execution + "/" + "rankedEdges_" + allArgs['problemName'] + ".csv"
    if os.path.exists(currentFileName):
        localOpen = open(currentFileName, "r")
        for line in localOpen:
            URE_file.write(line)
        localOpen.close()
URE_file.close()

URE_file = open(finalDirectory + "/" + URE_fileName, "r")

EDGES = {}

for line in URE_file:
    localEdges = []
    splitting = line.split("\t")
    genes = splitting[0].strip() + "\t" + splitting[1].strip()
    regulationStrength = splitting[2].replace("\n", "")
    regulationStrength = regulationStrength.replace("'", "")
    regulationStrength = regulationStrength.strip()
    if genes in EDGES.keys():
        if float(regulationStrength) > EDGES[genes]:
            EDGES[genes] = float(regulationStrength)
    else:
        EDGES[genes] = float(regulationStrength)

URE_file.close()



uniqueValues = list(np.unique(list(EDGES.values())))
uniqueValues.sort(reverse=True)

FRE_fileName = 'rankedEdges_' + allArgs['problemName'] + ".csv"

FRE_file = open(finalDirectory + "/" +FRE_fileName, "w")
FRE_file.write("Gene1\tGene2\tEdgeWeight\n")

for value in uniqueValues:
    for regulation in EDGES.keys():
        if EDGES[regulation] == value:
            currentString = str(regulation) + "\t" + str(value) + "\n"
            FRE_file.write(currentString)

FRE_file.close()



    
moveData(finalDirectory)




        


