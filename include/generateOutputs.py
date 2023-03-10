import pandas as pd
import psutil
from time import perf_counter
import os

def generateOutputs(PROBLEM_NAME, genesNamesFile, discretizationFile, discretizationType, outputDir, oneOutputPerGene, ignoreTargetGenes, targetGenes=[]):
    problem_name = PROBLEM_NAME
    GLOBAL_NI = [0]

    t1_start = perf_counter()



    TODOS_GENES = []

    arquivo = open(genesNamesFile)
    for linha in arquivo:
        TODOS_GENES.append(linha.strip())
    arquivo.close()



    GLOBAL_NI[0] = len(TODOS_GENES)

    chunksize = 2000
    if chunksize < len(TODOS_GENES):
        chunksize = len(TODOS_GENES)

    fileName = discretizationFile

    dictStates = {}
    i = 0
    for chunk in pd.read_csv(fileName, chunksize=chunksize, index_col=0):
        timePoints = chunk.columns
        for timepoint in timePoints:
            currentData = chunk[timepoint]
            currentString = ""
            for bit in currentData:
                if bit != ' ':
                    if discretizationType == 1:
                        if int(bit) == 2:
                            currentString += "1"
                        else:
                            currentString += "0"
                    else:
                        currentString += str(int(bit))
            
            if timepoint.strip() in dictStates:
                dictStates[timepoint.strip()] += currentString
            else:
                dictStates[timepoint.strip()] = currentString
        i += 1


    transitions = []
    i = 0

    for timePoints in dictStates.keys():
        if i == 0:
            currentState = dictStates[timePoints]
            transitions.append(currentState)
            i += 1
        else:
            if str(dictStates[timePoints]) == str(transitions[len(transitions)-1]):
                pass
            else:
                currentState = dictStates[timePoints]
                transitions.append(currentState)
            i += 1
            
    dictStates = {}
    del dictStates


    print("TRANSITIONS: ")
    print(transitions)



    dictTransitions = {}

    for transition in range(len(transitions)):
        currentState = transitions[transition]
        if transition != len(transitions) - 1:
            if currentState in dictTransitions.keys():
                dictTransitions[currentState].append(transitions[transition + 1])
            else:
                dictTransitions[currentState] = []
                dictTransitions[currentState].append(transitions[transition + 1])


    print("DICT TRANSITIONS")
    print(dictTransitions)
    print("Solving ambiguous transitions...")

    dictFinalTransitions = {}

    for transition in dictTransitions.keys():
        if len(dictTransitions[transition]) != 1:
            solvedTransition = ""
            for bit in range(len(dictTransitions[transition][0])):
                bits = []
                for k in range(len(dictTransitions[transition])):
                    bits.append(int(dictTransitions[transition][k][bit]))
                '''    
                if bits.count(2) == bits.count(1):
                    solvedTransition += "9"
                elif bits.count(2) > bits.count(1):
                    solvedTransition += "2"
                else:
                    solvedTransition += "1"
                '''
                if bits.count(0) == bits.count(1):
                    solvedTransition += "9"
                elif bits.count(1) > bits.count(0):
                    solvedTransition += "1"
                else:
                    solvedTransition += "0"
                #print(bits)
            dictFinalTransitions[transition] = solvedTransition
        else:
            dictFinalTransitions[transition] = dictTransitions[transition][0]


    print("AQUI")
    print(len(dictFinalTransitions.keys()))
    print(dictFinalTransitions)
    
    #dictFinalTransitions[list(dictFinalTransitions.keys())[len(dictFinalTransitions.keys())-1]] = list(dictFinalTransitions.keys())[len(dictFinalTransitions.keys())-1]

    if len(dictFinalTransitions.keys()) == 0: #TODO: SE NAO EXISTEM TRANSICOES, A TT VAI COM 0 LINHAS NESSE CASO ABAIXO
        print("EH VAZIO")
        dictFinalTransitions['9'] = '9'
    else:
        dictFinalTransitions[list(dictFinalTransitions.keys())[len(dictFinalTransitions.keys())-1]] = list(dictFinalTransitions.keys())[len(dictFinalTransitions.keys())-1]

    

    #print(dictFinalTransitions)
    dictTransitions = {}
    del dictTransitions




    NOVAS_TRANSITIONS = []
    for local_entradas in range(len(list(dictFinalTransitions.keys())[0])):
        string = ""
        for value in dictFinalTransitions.values():
            string += value[local_entradas]
        NOVAS_TRANSITIONS.append(string)

    print(NOVAS_TRANSITIONS)
    
    print("Generating truth tables...")


    keys_local = list(dictFinalTransitions.keys())

    print(keys_local)
    

    if oneOutputPerGene == 1:
    
        for i in range(len(NOVAS_TRANSITIONS)):
            print(str(round((i / len(NOVAS_TRANSITIONS) * 100),2)) + "%")
            elemento = NOVAS_TRANSITIONS[i]
            N_LINHAS = len(elemento) - elemento.count("9")

            #print(TODOS_GENES)
            #print(NOVAS_TRANSITIONS)
            #print(i)
            #PROBLEMA AQUI COM GENE_ATUAL = TODOS_GENES[i]
            #O VETOR TODOS OS GENES NÃO TEM O MESMO TAMANHO DE NOVAS TRANSITIONS QUANDO USA -FD... REAVALIAR
            #APARENTEMENTE O PROBLEMA ESTÁ NA DISCRETIZAÇÃO... OS ARQUIVOS DISCRETIZADOS ESTÃO TODOS IGUAIS AO ORIGINAL, DEVERIAM TER SOMENTE OS GENES PRESENTES EM GENENAMES
            gene_atual = TODOS_GENES[i]
            

            if (ignoreTargetGenes == 1):
                
                
                nome_arquivo_saida = outputDir + str(gene_atual) + "_" + problem_name + ".txt"
                cabecalho = str(GLOBAL_NI[0]) + " 1 " + str(N_LINHAS) + "\n" 
                
                arquivo_saida = open(nome_arquivo_saida, "w")
                arquivo_saida.write(cabecalho)
                for elemento2 in range(len(elemento)):
                    
                    if elemento[elemento2] != "9":
                        string = ""
                        entradas_p_arquivo = str(keys_local[elemento2])
                        for g1 in entradas_p_arquivo:
                            string += g1
                            string += " "
                        saida_p_arquivo = str(elemento[elemento2])
                        string += saida_p_arquivo
                        arquivo_saida.write(string)
                        arquivo_saida.write("\n")
                arquivo_saida.close()
            else:
                if gene_atual in targetGenes:
                    nome_arquivo_saida = outputDir + str(gene_atual) + "_" + problem_name + ".txt"
                    cabecalho = str(GLOBAL_NI[0]) + " 1 " + str(N_LINHAS) + "\n" 
                    
                    arquivo_saida = open(nome_arquivo_saida, "w")
                    arquivo_saida.write(cabecalho)
                    for elemento2 in range(len(elemento)):
                        
                        if elemento[elemento2] != "9":
                            string = ""
                            entradas_p_arquivo = str(keys_local[elemento2])
                            for g1 in entradas_p_arquivo:
                                string += g1
                                string += " "
                            saida_p_arquivo = str(elemento[elemento2])
                            string += saida_p_arquivo
                            arquivo_saida.write(string)
                            arquivo_saida.write("\n")
                            #print(str(keys_local[elemento2]) + " - " + str(elemento[elemento2]))
                    arquivo_saida.close()            

        print("100%")

    else:
        #gera uma tabela só
        print("gera uma tabela só")

        for i in range(len(NOVAS_TRANSITIONS)):
            print(str(round((i / len(NOVAS_TRANSITIONS) * 100),2)) + "%")
            elemento = NOVAS_TRANSITIONS[i]
            N_LINHAS = len(elemento)

            gene_atual = TODOS_GENES[i]
            

            if (ignoreTargetGenes == 1):
                
                
                nome_arquivo_saida = outputDir + "truthTable_" + problem_name + ".txt"
                cabecalho = str(GLOBAL_NI[0]) + " " + str(len(NOVAS_TRANSITIONS)) + " " + str(N_LINHAS) + "\n" 
                
                arquivo_saida = open(nome_arquivo_saida, "w")
                arquivo_saida.write(cabecalho)
                for elemento2 in range(len(elemento)):
                    
                    if elemento[elemento2] != "8":
                        string = ""
                        entradas_p_arquivo = str(keys_local[elemento2])
                        for g1 in entradas_p_arquivo:
                            string += g1
                            string += " "

                        for nt in range(len(NOVAS_TRANSITIONS)):
                            saida_p_arquivo = str(NOVAS_TRANSITIONS[nt][elemento2])
                            string += saida_p_arquivo
                            if nt != len(NOVAS_TRANSITIONS) - 1:
                                string += " "
                        #saida_p_arquivo = str(elemento[elemento2])
                        #print("aqui")
                        #print(NOVAS_TRANSITIONS)
                        #print("aqui elemento dentro")
                        #print(elemento)

                        #print(saida_p_arquivo)
                        #string += saida_p_arquivo
                        #if lsaida != len(elemento) - 1:
                        #    string += " "
                        #string += saida_p_arquivo
                        arquivo_saida.write(string)
                        arquivo_saida.write("\n")
                arquivo_saida.close()
            else:
                #NÃO PENSADO - NÃO FUNCIONANDO
                print("Não faz sentido ter para correlationMethods")
                '''
                if gene_atual in targetGenes:
                    nome_arquivo_saida = outputDir + "truthTable_" + problem_name + ".txt"
                    cabecalho = str(GLOBAL_NI[0]) + " " + str(len(NOVAS_TRANSITIONS)) + " " + str(N_LINHAS) + "\n" 
                    
                    arquivo_saida = open(nome_arquivo_saida, "w")
                    arquivo_saida.write(cabecalho)
                    for elemento2 in range(len(elemento)):
                        
                        if elemento[elemento2] != "9":
                            string = ""
                            entradas_p_arquivo = str(keys_local[elemento2])
                            for g1 in entradas_p_arquivo:
                                string += g1
                                string += " "
                            saida_p_arquivo = str(elemento[elemento2])
                            string += saida_p_arquivo
                            arquivo_saida.write(string)
                            arquivo_saida.write("\n")
                            #print(str(keys_local[elemento2]) + " - " + str(elemento[elemento2]))
                    arquivo_saida.close()
                '''

        print("100%")        

        
    print("Final memory usage: ", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2) 

    print("End")
    
    t1_stop = perf_counter()
    print("Total elapsed time:", t1_stop-t1_start)
