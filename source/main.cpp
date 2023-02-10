#include <iostream>

#include "cgp.h"
#include <fstream>
#include <map>
#include "utils.h"
#include "GPTime.h"
#include "OCLConfig.h"


//std::vector<int>
void stackLocalNode(Chromosome *c, unsigned int nodePos, std::vector<int> *vetor, int nInputs){
    //std::cout << c->nodes[nodePos].inputs[0] << std::endl;
    //std::cout << c->nodes[nodePos].inputs[1] << std::endl;
    if(c->nodes[nodePos].function == 14){
        if(c->nodes[nodePos].inputs[0] >= nInputs){
            if ( std::find(vetor->begin(), vetor->end(), c->nodes[nodePos].inputs[0] - nInputs) == vetor->end() ){
                vetor->push_back(c->nodes[nodePos].inputs[0] - nInputs);
                stackLocalNode(c, c->nodes[nodePos].inputs[0] - nInputs, vetor, nInputs);
            }
        }
        else{
            if ( std::find(vetor->begin(), vetor->end(), c->nodes[nodePos].inputs[0] - nInputs) == vetor->end() ){
                vetor->push_back(c->nodes[nodePos].inputs[0] - nInputs);
            }
        }
    }
    else{
        if(c->nodes[nodePos].inputs[0] >= nInputs){
            if ( std::find(vetor->begin(), vetor->end(), c->nodes[nodePos].inputs[0] - nInputs) == vetor->end() ){
                vetor->push_back(c->nodes[nodePos].inputs[0] - nInputs);
                stackLocalNode(c, c->nodes[nodePos].inputs[0] - nInputs, vetor, nInputs);
            }
        }
        else{
            if ( std::find(vetor->begin(), vetor->end(), c->nodes[nodePos].inputs[0] - nInputs) == vetor->end() ){
                vetor->push_back(c->nodes[nodePos].inputs[0] - nInputs);
            }
        }
        if(c->nodes[nodePos].inputs[1] >= nInputs){
            if ( std::find(vetor->begin(), vetor->end(), c->nodes[nodePos].inputs[1] - nInputs) == vetor->end() ){
                vetor->push_back(c->nodes[nodePos].inputs[1] - nInputs);
                stackLocalNode(c, c->nodes[nodePos].inputs[1] - nInputs, vetor, nInputs);
            }
        }
        else{
            if ( std::find(vetor->begin(), vetor->end(), c->nodes[nodePos].inputs[1] - nInputs) == vetor->end() ){
                vetor->push_back(c->nodes[nodePos].inputs[1] - nInputs);
            }
        }


    }
}


void printFileFiveExe(Chromosome *c, Parameters *p, std::ofstream& factivel_file) {
    for(int i = 0; i < MAX_NODES; i++) {
        if (c->nodes[i].active) {
            factivel_file << "Node" << i + p->N << " " << c->nodes[i].inputs[0]
            << " " << c->nodes[i].inputs[1] << " " <<  c->nodes[i].function << "\n";
        }
    }
    for(int i = 0; i < MAX_OUTPUTS; i++){
        factivel_file << "Output " << c->output[i] + p->N << "\n";
    }

    factivel_file << "\n";
    factivel_file << "\n";
}

int main(int argc, char** argv) {


    std::string newSeed = argv[5];
    std::string geneNamesStr = argv[1];
    // std::cout << geneNamesStr << std::endl;
    std::string argExe = argv[2];
    std::string argProblemName = argv[3];
    std::ifstream geneNamesFile(geneNamesStr);
    std::vector<std::string> geneNames;
    std::string gene;
    std::string truthTableDirectory = argv[6];
    std::string fullTruthTable = argv[7];

    //std::cout << "aqui = " << fullTruthTable << std::endl;
    int numGenes = 0;

    while(std::getline (geneNamesFile, gene)) {
        geneNames.push_back(gene);
        numGenes++;
    }

    // std::cout << geneNames.size() << std::endl;

    geneNamesFile.close();

    std::string currentGene = argv[4];

    std::vector<int> todasRedes;
    std::string datasetFile;
    //std::string datasetFile = currentGene + "_" + argProblemName + ".bin";
    if(atoi(argv[7]) == 0){
        datasetFile = truthTableDirectory + "/" + currentGene + "_" + argProblemName + ".txt";
    }
    else{
        datasetFile = truthTableDirectory + "/" + "truthTable" + "_" + argProblemName + ".txt";
    }



#if PARALLEL

    std::string unfeasiblesFile = "./executions_parallel/" + argExe + "/unfeasibles_" + argProblemName + ".txt";
    std::string rankedEdgesFile = "./executions_parallel/" + argExe + "/rankedEdges_" + argProblemName + ".csv";
    std::ofstream rankedEdges;
    std::ofstream unfeasibles;

    // std::cout << unfeasiblesFile << std::endl;
    // std::cout << rankedEdgesFile << std::endl;

    rankedEdges.open(rankedEdgesFile, std::ios_base::app);
     if (!rankedEdges) {
        std::cout << "Error file ranked edges" << std::endl;
        exit(1);
    }
    unfeasibles.open(unfeasiblesFile, std::ios_base::app);
     if (!unfeasibles) {
        std::cout << "Error file unfeasibles" << std::endl;
        exit(1);
    }

    std::ofstream factivelFile;

    std::string nomeArquivo = currentGene + "_" + newSeed + "_" + argExe;
    std::string caminhoArquivo = "./executions_parallel/" + argExe + "/" + nomeArquivo + ".txt";
    std::cout << caminhoArquivo << std::endl;
    factivelFile.open(caminhoArquivo, std::ios::out);
    if (!factivelFile) {
        std::cout << "Error file factivelFile" << std::endl;
        exit(1);
    }

    std::string caminhoArquivoTime = "./time_counting/" + argExe + "/" + nomeArquivo + ".txt";
    FILE *f_CGP_time_parallel = fopen(caminhoArquivoTime.c_str(), "w");
#else

    std::string unfeasiblesFile = "./executions_sequential/" + argExe + "/unfeasibles_" + argProblemName + ".txt";
    std::string rankedEdgesFile = "./executions_sequential/" + argExe + "/rankedEdges_" + argProblemName + ".csv";
    std::ofstream rankedEdges;
    std::ofstream unfeasibles;
    rankedEdges.open(rankedEdgesFile, std::ios_base::app);
    unfeasibles.open(unfeasiblesFile, std::ios_base::app);

    std::ofstream factivelFile;

    std::string nomeArquivo = currentGene + "_" + newSeed + "_" + argExe;
    std::string caminhoArquivo = "./executions_sequential/" + argExe + "/" + nomeArquivo + ".txt";
    factivelFile.open(caminhoArquivo, std::ios::out);
    if (!factivelFile) {
        std::cout << "Error file" << std::endl;
        exit(1);
    }

    std::string caminhoArquivoTime = "./time_counting_sequential/" + argExe + "/" + nomeArquivo + ".txt";
    FILE *f_CGP_time_sequential = fopen(caminhoArquivoTime.c_str(), "w");
#endif

#if DEFAULT
    /*resultFile = "./results/cgpann_standard.txt";
    resultFileTime = "./results/cgpann_time_standard.txt";
    resultFileTimeIter = "./results/cgpann_timeIter_standard.txt";
    resultFileTimeKernel = "./results/cgpann_timeKernel_standard.txt";*/
#elif COMPACT
    resultFile = "./results/cgpann_compact.txt";
    resultFileTime = "./results/cgpann_time_compact.txt";
    resultFileTimeIter = "./results/cgpann_timeIter_compact.txt";
    resultFileTimeKernel = "./results/cgpann_timeKernel_compact.txt";
#elif IMAGE_R
    resultFile = "./results/cgpann_img_r.txt";
    resultFileTime = "./results/cgpann_time_img_r.txt";
    resultFileTimeIter = "./results/cgpann_timeIter_img_r.txt";
    resultFileTimeKernel = "./results/cgpann_timeKernel_img_r.txt";
#elif IMAGE_RG
    resultFile = "./results/cgpann_img_rg.txt";
    resultFileTime = "./results/cgpann_time_img_rg.txt";
    resultFileTimeIter = "./results/cgpann_timeIter_img_rg.txt";
    resultFileTimeKernel = "./results/cgpann_timeKernel_img_rg.txt";
#elif IMAGE_RGBA
    resultFile = "./results/cgpann_img_rgba.txt";
    resultFileTime = "./results/cgpann_time_img_rgba.txt";
    resultFileTimeIter = "./results/cgpann_timeIter_img_rgba.txt";
    resultFileTimeKernel = "./results/cgpann_timeKernel_img_rgba.txt";
#elif COMPACT_R
    resultFile = "./results/cgpann_compact_img_r.txt";
    resultFileTime = "./results/cgpann_time_compact_img_r.txt";
    resultFileTimeIter = "./results/cgpann_timeIter_compact_img_r.txt";
    resultFileTimeKernel = "./results/cgpann_timeKernel_compact_img_r.txt";
#elif COMPACT_RG
    resultFile = "./results/cgpann_compact_img_rg.txt";
    resultFileTime = "./results/cgpann_time_compact_img_rg.txt";
    resultFileTimeIter = "./results/cgpann_timeIter_compact_img_rg.txt";
    resultFileTimeKernel = "./results/cgpann_timeKernel_compact_img_rg.txt";
#elif  COMPACT_RGBA
    resultFile = "./results/cgpann_compact_img_rgba.txt";
    resultFileTime = "./results/cgpann_time_compact_img_rgba.txt";
    resultFileTimeIter = "./results/cgpann_timeIter_compact_img_rgba.txt";
    resultFileTimeKernel = "./results/cgpann_timeKernel_compact_img_rgba.txt";
#else
    resultFile = "./results/cgpann.txt";
    resultFileTime = "./results/cgpann_time.txt";
    resultFileTimeIter = "./results/cgpann_timeIter.txt";
    resultFileTimeKernel = "./results/cgpann_timeKernel.txt";
#endif


    GPTime timeManager(4);
    timeManager.getStartTime(Total_T);

    Parameters *params;
    params = new Parameters;

    Dataset fullData;
    readDataset(params, &fullData, datasetFile);
    //readDataset_2(params, &fullData, datasetFile);
    std::cout << "-----------------PRINT DATASET-------------------" << std::endl;
    printDataset(&fullData);
    std::cout << "-----------------PRINT DATASET-------------------" << std::endl;


    int trainSize, validSize, testSize;
    calculateDatasetsSize(&fullData, &trainSize, &validSize, &testSize);

    /**OPENCL CONFIG */
    OCLConfig* ocl = new OCLConfig();
    ocl->allocateBuffers(params, trainSize, validSize, testSize);
    ocl->setNDRages();
    ocl->setCompileFlags();
    ocl->buildProgram(params, &fullData, "kernels/kernel.cl");
    ocl->buildKernels();
#if IMAGE_R
    ocl->setupImageBuffers();
#elif IMAGE_RG
    ocl->setupImageBuffersHalf();
#elif IMAGE_RGBA
    ocl->setupImageBuffersQuarter();
#elif  COMPACT_R
    ocl->setupImageBuffersCompact();
#elif  COMPACT_RG
    ocl->setupImageBuffersHalfCompact();
#elif  COMPACT_RGBA
    ocl->setupImageBuffersQuarterCompact();
#endif
    /**OPENCL CONFIG */


    // O newSeed será o valor da SEED
    
    int* seeds;
    seeds = new int [ocl->maxLocalSize * NUM_INDIV];
    //seeds = new int [1];

    srand(atoi(argv[5]));

    /*random seeds used in parallel code*/
    for(int i = 0; i < ocl->maxLocalSize * NUM_INDIV; i++){
        seeds[i] = atoi(argv[5]);
    }


    std::vector<int> rede;
    int countUnfeasible;
    std::vector<int> rede_local;

    Dataset* trainingData = &fullData;

    ocl->transposeDatasets(trainingData);
    double timeIter = 0;
    double timeIterTotal = 0;
    double timeKernel = 0;
    timeManager.getStartTime(Evolucao_T);
#if PARALLEL
    Chromosome* executionBest = PCGP(trainingData, params, ocl, seeds, &timeIter, &timeKernel, factivelFile);

    for(int i = 0; i < NUM_EXECUTIONS; i++) {
        std::cout << "Fitness - exe " << i << " : " << executionBest[i].fitness << std::endl;
        if(executionBest[i].fitness == (params->M * params->O))
            printFileFiveExe(&executionBest[i], params, factivelFile);
        else
            factivelFile << "Nao factivel\n\n";
    }

    countUnfeasible = 0;

    for(int i = 0; i < NUM_EXECUTIONS; i++) {
        if(executionBest[i].fitness != (params->M * params->O)) {
            countUnfeasible += 1;
            continue;
        }

        for(int j = 0; j < MAX_NODES; j++){
            if(executionBest[i].nodes[j].active == 1){
                for(int k = 0; k < MAX_ARITY; k++){
                    if(executionBest[i].nodes[j].inputs[k] < trainingData->N){
                        //std::cout << "Input: " << k << " " << executionBest.nodes[j].inputs[k] << std::endl;
                        auto search = find(rede_local.begin(), rede_local.end(), executionBest[i].nodes[j].inputs[k]);
                        if(search == rede_local.end()){
                            rede_local.push_back(executionBest[i].nodes[j].inputs[k]);
                        }
                    }
                }

            }
        }

        for(int & j : rede_local){
            rede.push_back(j);
        }

        rede_local.clear();
    }


    //std::cout << executionBest[0].nodes[10].inputs[0] << std::endl;
    //std::cout << executionBest[0].nodes[11].inputs[0] << std::endl;
    /*for(int i = 0; i < MAX_NODES; i++){
        std::cout << "Node " << i << " input 0 " << executionBest[1].nodes[i].inputs[0] << " input 1 " << executionBest[1].nodes[i].inputs[1] << std::endl;
    }
    for(int i = 0; i < MAX_OUTPUTS; i++){
        std::cout << "Output " << i << " valor " << executionBest[1].output[i] << std::endl;
    }*/


    std::vector<std::vector<int>> inputsPerOutput;
    for(int i = 0; i < MAX_OUTPUTS; i++){
        inputsPerOutput.push_back({});
    }

    for(int i = 0; i < NUM_EXECUTIONS; i++){
        //std::cout << "Individual " << i << std::endl;
        if(executionBest[i].fitness >= ((params-> M * params -> O)*0.99)){
            std::vector<int> individualStack;
            for(int j = 0; j < MAX_OUTPUTS; j++){
                std::vector<int> localNodesPerOutput;

                
                //std::cout << "Gene " << geneNames[j] << " eh regulado por" << std::endl;

                localNodesPerOutput.push_back(executionBest[i].output[j]);


                stackLocalNode(&executionBest[i], executionBest[i].output[j], &localNodesPerOutput, params->N);

                for(int k = 0; k < localNodesPerOutput.size(); k++){
                    if(localNodesPerOutput[k] < 0){
                        //std::cout << "k = " << k << " gene " << geneNames[localNodesPerOutput[k] + params -> N] << std::endl;
                        inputsPerOutput[j].push_back(localNodesPerOutput[k] + params->N);
                    }
                    //std::cout << "k = " << k << " valor " << localNodesPerOutput[k] << std::endl;
                }
            }
        }
    }

    //std::cout << "Chegou aqui " << inputsPerOutput.size() << std::endl;

    //for(int i = 0; i < inputsPerOutput.size(); i++){
    //    std::cout << "Saida atual: " << i << std::endl;
    //    for(int j = 0; j < inputsPerOutput[i].size(); j++){
    //        std::cout << "Valor: " << inputsPerOutput[i][j] << " gene " << geneNames[inputsPerOutput[i][j]] << std::endl;
    //    }
    //}


    if(MAX_OUTPUTS > 1){
        std::vector<std::vector<int>> countInputsPerOutput;
        for(int i = 0; i < MAX_OUTPUTS; i++){
            countInputsPerOutput.push_back({});
        }

        for(int i = 0; i < inputsPerOutput.size(); i++){
            //std::cout << "Target " << i << std::endl;
            for(int j = 0; j < params->N; j++){
                float counted = std::count(inputsPerOutput[i].begin(), inputsPerOutput[i].end(), j);
                //std::cout << "Current input count " << j << " value " << counted << std::endl;
                if(counted != 0){
                    rankedEdges << geneNames[j] << "\t" << geneNames[i] << "\t" << counted/NUM_EXECUTIONS << "\n";
                }
                countInputsPerOutput[i].push_back(counted/NUM_EXECUTIONS);
            }
        }
    }



#else
    Chromosome* executionBest = CGP(trainingData, params, seeds, &timeIter, &timeKernel, factivelFile);

    for(int i = 0; i < NUM_EXECUTIONS; i++) {
        std::cout << "Fitness - exe " << i << " : " <<executionBest[i].fitness << std::endl;
        if(executionBest[i].fitness == params->M)
            printFileFiveExe(&executionBest[i], params, factivelFile);
        else
            factivelFile << "Nao factivel\n\n";
    }

    countUnfeasible = 0;

    for(int i = 0; i < NUM_EXECUTIONS; i++) {
        if(executionBest[i].fitness != params->M) {
            countUnfeasible += 1;
            continue;
        }

        for(int j = 0; j < MAX_NODES; j++){
            if(executionBest[i].nodes[j].active == 1){
                for(int k = 0; k < MAX_ARITY; k++){
                    if(executionBest[i].nodes[j].inputs[k] < trainingData->N){
                        //std::cout << "Input: " << k << " " << executionBest.nodes[j].inputs[k] << std::endl;
                        auto search = find(rede_local.begin(), rede_local.end(), executionBest[i].nodes[j].inputs[k]);
                        if(search == rede_local.end()){
                            rede_local.push_back(executionBest[i].nodes[j].inputs[k]);
                        }
                    }
                }

            }
        }

        for(int & j : rede_local){
            rede.push_back(j);
        }

        rede_local.clear();
    }

#endif
    timeManager.getEndTime(Evolucao_T);
    timeIterTotal = timeManager.getElapsedTime(Evolucao_T);
    printf("Evol time: %f \n", timeIterTotal);

#if PARALLEL
    /*fprintf(f_CGP_time_parallel, "Fitness best: \t%.4f\n", executionBest.fitness);*/
    fprintf(f_CGP_time_parallel, "timeIter: \t%.4f\n", timeIter);
    fprintf(f_CGP_time_parallel, "timeIterTotal: \t%.4f\n", timeIterTotal);
    fprintf(f_CGP_time_parallel, "timeKernel: \t%.4f\n\n", timeKernel);
#else
    fprintf(f_CGP_time_sequential, "Fitness best: \t%.4f\n", executionBest->fitness);
    fprintf(f_CGP_time_sequential, "timeIter: \t%.4f\n", timeIter);
    fprintf(f_CGP_time_sequential, "timeIterTotal: \t%.4f\n", timeIterTotal);
    fprintf(f_CGP_time_sequential, "timeKernel: \t%.4f\n\n", timeKernel);
#endif

    timeManager.getEndTime(Total_T);

    std::vector<float> counting;

    for(int i = 0; i < fullData.N; i++){
        float counted = std::count(rede.begin(), rede.end(), i);
        counting.push_back(counted/NUM_EXECUTIONS);
    }

    // std::cout << "xxxxxx Contagem xxxxxx" << std::endl;

    // for(int i = 0; i < counting.size(); i++){
    //     std::cout << counting.at(i) << " ";
    // }

    // std::cout << std::endl;

    
    //RANKED EDGES ORIGINAL
    if(MAX_OUTPUTS == 1){
        for(int i = 0; i < geneNames.size(); i++) {
            if(counting.at(i) != 0) {
                rankedEdges << geneNames[i] << "\t" << currentGene << "\t" << counting.at(i) << "\n";
            }
        }
    }

    if(countUnfeasible == NUM_EXECUTIONS) {
        unfeasibles << currentGene << "\n";
    }


#if PARALLEL
    fprintf(f_CGP_time_parallel, "\n");
    fprintf(f_CGP_time_parallel, "Total time: \t%.4f\n", timeManager.getElapsedTime(Total_T));
    fprintf(f_CGP_time_parallel, "\n");
#else
    fprintf(f_CGP_time_sequential, "\n");
    fprintf(f_CGP_time_sequential, "Total time: \t%.4f\n", timeManager.getElapsedTime(Total_T));
    fprintf(f_CGP_time_sequential, "\n");
#endif

    std::cout << "Total time  = " << timeManager.getElapsedTime(Total_T) << std::endl;


    factivelFile.close();
    delete params;

    rankedEdges.close();
    unfeasibles.close();
    return 0;
}