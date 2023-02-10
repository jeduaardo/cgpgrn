//
// Created by bruno on 04/02/2020.
//

#include "utils.h"
#include <iostream>

int rand2(int *seed){
    int s  = *seed;
    //s = ((unsigned short int)(s * 16807) % 2147483647);//(int)(pown(2.0, 31)-1));
    s = ((unsigned int)(s * 16807) % 2147483647);//(int)(pown(2.0, 31)-1));
    *seed = s;

    return s;
}


//unsigned short int randomInput(Parameters *p, unsigned short int index, int *seed) {
unsigned int randomInput(Parameters *p, unsigned short int index, int *seed) {
    return (rand() % (p->N + index));
    //return (rand2(seed) % (p->N + index));
}

//unsigned short int randomOutputIndex(int* seed){
unsigned int randomOutputIndex(int* seed){    
    return (rand() % MAX_NODES);
    //return (rand2(seed) % MAX_NODES);
}

//unsigned short int randomFunction(Parameters *p, int *seed) {
unsigned int randomFunction(Parameters *p, int *seed) {    
    return (rand() % (p->NUM_FUNCTIONS));
    //return (rand2(seed) % (p->NUM_FUNCTIONS));
}

float randomConnectionWeight(Parameters *p, int *seed) {
    return 1;
    //return (((float) rand2(seed) / (float) (2147483647) ) * 2 * p->weightRange) - p->weightRange;
}

int randomInterval(int inf_bound, int sup_bound, int *seed) {
    return rand() % (sup_bound - inf_bound + 1) + inf_bound;
    //return rand2(seed) % (sup_bound - inf_bound + 1) + inf_bound;
}

float randomProb(int* seed){
    return (float)rand2(seed) / 2147483647;//pown(2.0, 31);
}

//unsigned short int getFunctionInputs(unsigned short int function){
unsigned int getFunctionInputs(unsigned int function){    
    switch (function) {
        case ADD:
        case SUB:
        case MUL:
        case DIV:
        case AND:
        case OR:
        case XOR:
        case NAND:
        case NOR:
        case XNOR:
        case SIG:
        case GAUSS:
        case STEP:
        case SOFTSIGN:
        case TANH:
            return MAX_ARITY;
        case RAND:
        case PI:
        case ONE:
        case ZERO:

            return 0;
        case ABS:
        case SQRT:
        case SQ:
        case CUBE:
        case EXP:
        case SIN:
        case COS:
        case TAN:
        case NOT:
        case WIRE:
            return 1;
        case POW:
            return 2;
        default:
            break;
    }
}

bool IsPowerOf2( int n ){
    return (n & -n) == n;
}

//unsigned short int NextPowerOf2( unsigned short int n ){
unsigned int NextPowerOf2( unsigned short int n ){    
    n--;
    n |= n >> 1;  // handle  2 bit numbers
    n |= n >> 2;  // handle  4 bit numbers
    n |= n >> 4;  // handle  8 bit numbers
    n |= n >> 8;  // handle 16 bit numbers
    n |= n >> 16; // handle 32 bit numbers
    n++;

    return n;
}

std::string ToString( double t ){
    std::stringstream ss; ss << std::setprecision(32) << t; return ss.str();
}

void readDataset(Parameters* params, Dataset* fulldata, std::string filename){

    std::fstream arq;

    int i, j, k;
    int readLabel = 0;
    int readOps;
    int info;

    std::cout << "Lendo Dados Arquivo... " << filename << std::endl;
    arq.open(filename, std::fstream::in);

    /** Read the dataset size (M) and number of inputs (N) */
    std::string value;
/*
    arq >> value;
    if(value == ".p")
        arq >> (params->M);
    arq >> value;
    if(value == ".i")
        arq >> (params->N);
    arq >> value;
    if(value == ".o")
        arq >> (params->O);
*/
    arq >> (params->N);
    arq >> (params->O);
    arq >> (params->M);


    //arq >> (readLabel);

    unsigned int M = params->M;
    unsigned int N = params->N;
    unsigned int O = params->O;
    //std::cout << M << " " << N << " " << O << std::endl;

    fulldata->M = M;
    fulldata->N = N;
    fulldata->O = O;

    //(fulldata->data) = new unsigned short int* [(M)];
    (fulldata->data) = new float* [(M)];
    for(i = 0; i < (M); i++){
        //(fulldata->data)[i] = new unsigned short int [(N)];
        (fulldata->data)[i] = new float [(N)];
    }

    //(fulldata->output) = new unsigned short int* [(M)];
    (fulldata->output) = new float* [(M)];
    for(i = 0; i < (M); i++) {
        //(fulldata->output)[i] = new unsigned short int[(O)];
        (fulldata->output)[i] = new float[(O)];
    }

    (params->labels) = new char* [(N + O)];
    for(i = 0; i < (N + O); i++){
        (params->labels)[i] = new char [10];
    }

    //LABELS
    for(i = 0; i < params->N; i++){
        std::stringstream ss;
        std::string str;
        ss << "i";
        ss << i;
        ss >> str;
        strcpy((params->labels)[i], (str.c_str()));
    }
    for(; i < params->N+params->O; i++){
        std::stringstream ss;
        std::string str;
        ss << "o";
        ss << i;
        ss >> str;
        strcpy((params->labels)[i], (str.c_str()));
    }


    /** Read the dataset */
    std::string line;
    for(i = 0; i < (M); i++){
        //arq >> line;
        //std::cout << line <<std::endl;
        for(j = 0; j < (N); j++){
            arq >> (fulldata->data)[i][j] ;//= line[j] - '0';
            //std::cout << (*dataset)[i][j] << " ";
        }
        for(k = 0; j<(N+O); j++, k++){
            arq >> (fulldata->output)[i][k];// = line[j] - '0';
            //std::cout << (*outputs)[i][k] << " ";
        }
        //std::cout << std::endl;
    }

    arq >> readOps;


    params->NUM_FUNCTIONS = 7;
    //(params->functionSet) = new unsigned short int [params->NUM_FUNCTIONS];
    (params->functionSet) = new unsigned int [params->NUM_FUNCTIONS];

    i = 0;

    (params->functionSet)[0] = AND;
    //(params->maxFunctionInputs)[i++] = 2;

    (params->functionSet)[1] = OR;
    //(params->maxFunctionInputs)[i++] = 2;

    (params->functionSet)[2] = XOR;
    //(params->maxFunctionInputs)[i++] = 2;

    (params->functionSet)[3] = NAND;
    //(params->maxFunctionInputs)[i++] = 2;

    (params->functionSet)[4] = NOR;
    //(params->maxFunctionInputs)[i++] = 2;

    (params->functionSet)[5] = XNOR;
    //(params->maxFunctionInputs)[i++] = 2;


    (params->functionSet)[6] = NOT;
    //(params->maxFunctionInputs)[i++] = 1;



    params->weightRange = 5;
}

void printDataset(Dataset* data){
    //int i, j;
    unsigned int i, j;

    std::cout << "Dataset" << std::endl; 
    for(i = 0; i < data->M; i++){
        std::cout << i << " - ";
        for(j = 0; j < data->N; j++) {
            std::cout << data->data[i][j] << " ";
        }
        std::cout << "| ";
        for(j = 0; j < data->O; j++) {
            std::cout << data->output[i][j] << " ";
        }
        std::cout << std::endl;
    }
}



//bool stopCriteria(unsigned short int it){
bool stopCriteria(unsigned long int it){    
    return it < NUM_GENERATIONS;
    //return (it * NUM_INDIV < NUM_EVALUATIONS);
}



Dataset* generateFolds(Dataset* data, int* indexesData, int* indexesDataInFolds){

    std::cout << "Generating folds... " << std::endl;
    int i, j, k, l, count;

    Dataset* folds;
    folds = new Dataset[KFOLDS];

    int foldsSize = (int)data->M/KFOLDS;
    //std::cout << "foldsSize " << foldsSize << std::endl;
    int excessData= data->M % KFOLDS;
    //std::cout << "excessData " << excessData << std::endl;

    for (i = 0; i < KFOLDS; i++)
    {
        folds[i].N = data->N;
        folds[i].O = data->O;
        folds[i].M = foldsSize;
    }
/*
    i = 0;
    count = 0;
    while(1) // set the size of each fold
    {
        folds[i].M = folds[i].M + 1;
        count++;
        if(count == data->M)
            break;
        if(i == 9)
            i = 0;
        else
            i++;
    }
*/
    // allocate memory for the folds data
    for(i = 0; i < KFOLDS; i++) // for each fold
    {
        //folds[i].data = new unsigned short int* [folds[i].M];
        folds[i].data = new float* [folds[i].M];
        //folds[i].output = new unsigned short int* [folds[i].M];
        folds[i].output = new float* [folds[i].M];

        for(j = 0; j < folds[i].M; j++) // for each instance of each fold
        {
            //folds[i].data[j] = new unsigned short int [folds[i].N];
            folds[i].data[j] = new float [folds[i].N];
            //folds[i].output[j] = new unsigned short int [folds[i].O];
            folds[i].output[j] = new float [folds[i].O];
        }
    }


    // keep the same class proportion in each fold
    int counter[KFOLDS];// k = 10 // = (int*)malloc(10*sizeof(int));
    for(i = 0; i < KFOLDS; i++)
    {
        counter[i] = 0;
    }


    //printDataset(folds[0]);
    k = 0;
    for(i = 0; i < data->O; i++) // for each class
    {
        for(j = 0; j < data->M - excessData; j++) // for each instance
        {
            std::cout << "Cheguei aqui... " << data->output[j][i] << std::endl;
            if(data->output[j][i] == 1.0)
            {
                std::cout << "Entrei" << std::endl;
                for (l = 0; l < data->N; l++)
                {
                    folds[k].data[counter[k]][l] = data->data[j][l];
                }

                for (l = 0; l < data->O; l++)
                {
                    folds[k].output[counter[k]][l] = data->output[j][l];
                }

                indexesDataInFolds[counter[k] + k * foldsSize] = indexesData[j];

                counter[k] = counter[k] + 1;
                if(k == (KFOLDS - 1))
                    k = 0;
                else
                    k++;
            }
        }
    }
    printDataset(folds);

    return folds;
}


void calculateDatasetsSize(Dataset* data, int* trainSize, int* validationSize, int* testSize){
    /// Garantir que todos os folds tem o mesmo tamanho para que os tamanhos dos kernels/buffers não precisem mudar

    int foldsSize  = (int)data->M/KFOLDS;

    *trainSize = TRAIN_FOLDS * foldsSize;
    *validationSize = VALID_FOLDS * foldsSize;
    *testSize = TEST_FOLDS * foldsSize;
}

void shuffleData(Dataset* data, int* indexesData, int* seed) {
    //printDataset(data);
    std::cout <<"Shuffling dataset..."<< std::endl;
    for(int i = 0; i < data->M; i++){
        int index1 = randomInterval(0, data->M-1, seed);
        int index2 = randomInterval(0, data->M-1, seed);

        indexesData[index1] = index2;
        indexesData[index2] = index1;

        float* aux1_input = (float*) malloc(data->N * sizeof(float));
        float* aux1_output = (float*) malloc(data->O * sizeof(float));

        std::memcpy(aux1_input, data->data[index1], data->N * sizeof(float));
        std::memcpy(aux1_output, data->output[index1], data->O * sizeof(float));

        std::memcpy(data->data[index1], data->data[index2], data->N * sizeof(float));
        std::memcpy(data->output[index1], data->output[index2], data->O * sizeof(float));

        std::memcpy(data->data[index2], aux1_input, data->N * sizeof(float));
        std::memcpy(data->output[index2], aux1_output, data->O * sizeof(float));

        free(aux1_input);
        free(aux1_output);

    }
    //printDataset(data);
}

Dataset* getSelectedDataset(Dataset* folds, int* indexes, int index_start, int index_end){
    Dataset* newDataset = new Dataset;

    newDataset->N = folds[0].N;
    newDataset->O = folds[0].O;
    newDataset->M = 0;

    for(int i = index_start; i <= index_end; i++){
        newDataset->M += folds[indexes[i]].M;
    }

    //(newDataset->data) = new unsigned short int* [newDataset->M];
    (newDataset->data) = new float* [newDataset->M];
    //(newDataset->output) = new unsigned short int* [newDataset->M];
    (newDataset->output) = new float* [newDataset->M];

    for(int i = 0; i < newDataset->M; i++){
        //(newDataset->data)[i] = new unsigned short int [newDataset->N];
        (newDataset->data)[i] = new float [newDataset->N];
        //(newDataset->output)[i] = new unsigned short int[newDataset->O];
        (newDataset->output)[i] = new float[newDataset->O];
    }

    int l = 0;
    for(int i = index_start; i <= index_end; i++){
        int foldIndex = indexes[i];
        for(int j = 0; j < folds[foldIndex].M; j++){

            for(int k = 0; k < newDataset->N; k++){
                newDataset->data[l][k] = folds[foldIndex].data[j][k];
            }

            for(int k = 0; k < newDataset->O; k++){
                newDataset->output[l][k] = folds[foldIndex].output[j][k];
            }
            l++;
        }
    }
    return newDataset;

}

/* Fisher–Yates shuffle */
void shuffleArray(std::vector<int>* array, int size, int* seed){
    for(int i = size-1; i > 0; i--){
        int index = randomInterval(0, i, seed);

        std::swap((*array)[i], (*array)[index]);
        /*
        int temp = array[i];
        array[i] = array[index];
        array[index] = temp;
         */
    }
}

void getIndexes(int* indices, int k, int excludeIndex, int* seed){

    std::vector<int> indexes;
    for(int i = 0; i < k; i++){
        if(i != excludeIndex){
            indexes.emplace_back(i);
        }
    }
/*
    for(int i = 0; i < indexes.size(); i++){
        std::cout << indexes[i] << " ";
    }
    std::cout << std::endl;
    */

    shuffleArray(&indexes, indexes.size(), seed);

    for(int i = 0; i < indexes.size(); i++){
        indices[i] = indexes[i];
        //std::cout << indices[i]  << " ";
    }
    //std::cout << std::endl;


}