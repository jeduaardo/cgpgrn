//
// Created by bruno on 04/02/2020.
//

#include "cgp.h"
#include "GPTime.h"

//void newNode(Chromosome* c, Parameters* params, unsigned short int index, int* seed){
void newNode(Chromosome* c, Parameters* params, unsigned int index, int* seed){    
    /** set node function */
    c->nodes[index].function = params->functionSet[randomFunction(params, seed)];


    c->nodes[index].maxInputs = getFunctionInputs(c->nodes[index].function);
    //std::cout << "GetFunctionInputs: " << getFunctionInputs(c->nodes[index].function) << std::endl;
    //std::cout << "Function: " << c->nodes[index].function << " - Max Inputs: " << c->nodes[index].maxInputs << std::endl;

    /** set node inputs */
    for(int pos = 0; pos < MAX_ARITY; pos++){
        c->nodes[index].inputs[pos] = randomInput(params, index, seed);
        c->nodes[index].inputsWeight[pos] = randomConnectionWeight(params, seed);
    }

    /** set to unactive as default */
    c->nodes[index].active = 0;

    /** set to output 0 as default */
    //c->nodes[index].output = 0;

    //c->nodes[index].inputsEvaluated = 0;

}

/*
	used by qsort in sortIntArray
*/
static int cmpInt(const void * a, const void * b) {
    return ( *(int*)a - * (int*)b );
}

//void sortActiveArray(unsigned short int *array, const int length) {
void sortActiveArray(unsigned int *array, const int length) {    

    qsort(array, length, sizeof(int), cmpInt);
}



void activateNodes(Chromosome* c, Parameters* p){

    int i, j;
    int alreadyEvaluated[MAX_NODES];
    for(i = 0; i < MAX_NODES; i++) {
        alreadyEvaluated[i] = -1;
        c->activeNodes[i] = MAX_NODES + 1;
        c->nodes[i].active = 0;
    }
    c->numActiveNodes = 0;

    Stack s;
    s.topIndex = -1;


    for(i = 0; i < MAX_OUTPUTS; i++) {
       // unsigned short int nodeIndex = c->output[i];
        unsigned int nodeIndex = c->output[i];
        //std::cout << " node index " << nodeIndex << std::endl;
        push(&s, nodeIndex);

        while(s.topIndex != -1) {
            //unsigned short int node = pop(&s);
            unsigned int node = pop(&s);
            //std::cout << "Morre aqui no node " << node << std::endl;
            if( c->nodes[node].active == 0) {
                for (j = 0; j < MAX_ARITY; j++) {
                    //std::cout << "Morre aqui no j " << j << std::endl;
                    if (c->nodes[node].inputs[j] >= p->N) {
                        push(&s, c->nodes[node].inputs[j] - p->N);
                    }
                }
                c->activeNodes[c->numActiveNodes] = node;
                c->nodes[node].active = 1;
                c->numActiveNodes++;
               // std::cout << "Morre aqui no if" << std::endl;
            }

        }
    }
    sortActiveArray(c->activeNodes, c->numActiveNodes);
}

void circuitGenerator(Chromosome* c, Parameters* params, int* seed){
    //unsigned short int i;
    unsigned int i;

    for(i = 0; i < MAX_NODES; i++){
        newNode(c, params, i, seed);
    }

    for(i = 0; i < MAX_OUTPUTS; i++){
        c->output[i] = randomOutputIndex(seed);
    }

    c->fitness = 0.0;
    c->fitnessValidation = 0.0;
    c->numActiveNodes = 0;
    activateNodes(c, params);
}

void initializePopulation(Chromosome* pop, Parameters* p, int* seed) {
    for(int i = 0; i < NUM_INDIV; i++){
        circuitGenerator(&pop[i], p, seed);
    }
}


void evaluateCircuit(Chromosome* c, Dataset* data) {
    int i;
    c->fitness = 0.0;
    for(i = 0; i < data->M; i++){
        runCircuit(c, data, i, 0);
        //setFitness(c, p, out[i], &fitness);
    }
    //std::cout << "FITNESS INDIVIDUO " << c->fitness << std::endl;
    // c->fitness = c->fitness / (float) data->M;
}

void evaluateCircuitValidation(Chromosome* c, Dataset* data) {
    int i;
    c->fitnessValidation = 0.0;
    for(i = 0; i < data->M; i++){
        runCircuit(c, data, i, 1);
        //setFitness(c, p, out[i], &fitness);
    }
    c->fitnessValidation = c->fitnessValidation / (float) data->M;
}

void evaluateCircuitLinear(Chromosome* c, Dataset* data) {
    int i;
    c->fitness = 0.0;
    for(i = 0; i < data->M; i++){
        runCircuitLinear(c, data, i, 0);
        //setFitness(c, p, out[i], &fitness);
    }
    c->fitness = c->fitness / (float) data->M;
}

void evaluateCircuitValidationLinear(Chromosome* c, Dataset* data) {
    int i;
    c->fitnessValidation = 0.0;
    for(i = 0; i < data->M; i++){
        runCircuitLinear(c, data, i, 1);
        //setFitness(c, p, out[i], &fitness);
    }
    c->fitnessValidation = c->fitnessValidation / (float) data->M;
}

float executeFunction(Chromosome* c, int node, ExStack* exStack){
    int i;
    float result, sum;
    //unsigned short int inputs = c->nodes[node].maxInputs;
    unsigned int inputs = c->nodes[node].maxInputs;

    int r1, r2;



    switch (c->nodes[node].function){
        case ADD:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result += exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        case SUB:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result -= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        case MUL:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result *= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        case DIV:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result /= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        case ABS:
            result = fabs(popEx(exStack));
            break;
        case SQRT:
            result = sqrt(popEx(exStack));
            break;
        case SQ:
            result = pow(popEx(exStack), 2);
            break;
        case CUBE:
            result = pow(popEx(exStack), 3);
            break;
        case POW:
            result = popEx(exStack);
            result = pow(popEx(exStack), result);
            break;
        case AND:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 0){
                    result = 0;
                }
            }
           // std::cout << "AND " << result << std::endl;
            break;
        case OR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result = 1;
                }
            }
           // std::cout << "OR " << result << std::endl;
            break;
        case XOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result += 1;
                }
            }
            if(result != 1){
                result = 0;
            }
            //r1 = popEx(exStack);
            //r2 = popEx(exStack);
            //std::cout << "valor r1: " << r1 << " valor r2: " << r2 << std::endl;


            //std::cout << "XOR " << result << std::endl;
            break;
        case NAND:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 0){
                    result = 1;
                }
            }
            //std::cout << "NAND " << result << std::endl;
            break;
        case NOR:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result = 0;
                }
            }
            //std::cout << "NOR " << result << std::endl;
            break;
        case XNOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result += 1;
                }
            }
            if(result == 1){
                result = 0;
            } else {
                result = 1;
            }
            //std::cout << "XNOR " << result << std::endl;
            break;
        case NOT:
            result = 0;
            if(popEx(exStack) == 0){
                result = 1;
            }
           // std::cout << "NOT " << result << std::endl;
            break;
        case EXP:
            result = exp(popEx(exStack));
            break;
        case SIN:
            result = sin(popEx(exStack));
            break;
        case COS:
            result = cos(popEx(exStack));
            break;case TAN:
            result = tan(popEx(exStack));
            break;
        case ONE:
            result = 1;
            break;
        case ZERO:
            result = 0;
            break;
        case PI:
            result = CONST_PI;
            break;
        case WIRE:
            result = popEx(exStack);
            break;
        case SIG:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = 1.0f / (1 + exp(-sum));
            break;
        case GAUSS:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = exp(-(pow(sum - 0, 2)) / (2 * pow(1, 2)));
            break;

        case STEP:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            if(sum < 0) {
                result = 0;
            } else {
                result = 1;
            }
           break;
        case SOFTSIGN:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = sum / (1 + fabs(sum));
            break;
        case TANH:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = tanh(sum);
            break;
        case RAND:
            //???
        break;
        default:
            break;
    }
    return result;
}

void runCircuit(Chromosome* c, Dataset* dataset, int index, int validation){

    /*std::cout << "RODANDO UM INDIVIDUO" << std::endl;*/
    int i;
    float maxPredicted = -DBL_MAX;
    int predictedClass = 0;
    int correctClass = 0;

    float executionOut[MAX_OUTPUTS];
    float alreadyEvaluated[MAX_NODES];
    int inputsEvaluatedAux[MAX_NODES];

    for(i = 0; i < MAX_NODES; i++){
        alreadyEvaluated[i] = -DBL_MAX;
        inputsEvaluatedAux[i] = 0;
        //c->nodes[i].inputsEvaluated = 0;
    }

    Stack s;
    s.topIndex = -1;

    ExStack exStack;
    exStack.topIndex = -1;


    for( i = 0; i < MAX_OUTPUTS; i++) {
        //std::cout << "Avaliando saida: " << i << std::endl;
        //unsigned short int nodeIndex = c->output[i];
        unsigned int nodeIndex = c->output[i];
        push(&s, nodeIndex);




        while(s.topIndex != -1) {
            //unsigned short int node = pop(&s);
            unsigned int node = pop(&s);
            //std::cout << "Function: " << c->nodes[node].function << " - Max inputs: " << c->nodes[node].maxInputs << std::endl;
            /*if(c->nodes[node].function == 14){
                c->nodes[node].maxInputs = 1;
            }
            else{
                c->nodes[node].maxInputs = 2;
            }*/

            for (int j = inputsEvaluatedAux[node]; j < c->nodes[node].maxInputs; j++) {
                //std::cout << "J ATUAL: " << j << std::endl;
                if (c->nodes[node].inputs[j] >= dataset->N) { // se é um outro nó, empilha nó ou o resultado
                    //unsigned short int refIndex = c->nodes[node].inputs[j] - dataset->N;
                    unsigned int refIndex = c->nodes[node].inputs[j] - dataset->N;

                    if(alreadyEvaluated[refIndex] > -DBL_MAX) {
                        inputsEvaluatedAux[node]++;//c->nodes[node].inputsEvaluated++;
                        //c->nodes->inputsOutputs[j] = alreadyEvaluated[refIndex];
                        pushEx(&exStack, alreadyEvaluated[refIndex]);
                    } else {
                        push(&s, node); // reinsere o nó que nao terminou de ser avaliado
                        push(&s, refIndex); //avalia o próximo
                        break;
                    }
                } else {
                    inputsEvaluatedAux[node]++;//c->nodes[node].inputsEvaluated++;
                    //c->nodes->inputsOutputs[j] = dataset->data[index][c->nodes[node].inputs[j]];
                    //std::cout << "Input 0: " << c->nodes[node].inputs[0] << " - Input 1: " << c->nodes[node].inputs[1] << " - Function: " << c->nodes[node].function << std::endl;
                    pushEx(&exStack, dataset->data[index][c->nodes[node].inputs[j]]);
                }
            }

            if(inputsEvaluatedAux[node] == c->nodes[node].maxInputs){

                if(!(alreadyEvaluated[node] > -DBL_MAX)) {

                    alreadyEvaluated[node] = executeFunction(c, node, &exStack);
                }
                //alreadyEvaluated[node] = executeFunction(c, node, &exStack);
                //pushEx(&exStack, alreadyEvaluated[node]);

                //alreadyEvaluated[node] = 1;
            }

        }
        executionOut[i] = alreadyEvaluated[nodeIndex];//c->nodes[c->output[i]].output;//popEx(&exStack);
        /*std::cout << "---------------------------" << std::endl;
        std::cout << "executionOut[i] " << executionOut[i] << std::endl;
        std::cout << "maxPredicted " << maxPredicted << std::endl;
        std::cout << "output[i] " << c->output[i] << std::endl;
        std::cout << "dataset " << dataset->output[index][i] << std::endl;
        std::cout << "---------------------------" << std::endl;*/

        /*if(executionOut[i] > maxPredicted) {
            maxPredicted = executionOut[i];
            predictedClass = i;
        }

        if(dataset->output[index][i] == 1.0) {
            correctClass = i;
        }*/


        if(dataset->output[index][i] == 9){
            //std::cout << "EH NOVE" << std::endl;
            //std::cout << "Index: " << index << " i: " << i << " Valor desejado: " << dataset->output[index][i] << " - Valor obtido " << executionOut[i] << std::endl;
            (c->fitness)++;
        }
        if(dataset->output[index][i] == executionOut[i]) {
            //std::cout << "Index: " << index << " i: " << i << " Valor desejado: " << dataset->output[index][i] << " - Valor obtido " << executionOut[i] << std::endl;
            (c->fitness)++;
        }
        //std::cout << "- FIM AVALIACAO -" << std::endl;
        //std::cout << "Dataset " << dataset->output[index][i] << std::endl;
        //std::cout << "Execution out: " << executionOut[i] << std::endl;
    }

    /*std::cout << "Fitness " << c->fitness << std::endl;*/

    /*if(predictedClass == correctClass) {
        if(validation == 1){
            (c->fitnessValidation)++;
        } else {
            (c->fitness)++;
        }

    }*/
}

void runCircuitLinear(Chromosome* c, Dataset* dataset, int index, int validation){

    int i, j, currentActive, activeInputs;
    float maxPredicted = -DBL_MAX;
    int predictedClass = 0;
    int correctClass = 0;

    float executionOut[MAX_OUTPUTS];
    float alreadyEvaluated[MAX_NODES];

    for(i = 0; i < MAX_NODES; i++){
        alreadyEvaluated[i] = -DBL_MAX;
    }

    ExStack exStack;
    exStack.topIndex = -1;

    for(i = 0; i < c->numActiveNodes; i++){
        currentActive = c->activeNodes[i];
        activeInputs = c->nodes[currentActive].maxInputs;

        for(j = 0; j < activeInputs; j++){
            if (c->nodes[currentActive].inputs[j] >= dataset->N) { // se é um outro nó, empilha nó ou o resultado
                //unsigned short int refIndex = c->nodes[currentActive].inputs[j] - dataset->N;
                unsigned int refIndex = c->nodes[currentActive].inputs[j] - dataset->N;

                if(alreadyEvaluated[refIndex] > -DBL_MAX) {
                    pushEx(&exStack, alreadyEvaluated[refIndex]);
                } else {
                    printf("ERRO. \n");
                }
            } else {
                pushEx(&exStack, dataset->data[index][c->nodes[currentActive].inputs[j]]);
            }
        }

        alreadyEvaluated[currentActive] = executeFunction(c, currentActive, &exStack);
/*

        if (std::isnan(alreadyEvaluated[currentActive]) != 0) {
            alreadyEvaluated[currentActive] = 0;
        }

        else if (std::isinf(alreadyEvaluated[currentActive]) != 0 ) {

            if (alreadyEvaluated[currentActive] > 0) {
                alreadyEvaluated[currentActive] = DBL_MAX;
            }
            else {
                alreadyEvaluated[currentActive] = DBL_MIN;
            }
        }
*/
    }

    for( i = 0; i < MAX_OUTPUTS; i++) {
        //unsigned short int nodeIndex = c->output[i];
        unsigned int nodeIndex = c->output[i];

        executionOut[i] = alreadyEvaluated[nodeIndex];//c->nodes[c->output[i]].output;//popEx(&exStack);

        if(executionOut[i] > maxPredicted) {
            maxPredicted = executionOut[i];
            predictedClass = i;
        } else {
            maxPredicted = maxPredicted;
            predictedClass = predictedClass;
        }

        correctClass = (dataset->output[index][i] == 1.0) ? i : correctClass;

    }

    if(validation == 1){
        c->fitnessValidation = (predictedClass == correctClass) ? c->fitnessValidation+1 : c->fitnessValidation+0;
    } else {
        c->fitness = (predictedClass == correctClass) ? c->fitness+1 : c->fitness+0;
    }
}

void evaluatePopulation(Chromosome* pop, Dataset* dataset, int validation, int bestIndex[]){
    int i, j, k, l;
    float bestFitness[NUM_EXECUTIONS];
    for(k = 0; k < NUM_EXECUTIONS; k++) {
        bestFitness[k] = 0;
    }

    //unsigned short int bestActiveNodes[NUM_EXECUTIONS];
    unsigned int bestActiveNodes[NUM_EXECUTIONS];

    for(k = 0; k < NUM_EXECUTIONS; k++) {
        bestActiveNodes[k] = 9999;
    }

    for(k = 0; k < NUM_EXECUTIONS; k++) {
        bestIndex[k] = -1;
    }

    for(j = 0; j < NUM_INDIV; j++) {
        evaluateCircuit(&pop[j], dataset);
    }
    std::cout << std::endl;

    int feasibles[NUM_EXECUTIONS];
    for(k = 0; k < NUM_EXECUTIONS; k++) {
        feasibles[k] = 0;
    }

    int prox = 0;
    for(l = 0; l < NUM_EXECUTIONS; l++) {
        for(i = prox; i < prox + NUM_INDIV_POP; i++) {
            if (pop[i].fitness == (dataset->M * dataset->O)) {
                feasibles[l] += 1;
            }
        }
        prox = i;
    }

    std::vector<int> equalFitness;
    int indBest = -1;

    prox = 0;
    for(l = 0; l < NUM_EXECUTIONS; l++) {
        if(feasibles[l] > 0) {
            if(feasibles[l] == 1) {
                for(i = prox; i < prox + NUM_INDIV_POP; i++) {
                    if(pop[i].fitness == (dataset->M * dataset->O)){
                        bestFitness[l] = pop[i].fitness;
                        bestActiveNodes[l] = pop[i].numActiveNodes;
                        bestIndex[l] = i;
                    }
                }
            } else if (feasibles[l] > 1) {
                for(i = prox; i < prox + NUM_INDIV_POP; i++) {
                    if(pop[i].fitness == (dataset->M * dataset->O)){
                        if(pop[i].numActiveNodes < bestActiveNodes[l]) {
                            bestFitness[l] = pop[i].fitness;
                            bestActiveNodes[l] = pop[i].numActiveNodes;
                            bestIndex[l] = i;
                        }
                    }
                }
            }
        } else {
            for(i = prox; i < prox + NUM_INDIV_POP; i++) {
                if(pop[i].fitness > bestFitness[l]) {
                    bestFitness[l] = pop[i].fitness;
                    bestActiveNodes[l] = pop[i].numActiveNodes;
                    bestIndex[l] = i;
                    equalFitness.clear();
                    equalFitness.push_back(i);
                } else if (pop[i].fitness == bestFitness[l]) {
                    equalFitness.push_back(i);
                }
            }

            if(!equalFitness.empty()) {
                if(equalFitness.size() == 1) {
                    bestFitness[l] = pop[equalFitness.at(0)].fitness;
                    bestActiveNodes[l] = pop[equalFitness.at(0)].numActiveNodes;
                    bestIndex[l] = equalFitness.at(0);
                } else {
                    indBest = rand() % (equalFitness.size() - 1);
                    bestFitness[l] = pop[equalFitness.at(indBest)].fitness;
                    bestActiveNodes[l] = pop[equalFitness.at(indBest)].numActiveNodes;
                    bestIndex[l] = equalFitness.at(indBest);
                }
            }
        }
        prox = i;

        equalFitness.clear();
    }
}


Chromosome *mutateTopologyProbabilistic(Chromosome *c, Parameters *p, int *seed, int type) {

    int i, j;
    for(i = 0; i < MAX_NODES; i++){

        if(randomProb(seed) <= PROB_MUT) {
            c->nodes[i].function = p->functionSet[randomFunction(p, seed)];
            c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
        }
        for(j = 0; j < c->nodes[i].maxInputs; j++) {
            if(randomProb(seed) <= PROB_MUT) {
                c->nodes[i].inputs[j] = randomInput(p, i, seed);
            }
            if(type == 0 && randomProb(seed) <= PROB_MUT){
                c->nodes[i].inputsWeight[j] = randomConnectionWeight(p, seed);
            }
        }
    }

    activateNodes(c, p);
    return  c;
}

void copyNode(Node* n, ActiveNode* an){
    an->function = n->function;
    an->maxInputs = n->maxInputs;
    for(int i = 0; i < MAX_ARITY; i++){
        an->inputs[i] = n->inputs[i];
        an->inputsWeight[i] = n->inputsWeight[i];
    }
}

void copyActiveNodes(Chromosome *c, ActiveChromosome* ac){

    for(int i = 0; i < NUM_INDIV; i++ ){

        //unsigned short int numActiveNodes = c[i].numActiveNodes;
        unsigned int numActiveNodes = c[i].numActiveNodes;
        ac[i].numActiveNodes = numActiveNodes;


        for(int j = 0; j < numActiveNodes; j++){
            //unsigned short int currentActive = c[i].activeNodes[j];
            unsigned int currentActive = c[i].activeNodes[j];

            copyNode(&(c[i].nodes[currentActive]), &(ac[i].nodes[j]));
            ac[i].nodes[j].originalIndex = currentActive;
        }


        for(int j = 0; j < MAX_OUTPUTS; j++){
            ac[i].output[j] = c[i].output[j];
        }


    }
}

Chromosome *mutateTopologyProbabilistic2(Chromosome *c, Parameters *p, int *seeds, int type, int index) {

    int i, j;

    for(i = 0; i < MAX_NODES; i++){
        int nodeIndex  = index * 1024 + ((i*MAX_ARITY) % 1024);
        //std::cout << "morreu aqui?  " << i << std::endl;
        //std::cout << "morreu aqui pq?  " << nodeIndex << std::endl;
        /*if(randomProb(&seeds[nodeIndex]) <= PROB_MUT) {
            c->nodes[i].function = p->functionSet[randomFunction(p, &seeds[nodeIndex])];
            c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
        }
*/
        for(j = 0; j < c->nodes[i].maxInputs; j++) {
            int arrayIndex = index * 1024 + ((i*MAX_ARITY + j) % 1024);
            if(randomProb(&seeds[arrayIndex]) <= PROB_MUT) {
                c->nodes[i].inputs[j] = randomInput(p, i, &seeds[arrayIndex]);
            }
            /*if(type == 0 && randomProb(&seeds[arrayIndex]) <= PROB_MUT){
                c->nodes[i].inputsWeight[j] = randomConnectionWeight(p, &seeds[arrayIndex]);
            }*/
            if(j == 0) {
                if(randomProb(&seeds[arrayIndex]) <= PROB_MUT) {
                    c->nodes[i].function = p->functionSet[randomFunction(p, &seeds[arrayIndex])];
                    c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
                }
            }

        }
    }

    activateNodes(c, p);
    return  c;
}

Chromosome *mutateTopologyProbabilisticActive(Chromosome *c, Parameters *p, int *seed, int type) {

    int i, j;
    for(i = 0; i < MAX_NODES; i++){
        if(c->nodes[i].active == 1){
            if(randomProb(seed) <= PROB_MUT) {
                //std::cout << "ANTES " << c->nodes[i].function << std::endl;
                c->nodes[i].function = p->functionSet[randomFunction(p, seed)];
                //c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
                //std::cout << "DEPOIS " << c->nodes[i].function << std::endl;
            }
            for(j = 0; j < c->nodes[i].maxInputs; j++) {
                if(randomProb(seed) <= PROB_MUT) {
                    //std::cout << "ANTES " << c->nodes[i].inputs[j] << std::endl;
                    c->nodes[i].inputs[j] = randomInput(p, i, seed);
                    //std::cout << "DEPOIS " << c->nodes[i].inputs[j] << std::endl;
                }
                if(type == 0 && randomProb(seed) <= PROB_MUT){
                    c->nodes[i].inputsWeight[j] = randomConnectionWeight(p, seed);
                }
            }
        }
    }

    activateNodes(c, p);
    return  c;
}

Chromosome *mutateSAM(Chromosome *c, Parameters *p, int *seed) {
    //std::cout << "inicio mutacao" << std::endl;
    int i, j, inputOrFunction, nodeOrOutput, indOutputToBeModified;
    int activeSelected = 0;

    nodeOrOutput = randomInterval(0, MAX_NODES + p->O - 1, seed);

    if (nodeOrOutput > MAX_NODES - 1) {
        activeSelected = 1;

        indOutputToBeModified = randomInterval(0, MAX_OUTPUTS, seed);
        //std::cout << "SAIDA: " << indOutputToBeModified << std::endl;

        c->output[indOutputToBeModified] = randomInterval(0, MAX_NODES - 1, seed);
    } else {
        while(activeSelected == 0) {
            i = randomInterval(0, MAX_NODES - 1, seed);
            //std::cout << "ind no " << i << " active node " << c->nodes[i].active << std::endl;
            if(c->nodes[i].active == 1) {
                activeSelected = 1;
            }
            inputOrFunction = randomInterval(0, 1, seed);
            if (!inputOrFunction) {
                c->nodes[i].function = p->functionSet[randomFunction(p, seed)];
                c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
            } else {
                j = randomInterval(0, 1, seed);
                c->nodes[i].inputs[j] = randomInput(p, i, seed);
            }
        }
    }

    activateNodes(c, p);
    //std::cout << "fim mutacao" << std::endl;
    return  c;
}

Chromosome *mutateTopologyPoint(Chromosome *c, Parameters *p, int *seed) {
    int mutationComplete = -1;
    //unsigned short int newIndex;
    unsigned int newIndex;
    //unsigned short int newInputIndex;
    unsigned int newInputIndex;
    //unsigned short int newValue;
    unsigned int newValue;

    int num_inputs = MAX_NODES * MAX_ARITY;
    while (mutationComplete == -1){
        //unsigned short int nodeIndex = randomInterval(0, MAX_NODES + (num_inputs) + MAX_OUTPUTS, seed); //Select any node or output
        unsigned int nodeIndex = randomInterval(0, MAX_NODES + (num_inputs) + MAX_OUTPUTS, seed); //Select any node or output
        if(nodeIndex < MAX_NODES) { // select function
            newIndex = nodeIndex;
            newValue = p->functionSet[randomFunction(p, seed)];
            if(newValue != c->nodes[newIndex].function){
                c->nodes[newIndex].function = newValue;
                c->nodes[newIndex].maxInputs = getFunctionInputs(newValue);
                if(c->nodes[newIndex].active > -1) {
                    mutationComplete = 1;
                }
            }
        } else if (nodeIndex <= MAX_NODES + (num_inputs)) { //select input
            //newIndex = (unsigned short int) ((nodeIndex - MAX_NODES) / MAX_ARITY);
            newIndex = (unsigned int) ((nodeIndex - MAX_NODES) / MAX_ARITY);
            //newInputIndex= (unsigned short int) ((nodeIndex - MAX_NODES) % MAX_ARITY);
            newInputIndex= (unsigned int) ((nodeIndex - MAX_NODES) % MAX_ARITY);

            newValue = randomInput(p, newIndex, seed);

            if(newValue != c->nodes[newIndex].inputs[newInputIndex]){
                c->nodes[newIndex].inputs[newInputIndex] = newValue;
                if(c->nodes[newIndex].active == 1) {
                    mutationComplete = 1;
                }
            }

        } else { // select an output
            newIndex = nodeIndex - (MAX_NODES + (num_inputs)) - 1;
            newValue = randomOutputIndex(seed);

            if(newValue != c->output[newIndex]) {
                c->output[newIndex] = newValue;
                mutationComplete = 1;
            }
        }

    }
    activateNodes(c, p);
    return  c;
}




Chromosome*
CGP(Dataset *training, Parameters *params, int *seeds, double *timeIter, double *timeKernel, std::ofstream& factivel_file) {
    GPTime timeManager(4);
    Chromosome *current_pop;
    current_pop = new Chromosome[NUM_INDIV];

    int factivel = 0;
    Chromosome* best = new Chromosome[NUM_EXECUTIONS];
    Chromosome* best_train = new Chromosome[NUM_EXECUTIONS];
    Chromosome mutated_best;

    initializePopulation(current_pop, params, &seeds[0]);

    int bestTrain[NUM_EXECUTIONS];
    evaluatePopulation(current_pop, training, 0, bestTrain);

    //unsigned short int bestActiveNodes[NUM_EXECUTIONS];
    unsigned int bestActiveNodes[NUM_EXECUTIONS];

    for(int k = 0; k < NUM_EXECUTIONS; k++) {
        bestActiveNodes[k] = 9999;
    }

    int feasibles[NUM_EXECUTIONS];
    for(int k = 0; k < NUM_EXECUTIONS; k++) {
        feasibles[k] = 0;
    }

    for(int i = 0; i < NUM_EXECUTIONS; i++) {
        best_train[i] = current_pop[bestTrain[i]];
    }

    for(int i = 0; i < NUM_EXECUTIONS; i++) {
        best[i] = best_train[i];
    }

    int prox, it;
    std::vector<int> equalFitness;
    int indBest = -1;
    int iterations = 0;
    while(stopCriteria(iterations)) {
        timeManager.getStartTime(Iteracao_T);

        //std::cout << "Morre aqui na iteracao " << iterations << std::endl;

        int group = -1;
        for(int k = 0; k < NUM_INDIV; k++){
            if(k % NUM_EXECUTIONS == 0) {
                group++;
            }
            //std::cout << "Morre aqui no k " << k << std::endl;
            current_pop[k] = best[group];
            mutateSAM(&current_pop[k], params, seeds);

        }

        it = 0;
        prox = 0;

        for(int l = 0; l < NUM_EXECUTIONS; l++) {
            for(it = prox; it < prox + NUM_INDIV_POP; it++) {
                timeManager.getStartTime(Avaliacao_T);
                evaluateCircuit(&current_pop[it], training);
                timeManager.getEndTime(Avaliacao_T);

                (*timeKernel) += timeManager.getElapsedTime(Avaliacao_T);
            }
            prox = it;
        }


        it = 0;
        prox = 0;

        for(int l = 0; l < NUM_EXECUTIONS; l++) {
            for(it = prox; it < prox + NUM_INDIV_POP; it++) {
                if (current_pop[it].fitness == params->M) {
                    feasibles[l] += 1;
                }
            }
            prox = it;
        }

        it = 0;
        prox = 0;

        for(int l = 0; l < NUM_EXECUTIONS; l++) {
            if(feasibles[l] > 0) {
                if(feasibles[l] == 1) {
                    for(it = prox; it < prox + NUM_INDIV_POP; it++) {
                        if(current_pop[it].fitness == params->M){
                            best_train[l] = current_pop[it];
                            bestActiveNodes[l] = current_pop[it].numActiveNodes;
                        }
                    }
                } else if (feasibles[l] > 1) {
                    for(it = prox; it < prox + NUM_INDIV_POP; it++) {
                        if(current_pop[it].fitness == params->M){
                            if(current_pop[it].numActiveNodes < bestActiveNodes[l]) {
                                best_train[l] = current_pop[it];
                                bestActiveNodes[l] = current_pop[it].numActiveNodes;
                            }
                        }
                    }
                }
            } else {
                for(it = prox; it < prox + NUM_INDIV_POP; it++) {
                    if(current_pop[it].fitness > best_train[l].fitness){
                        best_train[l] = current_pop[it];
                        bestActiveNodes[l] = current_pop[it].numActiveNodes;
                        equalFitness.clear();
                        equalFitness.push_back(it);
                    } else if (current_pop[it].fitness == best_train[l].fitness) {
                        equalFitness.push_back(it);
                    }
                }

                if(!equalFitness.empty()) {
                    if(equalFitness.size() == 1) {
                        bestActiveNodes[l] = current_pop[equalFitness.at(0)].numActiveNodes;
                        best_train[l] = current_pop[equalFitness.at(0)];
                    } else {
                        indBest = rand() % (equalFitness.size() - 1);
                        bestActiveNodes[l] = current_pop[equalFitness.at(indBest)].numActiveNodes;
                        best_train[l] = current_pop[equalFitness.at(indBest)];
                    }
                }
            }

            equalFitness.clear();
            prox = it;
        }

        for(int k = 0; k < NUM_EXECUTIONS; k++) {
            bestActiveNodes[k] = 9999;
        }

        for(int k = 0; k < NUM_EXECUTIONS; k++) {
            feasibles[k] = 0;
        }

        equalFitness.clear();
        indBest = -1;

        for(int j = 0; j < NUM_EXECUTIONS; j++) {
            best[j] = best_train[j];
        }


        timeManager.getEndTime(Iteracao_T);

        timeManager.getElapsedTime(Iteracao_T);

        iterations++;
    }

    (*timeIter) = timeManager.getTotalTime(Iteracao_T);

    return best;
}

Chromosome* PCGP(Dataset* training, Parameters* params, OCLConfig* ocl, int *seeds, double* timeIter, double* timeKernel, std::ofstream& factivel_file){
    GPTime timeManager(4);

    Chromosome* best = new Chromosome[NUM_EXECUTIONS];
    Chromosome* best_train = new Chromosome[NUM_EXECUTIONS];
    Chromosome* population = new Chromosome[NUM_INDIV];
    ActiveChromosome* activePopulation = new ActiveChromosome[NUM_INDIV];
    CompactChromosome *compactPopulation = new CompactChromosome[NUM_INDIV];

    initializePopulation(population, params, &seeds[0]);

    int bestTrain[NUM_EXECUTIONS];

    evaluatePopulation(population, training, 0, bestTrain);

    //unsigned short int bestActiveNodes[NUM_EXECUTIONS];
    unsigned int bestActiveNodes[NUM_EXECUTIONS];

    for(int k = 0; k < NUM_EXECUTIONS; k++) {
        bestActiveNodes[k] = 9999;
    }

    int feasibles[NUM_EXECUTIONS];
    for(int k = 0; k < NUM_EXECUTIONS; k++) {
        feasibles[k] = 0;
    }

    double kernelTime = 0;
    for(int k = 0; k < NUM_EXECUTIONS; k++) {
        best_train[k] = population[bestTrain[k]];
    }

    for(int k = 0; k < NUM_EXECUTIONS; k++) {
        best[k] = best_train[k];
    }

    ocl->writeReadOnlyBufers(params);

    int prox, it;
    long int iterations = 0;
    std::vector<int> equalFitness;
    int indBest = -1;

    while(stopCriteria(iterations)) {
        timeManager.getStartTime(Iteracao_T);

        int group = -1;
        for(int k = 0; k < NUM_INDIV; k++) {
            if(k % NUM_EXECUTIONS == 0) {
                group++;
            }
            population[k] = best[group];
            mutateSAM(&population[k], params, seeds);
        }



#if DEFAULT
        ocl->writePopulationBuffer(population);
        ocl->finishCommandQueue();

        ocl->enqueueTrainKernel();
        //ocl->enqueueValidationKernel();
#elif COMPACT
        ocl->compactChromosome(population, compactPopulation);
        ocl->writePopulationCompactBuffer(compactPopulation);
        ocl->finishCommandQueue();

        ocl->enqueueTrainCompactKernel();
        ocl->enqueueValidationCompactKernel();
#elif IMAGE_R
        ocl->writeImageBuffer(population);
        ocl->finishCommandQueue();

        ocl->enqueueEvaluationImageKernel();
        ocl->enqueueEvaluationImageValidationKernel();
#elif IMAGE_RG
        ocl->writeImageBufferHalf(population);
        ocl->finishCommandQueue();

        ocl->enqueueEvaluationImageHalfKernel();
        ocl->enqueueEvaluationImageValidationHalfKernel();
#elif IMAGE_RGBA
        ocl->writeImageBufferQuarter(population);
        ocl->finishCommandQueue();

        ocl->enqueueEvaluationImageQuarterKernel();
        ocl->enqueueEvaluationImageValidationQuarterKernel();
#elif COMPACT_R
        ocl->writeImageBufferCompact(population);
        ocl->finishCommandQueue();

        ocl->enqueueEvaluationImageCompactKernel();
        ocl->enqueueEvaluationImageValidationCompactKernel();

#elif COMPACT_RG
        ocl->writeImageBufferHalfCompact(population);
        ocl->finishCommandQueue();

        ocl->enqueueEvaluationImageHalfCompactKernel();
        ocl->enqueueEvaluationImageValidationHalfCompactKernel();
#elif  COMPACT_RGBA
        ocl->writeImageBufferQuarterCompact(population);
        ocl->finishCommandQueue();

        ocl->enqueueEvaluationImageQuarterCompactKernel();
        ocl->enqueueEvaluationImageValidationQuarterCompactKernel();
#endif

        ocl->finishCommandQueue();
        kernelTime+= ocl->getKernelElapsedTimeTrain();


        ocl->readFitnessBuffer();

        ocl->finishCommandQueue();

        it = 0;
        prox = 0;

        for(int l = 0; l < NUM_EXECUTIONS; l++) {
            for(it = prox; it < prox + NUM_INDIV_POP; it++) {
                population[it].fitness = ocl->fitness[it];
            }
            prox = it;
        }

        it = 0;
        prox = 0;

        for(int l = 0; l < NUM_EXECUTIONS; l++) {
            for(it = prox; it < prox + NUM_INDIV_POP; it++) {
                //std::cout << "APTIDAO: " << population[it].fitness << std::endl;
                if (population[it].fitness == (params->M * params->O)) {
                    feasibles[l] += 1;
                }
            }
            prox = it;
        }

        it = 0;
        prox = 0;

        for(int l = 0; l < NUM_EXECUTIONS; l++) {
            if(feasibles[l] > 0) {
                if(feasibles[l] == 1) {
                    for(it = prox; it < prox + NUM_INDIV_POP; it++) {
                        if(population[it].fitness == (params->M * params->O)){
                            best_train[l] = population[it];
                            bestActiveNodes[l] = population[it].numActiveNodes;
                        }
                    }
                } else if (feasibles[l] > 1) {
                    for(it = prox; it < prox + NUM_INDIV_POP; it++) {
                        if(population[it].fitness == (params->M * params->O)){
                            // resolver escolha aleatoria dps de comparação
                            // nós ativos
                            if(population[it].numActiveNodes < bestActiveNodes[l]) {
                                best_train[l] = population[it];
                                bestActiveNodes[l] = population[it].numActiveNodes;
                            }
                        }
                    }
                }
            } else {
                for(it = prox; it < prox + NUM_INDIV_POP; it++) {
                    if(population[it].fitness > best_train[l].fitness){
                        best_train[l] = population[it];
                        bestActiveNodes[l] = population[it].numActiveNodes;
                        equalFitness.clear();
                        equalFitness.push_back(it);
                    } else if (population[it].fitness == best_train[l].fitness) {
                        equalFitness.push_back(it);
                    }
                }

                if(!equalFitness.empty()) {
                    if(equalFitness.size() == 1) {
                        bestActiveNodes[l] = population[equalFitness.at(0)].numActiveNodes;
                        best_train[l] = population[equalFitness.at(0)];
                    } else {
                        indBest = rand() % (equalFitness.size() - 1);
                        bestActiveNodes[l] = population[equalFitness.at(indBest)].numActiveNodes;
                        best_train[l] = population[equalFitness.at(indBest)];
                    }
                }
            }

            equalFitness.clear();
            prox = it;
        }

        for(int k = 0; k < NUM_EXECUTIONS; k++) {
            bestActiveNodes[k] = 9999;
        }

        for(int k = 0; k < NUM_EXECUTIONS; k++) {
            feasibles[k] = 0;
        }


        for(int i  = 0; i < NUM_EXECUTIONS; i++) {
            best[i] = best_train[i];
        }

        equalFitness.clear();
        indBest = -1;

        timeManager.getEndTime(Iteracao_T);
        timeManager.getElapsedTime(Iteracao_T);

        if(iterations%100000 == 0){
            printf("Generation %d:\n", iterations);
            printf("Best 0 fitness: %f\n", best[0].fitness);
            printf("Best 1 fitness: %f\n", best[1].fitness);
            printf("Best 2 fitness: %f\n", best[2].fitness);
            printf("Best 3 fitness: %f\n", best[3].fitness);
            printf("Best 4 fitness: %f\n", best[4].fitness);
            printf("Time: %f\n", timeManager.getTotalTime(Iteracao_T));
            printf("Kernel Time: %f\n", kernelTime);
        }
        iterations++;

    }

    (*timeIter) = timeManager.getTotalTime(Iteracao_T);
    (*timeKernel) = kernelTime;

    //ocl->readSeedsBuffer(seeds);
    ocl->finishCommandQueue();
    /* for(int i = 0; i < NUM_INDIV * ocl->maxLocalSize; i++){
         std::cout << seeds[i] << " ";
     }*/
    std::cout << std::endl;


    //printChromosome(&best, params);
    return best;
}

void printChromosome(Chromosome *c, Parameters *p) {
    for(int i = 0; i < MAX_NODES; i++) {
        //std::cout << c->nodes[i].inputs[0] << " " << c->nodes[i].inputs[1] << " " << c->nodes[i].function << std::endl;
       if (c->nodes[i].active) {
           /*std::cout <<"---------------- comeco -----------------" << std::endl;
           std::cout <<"Indice do no " << i + p->N << std::endl;
           std::cout <<"No: " << std::endl;
           std::cout <<"Function: " << c->nodes[i].function << std::endl;
           std::cout <<"Inputs: " << c->nodes[i].inputs[0] << " "
           << c->nodes[i].inputs[1] << std::endl;
           std::cout <<"----------------   fim  -----------------" << std::endl;*/
           std::cout << "No" << i + p->N << " - " << c->nodes[i].inputs[0] << " " << c->nodes[i].inputs[1] << " " << c->nodes[i].function << std::endl;
       }
    }
    std::cout <<"Outputs: " << c->output[0] + p->N << std::endl;//
    //std::cout << "Output: " << c->output[0] << std::endl;
    std::cout << "- FIM -" << std::endl;
}

void printFile(Chromosome *c, Parameters *p, std::ofstream& factivel_file) {
    for(int i = 0; i < MAX_NODES; i++) {
        if (c->nodes[i].active) {
            factivel_file << "Node" << i + p->N << " " << c->nodes[i].inputs[0]
            << " " << c->nodes[i].inputs[1] << " " <<  c->nodes[i].function << "\n";
        }
    }
    factivel_file << "Output " << c->output[0] + p->N << "\n";
    factivel_file << "\n";
}

Chromosome CGPDE_IN();

Chromosome CGPDE_OUT();

Chromosome PCGPDE_IN();

Chromosome PCGPDE_OUT();
