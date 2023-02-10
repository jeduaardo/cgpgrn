void activateNodes(__global Chromosome* c){

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
        unsigned int nodeIndex = c->output[i];
        push(&s, nodeIndex);

        while(s.topIndex != -1) {
            unsigned int node = pop(&s);
            if( c->nodes[node].active == 0) {
                for (j = 0; j < MAX_ARITY; j++) {
                    if (c->nodes[node].inputs[j] >= N) {
                        push(&s, c->nodes[node].inputs[j] - N);
                    }
                }
                c->activeNodes[c->numActiveNodes] = node;
                c->nodes[node].active = 1;
                c->numActiveNodes++;
            }
        }
    }
}

void mutateTopologyProbabilistic(__global Chromosome *c, __global unsigned int* functionSet, int *seed, int type) {

    int i, j;
    int local_id = get_local_id(0);
    
    for(i = 0; i < MAX_NODES; i++){
    //if(local_id < MAX_NODES) {

        if(randomProb(seed) <= PROB_MUT) {
            c->nodes[i].function = functionSet[randomFunction(seed)];
            c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
        }

        for(j = 0; j < c->nodes[i].maxInputs; j++) {
            if(randomProb(seed) <= PROB_MUT) {
                c->nodes[i].inputs[j] = randomInput(i, seed);
            }
            if(type == 0 && randomProb(seed) <= PROB_MUT){
                c->nodes[i].inputsWeight[j] = randomConnectionWeight(seed);
            }
        }
    //}
    }
   // barrier(CLK_LOCAL_MEM_FENCE);
    //if(local_id == 0)
    activateNodes(c);

}

void mutateTopologyProbabilistic2(__global Chromosome *c, __global unsigned int* functionSet, int *seeds, int type) {

    int i, j, k;
    int group_id = get_group_id(0);
    int local_id = get_local_id(0);
    
    int size = MAX_NODES * MAX_ARITY;

    //printf("%d\n", size);
    //for(i = 0; i < MAX_NODES; i++){
    //if(local_id < MAX_NODES) {

    for(k = 0; k < ceil( (size)/ (float)LOCAL_SIZE_EVOL ) ; k++){

        int index = k * LOCAL_SIZE_EVOL + local_id;
        
        if( index < size){
            int indexNode = index / MAX_ARITY;
            int indexArray = (index % MAX_ARITY);
            //printf("%d\n", indexArray);

            //for(j = 0; j < c->nodes[k * LOCAL_SIZE_EVOL + local_id].maxInputs; j++) {

            if(randomProb(seeds) <= PROB_MUT) {
                c->nodes[indexNode].inputs[indexArray] = randomInput(indexNode, seeds);
            }
            if(type == 0 && randomProb(seeds) <= PROB_MUT){
                c->nodes[indexNode].inputsWeight[indexArray] = randomConnectionWeight(seeds);
            }

            if(indexArray == 0 ) {
                if(randomProb(seeds) <= PROB_MUT ){
                    c->nodes[indexNode].function = functionSet[randomFunction(seeds)];
                    c->nodes[indexNode].maxInputs = getFunctionInputs(c->nodes[indexNode].function);
                }
            }

            //}
        }
    }

    //}
    //}
    barrier(CLK_GLOBAL_MEM_FENCE);
    //if(local_id == 0)
    //    activateNodes(c);

}

void mutateTopologyProbabilisticActive(__global Chromosome *c, __global unsigned int* functionSet, int *seed, int type) {

    int i, j;
    for(i = 0; i < MAX_NODES; i++){
        if(c->nodes[i].active == 1){
            if(randomProb(seed) <= PROB_MUT) {
                c->nodes[i].function = functionSet[randomFunction(seed)];
                c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
            }
            for(j = 0; j < c->nodes[i].maxInputs; j++) {
                if(randomProb(seed) <= PROB_MUT) {
                    c->nodes[i].inputs[j] = randomInput(i, seed);
                }
                if(type == 0 && randomProb(seed) <= PROB_MUT){
                    c->nodes[i].inputsWeight[j] = randomConnectionWeight(seed);
                }
            }
        }
    }

    activateNodes(c);
}

void mutateTopologyPoint(__global Chromosome *c, __global unsigned int* functionSet, int *seed) {
    int mutationComplete = -1;
    unsigned int newIndex;
    unsigned int newInputIndex;
    unsigned int newValue;

    int num_inputs = MAX_NODES * MAX_ARITY;
    while (mutationComplete == -1){
        unsigned int nodeIndex = randomInterval(0, MAX_NODES + (num_inputs) + MAX_OUTPUTS, seed); //Select any node or output
        if(nodeIndex < MAX_NODES) { // select function
            newIndex = nodeIndex;
            newValue = functionSet[randomFunction(seed)];
            if(newValue != c->nodes[newIndex].function){
                c->nodes[newIndex].function = newValue;
                c->nodes[newIndex].maxInputs = getFunctionInputs(newValue);
                if(c->nodes[newIndex].active > -1) {
                    mutationComplete = 1;
                }
            }
        } else if (nodeIndex <= MAX_NODES + (num_inputs)) { //select input
            newIndex = (unsigned int) ((nodeIndex - MAX_NODES) / MAX_ARITY);
            newInputIndex= (unsigned int) ((nodeIndex - MAX_NODES) % MAX_ARITY);

            newValue = randomInput(newIndex, seed);

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
    activateNodes(c);
}