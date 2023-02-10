//
// Created by bruno on 05/02/2020.
//

#include "stack.h"

void push(Stack* s, unsigned int info){
    (s->topIndex)++;
    if(s->topIndex < MAX_NODES * MAX_ARITY){
        s->info[s->topIndex] = info;
    }
}


unsigned int pop(Stack* s){
    if(s->topIndex >= 0){
        (s->topIndex)--;
        return s->info[(s->topIndex) + 1];
    }
}

void pushEx(ExStack* s, float info) {
    (s->topIndex)++;
    if(s->topIndex < MAX_NODES * MAX_ARITY){
        s->info[s->topIndex] = info;
    }
}

float popEx(ExStack* s) {
    if(s->topIndex >= 0){
        (s->topIndex)--;
        return s->info[(s->topIndex) + 1];
    }
}