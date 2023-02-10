//
// Created by bruno on 05/02/2020.
//

#ifndef PPCGPDE_STACK_H
#define PCGPDE_STACK_H

#include "constants.h"


typedef struct {
    int topIndex;
    unsigned int info[MAX_NODES * MAX_ARITY];
} Stack;

typedef struct {
    int topIndex;
    float info[MAX_NODES * MAX_ARITY];
} ExStack;

void push(Stack* s, unsigned int info);
unsigned int pop(Stack* s);

void pushEx(ExStack* s, float info);
float popEx(ExStack* s);

#endif //PCGPDE_STACK_H
