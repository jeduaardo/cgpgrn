#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "float precision floating point not supported by OpenCL implementation."
#endif

#include "utils.cl"
#include "evol.cl"

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
                          CLK_ADDRESS_NONE |
                          CLK_FILTER_NEAREST; //Don't interpolate





float executeFunction(__global Chromosome* c, int node, ExStack* exStack){
    int i;
    float result, sum;
    unsigned int inputs = c->nodes[node].maxInputs;
    switch (c->nodes[node].function){
        #ifdef ADD
            case ADD:
                result = exStack->info[exStack->topIndex - inputs + 1];
                for(i = 1; i < inputs; i++){
                    result += exStack->info[exStack->topIndex - i + 1];
                }
                exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef SUB
        case SUB:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result -= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef MUL
        case MUL:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result *= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef DIV
        case DIV:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result /= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef ABS
        case ABS:
            result = fabs(popEx(exStack));
        break;
        #endif

        #ifdef SQRT
        case SQRT:
            result = sqrt(popEx(exStack));
        break;
        #endif
        
        #ifdef SQ
        case SQ:
            result = pow((float)popEx(exStack), (float)2);
        break;
        #endif
        
        #ifdef CUBE
        case CUBE:
            result = pow((float)popEx(exStack), (float)3);
            break;
        #endif
        
        #ifdef POW
        case POW:
            result = popEx(exStack);
            result = pow((float)popEx(exStack), (float)result);
            break;
        #endif
        
        #ifdef AND
        case AND:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 0){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef OR

        case OR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result = 1;
                }
            }
        break;
        #endif
        
        #ifdef XOR

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
        break;
        #endif
        
        #ifdef NAND
        

        case NAND:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 0){
                    result = 1;
                }
            }
        break;
        #endif

        #ifdef NOT
            case NOT:

                for(i = 0; i < 1; i++){
                    if(popExLinear(exStack) == 1){
                        result = 0;
                    } else {
                      result = 1;
                    }
                }
            break;
        #endif
        
        #ifdef NOR

        case NOR:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popEx(exStack) == 1){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef XNOR

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
        break;
        #endif
        
        #ifdef EXP
        case EXP:
            result = exp(popEx(exStack));
            break;
        #endif
        
        #ifdef SIN
        case SIN:
            result = sin(popEx(exStack));
            break;
        #endif
        
        #ifdef COS

        case COS:
            result = cos(popEx(exStack));
            break;
        #endif
        
        #ifdef TAN

        case TAN:
            result = tan(popEx(exStack));
            break;
        #endif
        
        #ifdef ONE

        case ONE:
            result = 1;
            break;
        #endif
        
        #ifdef ZERO

        case ZERO:
            result = 0;
            break;
        #endif
        
        #ifdef PI

        case PI:
            result = CONST_PI;
            break;
        #endif
        
        #ifdef WIRE

        case WIRE:
            result = popEx(exStack);
            break;
        #endif
        
        #ifdef SIG

        case SIG:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = 1.0f / (1 + exp(-sum));
            break;
        #endif
        
        #ifdef GAUSS

        case GAUSS:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = exp(-(pow((float) (sum - 0), (float) 2)) / (2 * pow((float)1, (float)2)));
            break;
        #endif
        
        #ifdef STEP

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
        #endif
        
        #ifdef SOFTSIGN

        case SOFTSIGN:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = sum / (1 + fabs(sum));
            break;
        #endif
        
        #ifdef TANH

        case TANH:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = tanh(sum);
            break;
        #endif
        
        default:
            break;
    }
    return result;
}

float executeFunctionLinear(__global Chromosome* c, int node, ExStackLinear* exStack){
    int i;
    float result, sum;
    unsigned int inputs = c->nodes[node].maxInputs;
    switch (c->nodes[node].function){
        #ifdef ADD
            case ADD:
                result = exStack->info[exStack->topIndex - inputs + 1];
                for(i = 1; i < inputs; i++){
                    result += exStack->info[exStack->topIndex - i + 1];
                }
                exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef SUB
        case SUB:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result -= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef MUL
        case MUL:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result *= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef DIV
        case DIV:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result /= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef ABS
        case ABS:
            result = fabs(popExLinear(exStack));
        break;
        #endif

        #ifdef SQRT
        case SQRT:
            result = sqrt(popExLinear(exStack));
        break;
        #endif
        
        #ifdef SQ
        case SQ:
            result = pow((float)popExLinear(exStack), (float)2);
        break;
        #endif
        
        #ifdef CUBE
        case CUBE:
            result = pow((float)popExLinear(exStack), (float)3);
            break;
        #endif
        
        #ifdef POW
        case POW:
            result = popExLinear(exStack);
            result = pow((float)popExLinear(exStack), (float)result);
            break;
        #endif
        
        #ifdef AND
        case AND:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 0){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef OR

        case OR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 1){
                    result = 1;
                }
            }
        break;
        #endif

        #ifdef NOT
            case NOT:

                for(i = 0; i < 1; i++){
                    if(popExLinear(exStack) == 1){
                        result = 0;
                    } else {
                      result = 1;
                    }
                }
            break;
        #endif
        
        #ifdef XOR

        case XOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 1){
                    result += 1;
                }
            }
            if(result != 1){
                result = 0;
            }
        break;
        #endif
        
        #ifdef NAND
        

        case NAND:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 0){
                    result = 1;
                }
            }
        break;
        #endif
        
        #ifdef NOR

        case NOR:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 1){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef XNOR

        case XNOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 1){
                    result += 1;
                }
            }
            if(result == 1){
                result = 0;
            } else {
                result = 1;
            }
        break;
        #endif
        
        #ifdef EXP
        case EXP:
            result = exp(popExLinear(exStack));
            break;
        #endif
        
        #ifdef SIN
        case SIN:
            result = sin(popExLinear(exStack));
            break;
        #endif
        
        #ifdef COS

        case COS:
            result = cos(popExLinear(exStack));
            break;
        #endif
        
        #ifdef TAN

        case TAN:
            result = tan(popExLinear(exStack));
            break;
        #endif
        
        #ifdef ONE

        case ONE:
            result = 1;
            break;
        #endif
        
        #ifdef ZERO

        case ZERO:
            result = 0;
            break;
        #endif
        
        #ifdef PI

        case PI:
            result = CONST_PI;
            break;
        #endif
        
        #ifdef WIRE

        case WIRE:
            result = popExLinear(exStack);
            break;
        #endif
        
        #ifdef SIG

        case SIG:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = 1.0f / (1.0f + exp(-sum));
            break;
        #endif
        
        #ifdef GAUSS

        case GAUSS:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = exp(-(pow((float) (sum - 0), (float) 2)) / (2 * pow((float)1, (float)2)));
            break;
        #endif
        
        #ifdef STEP

        case STEP:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            if(sum < 0) {
                result = 0;
            } else {
                result = 1;
            }
           break;
        #endif
        
        #ifdef SOFTSIGN

        case SOFTSIGN:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = sum / (1 + fabs(sum));
            break;
        #endif
        
        #ifdef TANH

        case TANH:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = tanh(sum);
            break;
        #endif
        
        default:
            break;
    }
    return result;
}

float executeFunctionLinearCompact(__global CompactChromosome* c, int node, ExStackLinear* exStack){
    int i;
    float result, sum;
    unsigned int aux;
    unsigned int inputs = MAX_ARITY;//c->nodes[node].maxInputs;
    //union IntFloat int_float;
    //printf("\n%d", c->nodes[node].function_inputs_active );
    switch (c->nodes[node].function_inputs_active >> 17){

        
        #ifdef SIG

        case SIG:
            sum = 0;
            for(i = 0; i < inputs; i++){
                //aux = c->nodes[node].inputsWeight[i] & (0xFFFF0000);
                //printf("\n%f", (*(float*)(&aux)));

                //sum += (popEx(exStack) * (*(float*)(&aux)));
                
               // aux = ( c->nodes[node].inputsWeight[i] << 16) ;
               //printf("\n%f", int_float.f);

                //sum += (popEx(exStack) * (*(float*)(&aux)));
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = 1.0f / (1.0f + exp(-sum));
            break;
        #endif
        
        
        default:
            break;
    }
    return result;
}

float executeFunctionLinearActive(__global ActiveChromosome* c, int node, ExStackLinear* exStack){
    int i;
    float result, sum;
    unsigned int inputs = c->nodes[node].maxInputs;
    switch (c->nodes[node].function){
        #ifdef ADD
            case ADD:
                result = exStack->info[exStack->topIndex - inputs + 1];
                for(i = 1; i < inputs; i++){
                    result += exStack->info[exStack->topIndex - i + 1];
                }
                exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef SUB
        case SUB:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result -= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef MUL
        case MUL:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result *= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef DIV
        case DIV:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result /= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef ABS
        case ABS:
            result = fabs(popExLinear(exStack));
        break;
        #endif

        #ifdef SQRT
        case SQRT:
            result = sqrt(popExLinear(exStack));
        break;
        #endif
        
        #ifdef SQ
        case SQ:
            result = pow((float)popExLinear(exStack), (float)2);
        break;
        #endif
        
        #ifdef CUBE
        case CUBE:
            result = pow((float)popExLinear(exStack), (float)3);
            break;
        #endif
        
        #ifdef POW
        case POW:
            result = popExLinear(exStack);
            result = pow((float)popExLinear(exStack), (float)result);
            break;
        #endif
        
        #ifdef AND
        case AND:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 0){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef OR

        case OR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 1){
                    result = 1;
                }
            }
        break;
        #endif
        
        #ifdef XOR

        case XOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 1){
                    result += 1;
                }
            }
            if(result != 1){
                result = 0;
            }
        break;
        #endif
        
        #ifdef NAND
        

        case NAND:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 0){
                    result = 1;
                }
            }
        break;
        #endif
        
        #ifdef NOR

        case NOR:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 1){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef XNOR

        case XNOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 1){
                    result += 1;
                }
            }
            if(result == 1){
                result = 0;
            } else {
                result = 1;
            }
        break;
        #endif
        
        #ifdef EXP
        case EXP:
            result = exp(popExLinear(exStack));
            break;
        #endif
        
        #ifdef SIN
        case SIN:
            result = sin(popExLinear(exStack));
            break;
        #endif
        
        #ifdef COS

        case COS:
            result = cos(popExLinear(exStack));
            break;
        #endif
        
        #ifdef TAN

        case TAN:
            result = tan(popExLinear(exStack));
            break;
        #endif
        
        #ifdef ONE

        case ONE:
            result = 1;
            break;
        #endif
        
        #ifdef ZERO

        case ZERO:
            result = 0;
            break;
        #endif
        
        #ifdef PI

        case PI:
            result = CONST_PI;
            break;
        #endif
        
        #ifdef WIRE

        case WIRE:
            result = popExLinear(exStack);
            break;
        #endif
        
        #ifdef SIG

        case SIG:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = 1.0f / (1 + exp(-sum));
            break;
        #endif
        
        #ifdef GAUSS

        case GAUSS:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = exp(-(pow((float) (sum - 0), (float) 2)) / (2 * pow((float)1, (float)2)));
            break;
        #endif
        
        #ifdef STEP

        case STEP:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            if(sum < 0) {
                result = 0;
            } else {
                result = 1;
            }
           break;
        #endif
        
        #ifdef SOFTSIGN

        case SOFTSIGN:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = sum / (1 + fabs(sum));
            break;
        #endif
        
        #ifdef TANH

        case TANH:
            sum = 0;
            for(i = 0; i < inputs; i++){
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = tanh(sum);
            break;
        #endif
        
        default:
            break;
    }
    return result;
}

float executeFunctionLinearActiveImage(__read_only image2d_array_t c, int node, ExStackLinear* exStack){
    int i;
    float result, sum;
    int group_id = get_group_id(0);
    
    uint4 pixelInt;
    float4 pixelFloat;

    pixelInt = read_imageui(c, sampler, (int4)(0,node+1,group_id,0));

    unsigned int inputs = MAX_ARITY;//c->nodes[node].maxInputs;
    switch (pixelInt.x){
        #ifdef ADD
            case ADD:
                result = exStack->info[exStack->topIndex - inputs + 1];
                for(i = 1; i < inputs; i++){
                    result += exStack->info[exStack->topIndex - i + 1];
                }
                exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef SUB
        case SUB:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result -= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef MUL
        case MUL:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result *= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef DIV
        case DIV:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result /= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef ABS
        case ABS:
            result = fabs(popExLinear(exStack));
        break;
        #endif

        #ifdef SQRT
        case SQRT:
            result = sqrt(popExLinear(exStack));
        break;
        #endif
        
        #ifdef SQ
        case SQ:
            result = pow((float)popExLinear(exStack), (float)2);
        break;
        #endif
        
        #ifdef CUBE
        case CUBE:
            result = pow((float)popExLinear(exStack), (float)3);
            break;
        #endif
        
        #ifdef POW
        case POW:
            result = popExLinear(exStack);
            result = pow((float)popExLinear(exStack), (float)result);
            break;
        #endif
        
        #ifdef AND
        case AND:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 0){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef OR

        case OR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 1){
                    result = 1;
                }
            }
        break;
        #endif
        
        #ifdef XOR

        case XOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 1){
                    result += 1;
                }
            }
            if(result != 1){
                result = 0;
            }
        break;
        #endif
        
        #ifdef NAND
        

        case NAND:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 0){
                    result = 1;
                }
            }
        break;
        #endif
        
        #ifdef NOR

        case NOR:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 1){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef XNOR

        case XNOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 1){
                    result += 1;
                }
            }
            if(result == 1){
                result = 0;
            } else {
                result = 1;
            }
        break;
        #endif
        
        #ifdef EXP
        case EXP:
            result = exp(popExLinear(exStack));
            break;
        #endif
        
        #ifdef SIN
        case SIN:
            result = sin(popExLinear(exStack));
            break;
        #endif
        
        #ifdef COS

        case COS:
            result = cos(popExLinear(exStack));
            break;
        #endif
        
        #ifdef TAN

        case TAN:
            result = tan(popExLinear(exStack));
            break;
        #endif
        
        #ifdef ONE

        case ONE:
            result = 1;
            break;
        #endif
        
        #ifdef ZERO

        case ZERO:
            result = 0;
            break;
        #endif
        
        #ifdef PI

        case PI:
            result = CONST_PI;
            break;
        #endif
        
        #ifdef WIRE

        case WIRE:
            result = popExLinear(exStack);
            break;
        #endif
        
        #ifdef SIG

        case SIG:
            sum = 0;
            for(i = 0; i < inputs; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));

                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            result = 1.0f / (1 + exp(-sum));
            break;
        #endif
        
        #ifdef GAUSS

        case GAUSS:
            sum = 0;
            for(i = 0; i < inputs; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            result = exp(-(pow((float) (sum - 0), (float) 2)) / (2 * pow((float)1, (float)2)));
            break;
        #endif
        
        #ifdef STEP

        case STEP:
            sum = 0;
            for(i = 0; i < inputs; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            if(sum < 0) {
                result = 0;
            } else {
                result = 1;
            }
           break;
        #endif
        
        #ifdef SOFTSIGN

        case SOFTSIGN:
            sum = 0;
            for(i = 0; i < inputs; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            result = sum / (1 + fabs(sum));
            break;
        #endif
        
        #ifdef TANH

        case TANH:
            sum = 0;
            for(i = 0; i < inputs; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            result = tanh(sum);
            break;
        #endif
        
        default:
            break;
    }
    return result;
}

float executeFunctionLinearActiveImageHalf(__read_only image2d_array_t c, int node, ExStackLinear* exStack){
    int i;
    float result, sum;
    int group_id = get_group_id(0);
    int local_id = get_group_id(0);
    
    uint4 pixelInt;
    float4 pixelFloat;

    pixelInt = read_imageui(c, sampler, (int4)(0,node+1,group_id,0));

    //if(group_id == 0 && local_id == 0)
    //    printf("\n%d %d \n", pixelInt.x, pixelInt.y);

    unsigned int inputs = MAX_ARITY;//c->nodes[node].maxInputs;
    switch (pixelInt.x){
        #ifdef ADD
            case ADD:
                result = exStack->info[exStack->topIndex - inputs + 1];
                for(i = 1; i < inputs; i++){
                    result += exStack->info[exStack->topIndex - i + 1];
                }
                exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef SUB
        case SUB:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result -= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef MUL
        case MUL:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result *= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef DIV
        case DIV:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for(i = 1; i < inputs; i++){
                result /= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
        break;
        #endif

        #ifdef ABS
        case ABS:
            result = fabs(popExLinear(exStack));
        break;
        #endif

        #ifdef SQRT
        case SQRT:
            result = sqrt(popExLinear(exStack));
        break;
        #endif
        
        #ifdef SQ
        case SQ:
            result = pow((float)popExLinear(exStack), (float)2);
        break;
        #endif
        
        #ifdef CUBE
        case CUBE:
            result = pow((float)popExLinear(exStack), (float)3);
            break;
        #endif
        
        #ifdef POW
        case POW:
            result = popExLinear(exStack);
            result = pow((float)popExLinear(exStack), (float)result);
            break;
        #endif
        
        #ifdef AND
        case AND:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 0){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef OR

        case OR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 1){
                    result = 1;
                }
            }
        break;
        #endif
        
        #ifdef XOR

        case XOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 1){
                    result += 1;
                }
            }
            if(result != 1){
                result = 0;
            }
        break;
        #endif
        
        #ifdef NAND
        

        case NAND:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 0){
                    result = 1;
                }
            }
        break;
        #endif
        
        #ifdef NOR

        case NOR:
            result = 1;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 1){
                    result = 0;
                }
            }
        break;
        #endif
        
        #ifdef XNOR

        case XNOR:
            result = 0;
            for(i = 0; i < inputs; i++){
                if(popExLinear(exStack) == 1){
                    result += 1;
                }
            }
            if(result == 1){
                result = 0;
            } else {
                result = 1;
            }
        break;
        #endif
        
        #ifdef EXP
        case EXP:
            result = exp(popExLinear(exStack));
            break;
        #endif
        
        #ifdef SIN
        case SIN:
            result = sin(popExLinear(exStack));
            break;
        #endif
        
        #ifdef COS

        case COS:
            result = cos(popExLinear(exStack));
            break;
        #endif
        
        #ifdef TAN

        case TAN:
            result = tan(popExLinear(exStack));
            break;
        #endif
        
        #ifdef ONE

        case ONE:
            result = 1;
            break;
        #endif
        
        #ifdef ZERO

        case ZERO:
            result = 0;
            break;
        #endif
        
        #ifdef PI

        case PI:
            result = CONST_PI;
            break;
        #endif
        
        #ifdef WIRE

        case WIRE:
            result = popExLinear(exStack);
            break;
        #endif
        
        #ifdef SIG

        case SIG:
            sum = 0;
            for(i = 0; i < inputs/2; i++){
                
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popExLinear(exStack) *  pixelFloat.x);
                sum += (popExLinear(exStack) *  pixelFloat.y);
                
               // pixelFloat = read_imageui(c, sampler, (int4)(i+1,node+1,group_id,0));
                //unsigned int aux = pixelFloat.x << 16;
                //printf("%f ", (*(float*)&aux));
                //sum += (popExLinear(exStack) *  (*(float*)&aux));
               // aux = pixelFloat.y << 16;
                //sum += (popExLinear(exStack) *  (*(float*)&aux));
            }
            result = 1.0f / (1 + exp(-sum));
            break;
        #endif
        /*
        #ifdef GAUSS

        case GAUSS:
            sum = 0;
            for(i = 0; i < inputs/2; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
            }
            result = exp(-(pow((float) (sum - 0), (float) 2)) / (2 * pow((float)1, (float)2)));
            break;
        #endif
        
        #ifdef STEP

        case STEP:
            sum = 0;
            for(i = 0; i < inputs/2; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
            }
            if(sum < 0) {
                result = 0;
            } else {
                result = 1;
            }
           break;
        #endif
        
        #ifdef SOFTSIGN

        case SOFTSIGN:
            sum = 0;
            for(i = 0; i < inputs/2; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
            }
            result = sum / (1 + fabs(sum));
            break;
        #endif
        
        #ifdef TANH

        case TANH:
            sum = 0;
            for(i = 0; i < inputs/2; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
            }
            result = tanh(sum);
            break;
        #endif
        */
        default:
            break;
    }
    return result;
}

float executeFunctionLinearActiveImageQuarter(__read_only image2d_array_t c, int node, ExStackLinear* exStack){
    int i;
    float result, sum;
    int group_id = get_group_id(0);
    int local_id = get_group_id(0);
    
    uint4 pixelInt;
    float4 pixelFloat;

    pixelInt = read_imageui(c, sampler, (int4)(0,node+1,group_id,0));

    //if(group_id == 0 && local_id == 0)
    //    printf("\n%d %d \n", pixelInt.x, pixelInt.y);

    unsigned int inputs = MAX_ARITY;//c->nodes[node].maxInputs;
    switch (pixelInt.x){

        #ifdef ONE

        case ONE:
            result = 1;
            break;
        #endif
        
        #ifdef ZERO

        case ZERO:
            result = 0;
            break;
        #endif
        
        #ifdef PI

        case PI:
            result = CONST_PI;
            break;
        #endif
        
        #ifdef WIRE

        case WIRE:
            result = popExLinear(exStack);
            break;
        #endif
        
        #ifdef SIG

        case SIG:
            sum = 0;
            for(i = 0; i < inputs/4; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popExLinear(exStack) *  pixelFloat.x);
                sum += (popExLinear(exStack) *  pixelFloat.y);
                sum += (popExLinear(exStack) *  pixelFloat.z);
                sum += (popExLinear(exStack) *  pixelFloat.w);
                //unsigned int aux = pixelFloat.x << 16;
                //printf("%f ", (*(float*)&aux));
                //sum += (popExLinear(exStack) *  (*(float*)&aux));
                //aux = pixelFloat.y << 16;
                //sum += (popExLinear(exStack) *  (*(float*)&aux));
            }
            result = 1.0f / (1 + exp(-sum));
            break;
        #endif
        /*
        #ifdef GAUSS

        case GAUSS:
            sum = 0;
            for(i = 0; i < inputs/2; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
            }
            result = exp(-(pow((float) (sum - 0), (float) 2)) / (2 * pow((float)1, (float)2)));
            break;
        #endif
        
        #ifdef STEP

        case STEP:
            sum = 0;
            for(i = 0; i < inputs/2; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
            }
            if(sum < 0) {
                result = 0;
            } else {
                result = 1;
            }
           break;
        #endif
        
        #ifdef SOFTSIGN

        case SOFTSIGN:
            sum = 0;
            for(i = 0; i < inputs/2; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
            }
            result = sum / (1 + fabs(sum));
            break;
        #endif
        
        #ifdef TANH

        case TANH:
            sum = 0;
            for(i = 0; i < inputs/2; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+1,node+1,group_id,0));
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
            }
            result = tanh(sum);
            break;
        #endif
        */
        default:
            break;
    }
    return result;
}

float executeFunctionLinearActiveImageCompact(__read_only image2d_array_t c, int node, ExStackLinear* exStack){
    int i;
    float result, sum;
    int group_id = get_group_id(0);
    
    uint4 pixelInt;
    float4 pixelFloat;

    pixelInt = read_imageui(c, sampler, (int4)(1,node+1,group_id,0));

    unsigned int inputs = MAX_ARITY;//c->nodes[node].maxInputs;
    switch (pixelInt.x){

    
        #ifdef WIRE

        case WIRE:
            result = popExLinear(exStack);
            break;
        #endif
        
        #ifdef SIG

        case SIG:
            sum = 0;
            for(i = 0; i < inputs; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+2,node+1,group_id,0));

                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            result = 1.0f / (1 + exp(-sum));
            break;
        #endif
        
        #ifdef GAUSS

        case GAUSS:
            sum = 0;
            for(i = 0; i < inputs; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+2,node+1,group_id,0));
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            result = exp(-(pow((float) (sum - 0), (float) 2)) / (2 * pow((float)1, (float)2)));
            break;
        #endif
        
        #ifdef STEP

        case STEP:
            sum = 0;
            for(i = 0; i < inputs; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+2,node+1,group_id,0));
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            if(sum < 0) {
                result = 0;
            } else {
                result = 1;
            }
           break;
        #endif
        
        #ifdef SOFTSIGN

        case SOFTSIGN:
            sum = 0;
            for(i = 0; i < inputs; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+2,node+1,group_id,0));
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            result = sum / (1 + fabs(sum));
            break;
        #endif
        
        #ifdef TANH

        case TANH:
            sum = 0;
            for(i = 0; i < inputs; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+2,node+1,group_id,0));
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            result = tanh(sum);
            break;
        #endif
        
        default:
            break;
    }
    return result;
}


/**PADRÃO*/

void evaluateCircuit(__global Chromosome* c,
                    __global float* data, 
                    __global float* out, 
                    __local float* error,
                    __global float* fitness) {

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M/LOCAL_SIZE) ; k++){

    #else
        for(k = 0; k < ceil( M/ (float)LOCAL_SIZE ) ; k++){
            
            if( k * LOCAL_SIZE + local_id < M){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;

            for(i = 0; i < c->numActiveNodes; i++){
                currentActive = c->activeNodes[i];
                activeInputs = c->nodes[currentActive].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    if (c->nodes[currentActive].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado

                        pushExLinear(&exStack, alreadyEvaluated[c->nodes[currentActive].inputs[j] - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + ( M * c->nodes[currentActive].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinear(c, currentActive, &exStack);

        /*
                if (isnan(alreadyEvaluated[currentActive]) != 0) {
                    alreadyEvaluated[currentActive] = 0;
                }
                else if (isinf(alreadyEvaluated[currentActive]) != 0 ) {

                    if (alreadyEvaluated[currentActive] > 0) {
                        alreadyEvaluated[currentActive] = FLT_MAX;
                    }
                    else {
                        alreadyEvaluated[currentActive] = FLT_MIN;
                    }
                }
    */
            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                //unsigned int nodeIndex = c->output[i];
    
                /*if(alreadyEvaluated[c->output[i]] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[c->output[i]];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE + local_id + (M*i)] == 1.0)? i : correctClass;*/

                if(out[k*LOCAL_SIZE + local_id + (M*i)] == alreadyEvaluated[c->output[i]]) {
                  erro += 1;
                }
            }

            /* erro += (predictedClass == correctClass)? 1.0 : 0.0; */
            /*erro = 1;*/
            
        
        #ifdef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){
        /* fitness[group_id] = error[0] / M; */
        fitness[group_id] = error[0];
    }
}

void evaluateCircuitTrainValidation(__global Chromosome* c,
                                                __global float* data, 
                                                __global float* out, 
                                                __local float* error,
                                                __global float* fitness) {
                
    
    
    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE_GLOBAL
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_VALIDATION/LOCAL_SIZE) ; k++){

    #else
        for(k = 0; k < ceil( M_VALIDATION/ (float)LOCAL_SIZE ) ; k++){
            
            if( k * LOCAL_SIZE + local_id < M_VALIDATION){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;


            ExStackLinear exStack;
            exStack.topIndex = -1;

            for(i = 0; i < c->numActiveNodes; i++){
                currentActive = c->activeNodes[i];
                activeInputs = c->nodes[currentActive].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    if (c->nodes[currentActive].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[c->nodes[currentActive].inputs[j] - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + ( M_VALIDATION * c->nodes[currentActive].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinear(c, currentActive, &exStack);

        /*
                if (isnan(alreadyEvaluated[currentActive]) != 0) {
                    alreadyEvaluated[currentActive] = 0;
                }
                else if (isinf(alreadyEvaluated[currentActive]) != 0 ) {

                    if (alreadyEvaluated[currentActive] > 0) {
                        alreadyEvaluated[currentActive] = FLT_MAX;
                    }
                    else {
                        alreadyEvaluated[currentActive] = FLT_MIN;
                    }
                }
    */
            }

                for( i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];
            
                
                if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE + local_id + (M_VALIDATION*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;
            
            
        
        #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE_GLOBAL
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){
        fitness[group_id] = error[0] / M_VALIDATION;
    }
}


void evaluateCircuitTest(__global Chromosome* c,
                                    __global float* data, 
                                    __global float* out, 
                                    __local float* error,
                                    __global float* fitness) {
    
    

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_TEST_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_TEST/LOCAL_SIZE_TEST) ; k++){

    #else
        for(k = 0; k < ceil( M_TEST/ (float)LOCAL_SIZE_TEST ) ; k++){
            
            if( k * LOCAL_SIZE_TEST + local_id < M_TEST){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;

            for(i = 0; i < c->numActiveNodes; i++){
                currentActive = c->activeNodes[i];
                activeInputs = c->nodes[currentActive].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    if (c->nodes[currentActive].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        
                        pushExLinear(&exStack, alreadyEvaluated[c->nodes[currentActive].inputs[j] - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TEST + local_id + ( M_TEST * c->nodes[currentActive].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinear(c, currentActive, &exStack);

            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];
    
                if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_TEST + local_id + (M_TEST*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_TEST_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_TEST_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_TEST_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_TEST) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){
        fitness[group_id] = error[0] / M_TEST;
    }
}

void evaluateCircuitTrain(__global Chromosome* c,
                                    __global float* data,
                                    __global float* out,
                                    __local float* error,
                                    __global float* fitness) {



    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_TRAIN/LOCAL_SIZE_TRAIN) ; k++){

    #else
        for(k = 0; k < ceil( M_TRAIN/ (float)LOCAL_SIZE_TRAIN ) ; k++){

            if( k * LOCAL_SIZE_TRAIN + local_id < M_TRAIN){
    #endif
        //printf("c");
        //int i, j;
        maxPredicted = -FLT_MAX ;
        predictedClass = 0;
        correctClass = 0;

        ExStackLinear exStack;
        exStack.topIndex = -1;


        for(i = 0; i < c->numActiveNodes; i++){
                currentActive = c->activeNodes[i];
                activeInputs = c->nodes[currentActive].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    if (c->nodes[currentActive].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado

                        pushExLinear(&exStack, alreadyEvaluated[c->nodes[currentActive].inputs[j] - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * c->nodes[currentActive].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinear(c, currentActive, &exStack);

            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];

                /* if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                } */

                /* correctClass = (out[k*LOCAL_SIZE_TRAIN + local_id + (M_TRAIN*i)] == 1.0)? i : correctClass; */

                if(out[k*LOCAL_SIZE_TRAIN + local_id + (M_TRAIN*i)] == alreadyEvaluated[c->output[i]]) {
                   erro += 1.0;
                }
            }

            /*erro += (predictedClass == correctClass)? 1.0 : 0.0;*/

        #ifdef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_TRAIN_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_TRAIN_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_TRAIN) )
    #endif
           error[local_id] += error[local_id + i];
    }

    if(local_id == 0){
        /* fitness[group_id] = error[0] / M_TRAIN; */
        fitness[group_id] = error[0];
    }
}

void evaluateCircuitValidation(__global Chromosome* c,
                                            __global float* data, 
                                            __global float* out, 
                                            __local float* error,
                                            __global float* fitness) {

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_VALIDATION/LOCAL_SIZE_VALIDATION) ; k++){

    #else
        for(k = 0; k < ceil( M_VALIDATION/ (float)LOCAL_SIZE_VALIDATION ) ; k++){
            
            if( k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;

            for(i = 0; i < c->numActiveNodes; i++){
                currentActive = c->activeNodes[i];
                activeInputs = c->nodes[currentActive].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    if (c->nodes[currentActive].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        //unsigned int refIndex = c->nodes[currentActive].inputs[j] - N;

                        
                        pushExLinear(&exStack, alreadyEvaluated[c->nodes[currentActive].inputs[j] - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * c->nodes[currentActive].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinear(c, currentActive, &exStack);

            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];
    
                if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){
        fitness[group_id] = error[0] / M_VALIDATION;
    }
}
/**PADRÃO */


/**ACTIVE */
void evaluateCircuitActive(__global ActiveChromosome* c,
                                        __global float* data, 
                                        __global float* out, 
                                        __local float* error,
                                        __global float* fitness) {
    
    
    
    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M/LOCAL_SIZE) ; k++){

    #else
        for(k = 0; k < ceil( M/ (float)LOCAL_SIZE ) ; k++){
            
            if( k * LOCAL_SIZE + local_id < M){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

    //ExStackLinear
            ExStackLinear exStack;
            exStack.topIndex = -1;

            for(i = 0; i < c->numActiveNodes; i++){
                currentActive = c->nodes[i].originalIndex;
                activeInputs = c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    if (c->nodes[i].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado

                        pushExLinear(&exStack, alreadyEvaluated[c->nodes[i].inputs[j] - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + ( M * c->nodes[i].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActive(c, i, &exStack);

        /*
                if (isnan(alreadyEvaluated[currentActive]) != 0) {
                    alreadyEvaluated[currentActive] = 0;
                }
                else if (isinf(alreadyEvaluated[currentActive]) != 0 ) {

                    if (alreadyEvaluated[currentActive] > 0) {
                        alreadyEvaluated[currentActive] = FLT_MAX;
                    }
                    else {
                        alreadyEvaluated[currentActive] = FLT_MIN;
                    }
                }
    */
            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];
    
                if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE + local_id + (M*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){

        fitness[group_id] = error[0] / M;

    }
}

void evaluateCircuitTrainValidationActive(__global ActiveChromosome* c,
                                                        __global float* data, 
                                                        __global float* out, 
                                                        __local float* error,
                                                        __global float* fitnessValidation) {
                
    
    
    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE_GLOBAL
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_VALIDATION/LOCAL_SIZE) ; k++){

    #else
        for(k = 0; k < ceil( M_VALIDATION/ (float)LOCAL_SIZE ) ; k++){
            
            if( k * LOCAL_SIZE + local_id < M_VALIDATION){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;

            for(i = 0; i < c->numActiveNodes; i++){
                currentActive = c->nodes[i].originalIndex;
                activeInputs = c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    if (c->nodes[i].inputs[j] >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[c->nodes[i].inputs[j] - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + ( M_VALIDATION * c->nodes[i].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActive(c, i, &exStack);

        /*
                if (isnan(alreadyEvaluated[currentActive]) != 0) {
                    alreadyEvaluated[currentActive] = 0;
                }
                else if (isinf(alreadyEvaluated[currentActive]) != 0 ) {

                    if (alreadyEvaluated[currentActive] > 0) {
                        alreadyEvaluated[currentActive] = FLT_MAX;
                    }
                    else {
                        alreadyEvaluated[currentActive] = FLT_MIN;
                    }
                }
    */
            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];
            
                
                if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE + local_id + (M_VALIDATION*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;
            
            
        
        #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE_GLOBAL
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){
        fitnessValidation[group_id] = error[0] / M_VALIDATION;
    }
}
/**ACTIVE */



/**COMPACT */
void evaluateCircuitTrainCompact(__global CompactChromosome* c,
                                                __global float* data, 
                                                __global float* out, 
                                                __local float* error,
                                                __global float* fitness) {
    
    

    int i, k, j = 0;
    unsigned int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_TRAIN/LOCAL_SIZE_TRAIN) ; k++){

    #else
        for(k = 0; k < ceil( M_TRAIN/ (float)LOCAL_SIZE_TRAIN ) ; k++){
            
            if( k * LOCAL_SIZE_TRAIN + local_id < M_TRAIN){
    #endif
        //printf("c");
        //int i, j;
        maxPredicted = -FLT_MAX ;
        predictedClass = 0;
        correctClass = 0;
        /*
        for(i = 0; i < MAX_NODES; i++){
            alreadyEvaluated[i] = -FLT_MAX ;
        }
        */
        ExStackLinear exStack;
        exStack.topIndex = -1;

        
        for(i = 0; i < ceil(c->numActiveNodes/(float)2); i++){
                //unsigned int active0 = c->activeNodes[i] >> 16;
                //unsigned int active1 = c->activeNodes[i] & (0xFFFF);

                currentActive0 = c->activeNodes[i] >> 16;
                currentActive1 = c->activeNodes[i] & (0xFFFF);
                activeInputs = MAX_ARITY;// c->nodes[currentActive].function_inputs_active >>;
                
                

                for(j = 0; j < activeInputs/2; j++){
                    input0 = (c->nodes[currentActive0].inputs[j] >> 16 );
                    input1 = (c->nodes[currentActive0].inputs[j] & (0xFFFF) );

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        //unsigned int refIndex = ;
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        //unsigned int refIndex = input1 - N;
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearCompact(c, currentActive0, &exStack);


                //**------------------------ */
                if(2*(i+1) > c->numActiveNodes) break;

                activeInputs = MAX_ARITY;// c->nodes[currentActive].function_inputs_active >>;

                for(j = 0; j < activeInputs/2; j++){
                    input0 = (c->nodes[currentActive1].inputs[j] >> 16 );
                    input1 = (c->nodes[currentActive1].inputs[j] & (0xFFFF) );

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        //unsigned int refIndex = input0 - N;
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        //unsigned int refIndex = input1 - N;
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearCompact(c, currentActive1, &exStack);
            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];
            
                if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_TRAIN + local_id + (M_TRAIN*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;
        
        #ifdef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_TRAIN_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_TRAIN_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_TRAIN) )
    #endif 
           error[local_id] += error[local_id + i];
    }
        
    if(local_id == 0){
        fitness[group_id] = error[0] / M_TRAIN;
    }
}

void evaluateCircuitValidationCompact(__global CompactChromosome* c,
                                                __global float* data, 
                                                __global float* out, 
                                                __local float* error,
                                                __global float* fitness) {
    
    

    int i, k, j = 0;
    int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_VALIDATION/LOCAL_SIZE_VALIDATION) ; k++){

    #else
        for(k = 0; k < ceil( M_VALIDATION/ (float)LOCAL_SIZE_VALIDATION ) ; k++){
            
            if( k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION){
    #endif
        //printf("c");
        //int i, j;
        maxPredicted = -FLT_MAX ;
        predictedClass = 0;
        correctClass = 0;
        /*
        for(i = 0; i < MAX_NODES; i++){
            alreadyEvaluated[i] = -FLT_MAX ;
        }
        */
        ExStackLinear exStack;
        exStack.topIndex = -1;

        
        for(i = 0; i < ceil(c->numActiveNodes/(float)2); i++){
                //unsigned int active0 = c->activeNodes[i] >> 16;
                //unsigned int active1 = c->activeNodes[i] & (0xFFFF);

                currentActive0 = c->activeNodes[i] >> 16;
                currentActive1 = c->activeNodes[i] & (0xFFFF);

                activeInputs = MAX_ARITY;// c->nodes[currentActive].function_inputs_active >>;

                for(j = 0; j < activeInputs/2; j++){
                    input0 = (c->nodes[currentActive0].inputs[j] >> 16 );
                    input1 = (c->nodes[currentActive0].inputs[j] & (0xFFFF) );

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        //unsigned int refIndex = ;
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        //unsigned int refIndex = input1 - N;
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearCompact(c, currentActive0, &exStack);


                //**------------------------ */
                
                activeInputs = MAX_ARITY;// c->nodes[currentActive].function_inputs_active >>;

                for(j = 0; j < activeInputs/2; j++){
                    input0 = (c->nodes[currentActive1].inputs[j] >> 16 );
                    input1 = (c->nodes[currentActive1].inputs[j] & (0xFFFF) );

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        //unsigned int refIndex = input0 - N;
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        //unsigned int refIndex = input1 - N;
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearCompact(c, currentActive1, &exStack);
            }

            //for( i = 0; i < MAX_OUTPUTS/2 + MAX_OUTPUTS%2; i++) {
            for( i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];
            
                
                if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;
        
        #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION) )
    #endif 
           error[local_id] += error[local_id + i];
    }
        
    if(local_id == 0){
        fitness[group_id] = error[0] / M_VALIDATION;
    }
}
/**COMPACT */


/**IMAGE_R */
void evaluateCircuitImage_R(__read_only image2d_array_t indiv,
                                        __global float* data, 
                                        __global float* out, 
                                        __local float* error,
                                        __global float* fitness) {
    

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M/LOCAL_SIZE) ; k++){

    #else
        for(k = 0; k < ceil( M/ (float)LOCAL_SIZE ) ; k++){
            
            if( k * LOCAL_SIZE + local_id < M){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;
            /*
            for(i = 0; i < MAX_NODES; i++){
                alreadyEvaluated[i] = -FLT_MAX ;
            }
            */
            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;

            pixel = read_imageui(indiv, sampler, (int4)(0,0,group_id,0));
            int activenodes = pixel.x;

            //if(group_id == 0 && local_id == 0)
            //    printf("%d \n", pixel.x);

            for(i = 0; i < activenodes; i++){
                int indexCalc = (2*i)+1;
                pixel = read_imageui(indiv, sampler, (int4)(0, indexCalc, group_id, 0));


                //if(group_id == 0 && local_id == 0)
                //    printf("%d ", pixel.x);


                currentActive = pixel.x;
                activeInputs = MAX_ARITY;// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+1, indexCalc, group_id, 0));


                    //if(group_id == 0 && local_id == 0)
                    //     printf("%d ", pixel.x);


                    if (pixel.x >= N) { // se é um outro nó, empilha nó ou o resultado
                        //unsigned int refIndex = pixel.x - N;
                        pushExLinear(&exStack, alreadyEvaluated[pixel.x - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + ( M * pixel.x)]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActiveImage(indiv, indexCalc, &exStack);
                //printf("%f\n",  alreadyEvaluated[currentActive]);
            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                pixel = read_imageui(indiv, sampler, (int4)(i+1, 0, group_id, 0));
                //if(group_id == 0 && local_id == 0)
                //    printf("\n - %d \n", pixel.x);
                //unsigned int nodeIndex = pixel.x;
    
                if(alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE + local_id + (M*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){

        fitness[group_id] = error[0] / M;

    }
}

void evaluateCircuitValidationImage_R(__read_only image2d_array_t indiv,
                                                        __global float* data, 
                                                        __global float* out, 
                                                        __local float* error,
                                                        __global float* fitness) {
    

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_VALIDATION/LOCAL_SIZE_VALIDATION) ; k++){

    #else
        for(k = 0; k < ceil( M_VALIDATION/ (float)LOCAL_SIZE_VALIDATION ) ; k++){
            
            if( k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;
            /*
            for(i = 0; i < MAX_NODES; i++){
                alreadyEvaluated[i] = -FLT_MAX ;
            }
            */
    //ExStackLinear
            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;

            pixel = read_imageui(indiv, sampler, (int4)(0,0,group_id,0));
            int activenodes = pixel.x;
            for(i = 0; i < activenodes; i++){
                int indexCalc = (2*i)+1;
                pixel = read_imageui(indiv, sampler, (int4)(0, indexCalc, group_id, 0));

                currentActive = pixel.x;
                activeInputs = MAX_ARITY;// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+1, indexCalc, group_id, 0));

                    if (pixel.x >= N) { // se é um outro nó, empilha nó ou o resultado
                        //unsigned int refIndex = pixel.x - N;
                        pushExLinear(&exStack, alreadyEvaluated[pixel.x - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * pixel.x)]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActiveImage(indiv, indexCalc, &exStack);

            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                pixel = read_imageui(indiv, sampler, (int4)(i+1, 0, group_id, 0));

                //unsigned int nodeIndex = pixel.x;
    
                if(alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){

        fitness[group_id] = error[0] / M_VALIDATION;

    }
}
/**IMAGE_R */



/**IMAGE_RG */
void evaluateCircuitImage_RG(__read_only image2d_array_t indiv,
                                                __global float* data, 
                                                __global float* out, 
                                                __local float* error,
                                                __global float* fitness) {
    

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];
    //unsigned int refIndex;
    #ifndef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M/LOCAL_SIZE) ; k++){

    #else
        for(k = 0; k < ceil( M/ (float)LOCAL_SIZE ) ; k++){
            
            if( k * LOCAL_SIZE + local_id < M){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;
    /*
            for(i = 0; i < MAX_NODES; i++){
                alreadyEvaluated[i] = -FLT_MAX ;
            }
    */
            ExStackLinear exStack;
            exStack.topIndex = -1;

            uint4 pixel;
            pixel = read_imageui(indiv, sampler, (int4)(0,0,group_id,0));
            int activenodes = pixel.x;

            //if(group_id == 0 && local_id == 0)
            //    printf("%d %d", pixel.x, pixel.y);

            for(i = 0; i < activenodes; i++){
                
                int indexCalc = (2*i)+1;
                pixel = read_imageui(indiv, sampler, (int4)(0, indexCalc, group_id, 0));


                //if(group_id == 0 && local_id == 0)
                //    printf("\n%d %d \n", pixel.x, pixel.y);
                

                currentActive = pixel.x;
                activeInputs = MAX_ARITY/2;// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){

                    pixel = read_imageui(indiv, sampler, (int4)(j+1, indexCalc, group_id, 0));


                    //if(group_id == 5 && local_id == 0)
                    //     printf("%d %d ", pixel.x, pixel.y);


                    if (pixel.x >= N) { // se é um outro nó, empilha nó ou o resultado
                        //refIndex = pixel.x - N;
                        pushExLinear(&exStack, alreadyEvaluated[pixel.x - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + ( M * pixel.x)]);
                    }

                    if (pixel.y >= N) { // se é um outro nó, empilha nó ou o resultado
                    //    if(group_id == 5 && local_id == 0)
                    //        printf("already eval y\n");
                        //refIndex = pixel.y - N;
                        pushExLinear(&exStack, alreadyEvaluated[pixel.y - N]);
                        
                    } else {
                    //    if(group_id == 5 && local_id == 0)
                    //        printf("data access y\n");
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + ( M * pixel.y)]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActiveImageHalf(indiv, indexCalc, &exStack);
                //printf("%f\n",  alreadyEvaluated[currentActive]);
            }
            int index = 1;
            for( i = 0; i < MAX_OUTPUTS; i+=2) {
                pixel = read_imageui(indiv, sampler, (int4)(index, 0, group_id, 0));
                index++;
                //if(group_id == 0 && local_id == 0)
                //    printf("\n - %d \n", i);
                //unsigned int nodeIndex = pixel.x;
    
                if(alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE + local_id + (M*i)] == 1.0)? i : correctClass;

                if(i+1 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.y;
                if(alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i+1;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE + local_id + (M*(i+1))] == 1.0)? (i+1) : correctClass;
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){

        fitness[group_id] = error[0] / M;

    }
}

void evaluateCircuitValidationImage_RG(__read_only image2d_array_t indiv,
                                                            __global float* data, 
                                                            __global float* out, 
                                                            __local float* error,
                                                            __global float* fitness) {
    

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_VALIDATION/LOCAL_SIZE_VALIDATION) ; k++){

    #else
        for(k = 0; k < ceil( M_VALIDATION/ (float)LOCAL_SIZE_VALIDATION ) ; k++){
            
            if( k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION){
    #endif
            //printf("c");
            //int i, j;
            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;
            /*
            for(i = 0; i < MAX_NODES; i++){
                alreadyEvaluated[i] = -FLT_MAX ;
            }
            */
    //ExStackLinear
            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;

            pixel = read_imageui(indiv, sampler, (int4)(0,0,group_id,0));
            int activenodes = pixel.x;
            for(i = 0; i < activenodes; i++){
                int indexCalc = (2*i)+1;
                pixel = read_imageui(indiv, sampler, (int4)(0, indexCalc, group_id, 0));

                currentActive = pixel.x;
                activeInputs = MAX_ARITY/2;// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+1, indexCalc, group_id, 0));

                    if (pixel.x >= N) { // se é um outro nó, empilha nó ou o resultado
                        //unsigned int refIndex = pixel.x - N;
                        pushExLinear(&exStack, alreadyEvaluated[pixel.x - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * pixel.x)]);
                    }

                    if (pixel.y >= N) { // se é um outro nó, empilha nó ou o resultado
                        //unsigned int refIndex = pixel.y - N;
                        pushExLinear(&exStack, alreadyEvaluated[pixel.y - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * pixel.y)]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActiveImageHalf(indiv, indexCalc, &exStack);

            }
            int index = 1;
            for( i = 0; i < MAX_OUTPUTS; i+=2) {
                pixel = read_imageui(indiv, sampler, (int4)(index, 0, group_id, 0));
                index++;
                //unsigned int nodeIndex = pixel.x;
    
                if(alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*i)] == 1.0)? i : correctClass; 

                if(i+1 == MAX_OUTPUTS) break;
                //nodeIndex = pixel.y;
    
                if(alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i+1;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*(i+1))] == 1.0)? i+1 : correctClass;
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){

        fitness[group_id] = error[0] / M_VALIDATION;

    }
}
/**IMAGE_RG */



/**IMAGE_RGBA */
void evaluateCircuitImage_RGBA(__read_only image2d_array_t indiv,
                                                    __global float* data, 
                                                    __global float* out, 
                                                    __local float* error,
                                                    __global float* fitness) {
    

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];
    //unsigned int refIndex;
    #ifndef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M/LOCAL_SIZE) ; k++){

    #else
        for(k = 0; k < ceil( M/ (float)LOCAL_SIZE ) ; k++){
            
            if( k * LOCAL_SIZE + local_id < M){
    #endif
            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;

            uint4 pixel;
            pixel = read_imageui(indiv, sampler, (int4)(0,0,group_id,0));
            int activenodes = pixel.x;

            //if(group_id == 0 && local_id == 0)
            //    printf("%d %d", pixel.x, pixel.y);

            for(i = 0; i < activenodes; i++){
                
                int indexCalc = (2*i)+1;
                pixel = read_imageui(indiv, sampler, (int4)(0, indexCalc, group_id, 0));


                //if(group_id == 0 && local_id == 0)
                //    printf("\n%d %d \n", pixel.x, pixel.y);
                

                currentActive = pixel.x;
                activeInputs = MAX_ARITY/4;// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){

                    pixel = read_imageui(indiv, sampler, (int4)(j+1, indexCalc, group_id, 0));


                    if (pixel.x >= N) { // se é um outro nó, empilha nó ou o resultado
                        //refIndex = pixel.x - N;
                        pushExLinear(&exStack, alreadyEvaluated[pixel.x - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + ( M * pixel.x)]);
                    }

                    if (pixel.y >= N) { 
                        //refIndex = pixel.y - N;
                        pushExLinear(&exStack, alreadyEvaluated[pixel.y - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + ( M * pixel.y)]);
                    }

                    if (pixel.z >= N) { 
                        //refIndex = pixel.z - N;
                        pushExLinear(&exStack, alreadyEvaluated[pixel.z - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + ( M * pixel.z)]);
                    }

                    if (pixel.w >= N) {
                        //refIndex = pixel.w - N;
                        pushExLinear(&exStack, alreadyEvaluated[pixel.w - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + ( M * pixel.w)]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActiveImageQuarter(indiv, indexCalc, &exStack);
                //printf("%f\n",  alreadyEvaluated[currentActive]);
            }

            int index = 1;
            for( i = 0; i < MAX_OUTPUTS; i+=4) {
                pixel = read_imageui(indiv, sampler, (int4)(index, 0, group_id, 0));
                index++;
                //if(group_id == 0 && local_id == 0)
                //    printf("\n - %d \n", i);
                //unsigned int nodeIndex = pixel.x;
    
                if(alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE + local_id + (M*i)] == 1.0)? i : correctClass;

                if(i+1 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.y;
                if(alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i+1;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE + local_id + (M*(i+1))] == 1.0)? (i+1) : correctClass;

                if(i+2 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.z;
                if(alreadyEvaluated[pixel.z] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.z];
                    predictedClass = i+2;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE + local_id + (M*(i+2))] == 1.0)? (i+2) : correctClass;

                if(i+3 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.w;
                if(alreadyEvaluated[pixel.w] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.w];
                    predictedClass = i+3;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE + local_id + (M*(i+3))] == 1.0)? (i+3) : correctClass;
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){

        fitness[group_id] = error[0] / M;

    }
}

void evaluateCircuitValidationImage_RGBA(__read_only image2d_array_t indiv,
                                                                __global float* data, 
                                                                __global float* out, 
                                                                __local float* error,
                                                                __global float* fitness) {
    

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_VALIDATION/LOCAL_SIZE_VALIDATION) ; k++){

    #else
        for(k = 0; k < ceil( M_VALIDATION/ (float)LOCAL_SIZE_VALIDATION ) ; k++){
            
            if( k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION){
    #endif
            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;

            pixel = read_imageui(indiv, sampler, (int4)(0,0,group_id,0));
            int activenodes = pixel.x;
            for(i = 0; i < activenodes; i++){
                int indexCalc = (2*i)+1;
                pixel = read_imageui(indiv, sampler, (int4)(0, indexCalc, group_id, 0));

                currentActive = pixel.x;
                activeInputs = MAX_ARITY/4;// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+1, indexCalc, group_id, 0));

                    if (pixel.x >= N) { // se é um outro nó, empilha nó ou o resultado
                        //unsigned int refIndex = pixel.x - N;
                        pushExLinear(&exStack, alreadyEvaluated[pixel.x - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * pixel.x)]);
                    }

                    if (pixel.y >= N) { // se é um outro nó, empilha nó ou o resultado
                        //unsigned int refIndex = pixel.y - N;
                        pushExLinear(&exStack, alreadyEvaluated[pixel.y - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * pixel.y)]);
                    }

                    if (pixel.z >= N) { // se é um outro nó, empilha nó ou o resultado
                        //unsigned int refIndex = pixel.z - N;
                        pushExLinear(&exStack, alreadyEvaluated[pixel.z - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * pixel.z)]);
                    }

                    if (pixel.w >= N) { // se é um outro nó, empilha nó ou o resultado
                        //unsigned int refIndex = pixel.w - N;
                        pushExLinear(&exStack, alreadyEvaluated[pixel.w - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * pixel.w)]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActiveImageQuarter(indiv, indexCalc, &exStack);

            }
            int index = 1;
            for( i = 0; i < MAX_OUTPUTS; i+=2) {
                pixel = read_imageui(indiv, sampler, (int4)(index, 0, group_id, 0));
                index++;
                //unsigned int nodeIndex = pixel.x;
    
                if(alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*i)] == 1.0)? i : correctClass; 

                if(i+1 == MAX_OUTPUTS) break;
                //nodeIndex = pixel.y;
    
                if(alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i+1;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*(i+1))] == 1.0)? i+1 : correctClass;

                if(i+2 == MAX_OUTPUTS) break;
                //nodeIndex = pixel.z;
    
                if(alreadyEvaluated[pixel.z] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.z];
                    predictedClass = i+2;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*(i+2))] == 1.0)? i+2 : correctClass;
             
                if(i+3 == MAX_OUTPUTS) break;
                   // nodeIndex = pixel.w;
    
                if(alreadyEvaluated[pixel.w] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.w];
                    predictedClass = i+3;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*(i+3))] == 1.0)? i+3 : correctClass;
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){

        fitness[group_id] = error[0] / M_VALIDATION;

    }
}
/**IMAGE_RGBA */


/**COMPACT IMAGE_R */
void evaluateCircuitTrainCompactImage_R(__read_only image2d_array_t indiv,
                                                        __global float* data, 
                                                        __global float* out, 
                                                        __local float* error,
                                                        __global float* fitness) {
    

    int i, k, j = 0;
    unsigned int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;

    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;
    int indexCalc;
    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_TRAIN/LOCAL_SIZE_TRAIN) ; k++){

    #else
        for(k = 0; k < ceil( M_TRAIN/ (float)LOCAL_SIZE_TRAIN ) ; k++){
            
            if( k * LOCAL_SIZE_TRAIN + local_id < M_TRAIN){
    #endif

            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;

            pixel = read_imageui(indiv, sampler, (int4)(0,0,group_id,0));
            int activenodes = pixel.x;
            /*if(group_id == 0 && local_id == 0)
                printf("%d %d \n",activenodes, activenodes);
    */

            for(i = 0; i < ceil(activenodes/(float)2); i++){
                uint4 pixelAux;
                pixelAux = read_imageui(indiv, sampler, (int4)(0, (i + 1), group_id, 0));

                currentActive0 = pixelAux.x >> 16;
                currentActive1 = pixelAux.x & (0xFFFF);
                /*if(group_id == 0 && local_id == 0)
                    printf("%d %d \n",currentActive0, currentActive1);
                */
                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = MAX_ARITY;// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs/2; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageCompact(indiv, indexCalc, &exStack);
                
                if(2*(i+1) >activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

                activeInputs = MAX_ARITY;// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs/2; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));
                    
                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageCompact(indiv, indexCalc, &exStack);

            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                pixel = read_imageui(indiv, sampler, (int4)(i+1, 0, group_id, 0));

                //unsigned int nodeIndex = pixel.x;
    
                if(alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_TRAIN + local_id + (M_TRAIN*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_TRAIN_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_TRAIN_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_TRAIN) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){

        fitness[group_id] = error[0] / M_TRAIN;

    }
}

void evaluateCircuitValidationCompactImage_R(__read_only image2d_array_t indiv,
                                                        __global float* data, 
                                                        __global float* out, 
                                                        __local float* error,
                                                        __global float* fitness) {
    

    int i, k, j = 0;
    unsigned int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;

    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;
    int indexCalc;
    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_VALIDATION/LOCAL_SIZE_VALIDATION) ; k++){

    #else
        for(k = 0; k < ceil( M_VALIDATION/ (float)LOCAL_SIZE_VALIDATION ) ; k++){
            
            if( k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION){
    #endif

            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;

            pixel = read_imageui(indiv, sampler, (int4)(0,0,group_id,0));
            int activenodes = pixel.x;
            /*if(group_id == 0 && local_id == 0)
                printf("%d %d \n",activenodes, activenodes);
    */

            for(i = 0; i < ceil(activenodes/(float)2); i++){
                uint4 pixelAux;
                pixelAux = read_imageui(indiv, sampler, (int4)(0, (i + 1), group_id, 0));

                currentActive0 = pixelAux.x >> 16;
                currentActive1 = pixelAux.x & (0xFFFF);
                //if(group_id == 1 && local_id == 0)
                //    printf("%d %d \n",currentActive0, currentActive1);
                
                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = MAX_ARITY;// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs/2; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageCompact(indiv, indexCalc, &exStack);

                if(2*(i+1) > activenodes) break;
                
                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

                activeInputs = MAX_ARITY;// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs/2; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));
                    
                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageCompact(indiv, indexCalc, &exStack);

            }

            for( i = 0; i < MAX_OUTPUTS; i++) {
                pixel = read_imageui(indiv, sampler, (int4)(i+1, 0, group_id, 0));

                //unsigned int nodeIndex = pixel.x;
    
                if(alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*i)] == 1.0)? i : correctClass; 
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){

        fitness[group_id] = error[0] / M_VALIDATION;

    }
}
/**COMPACT IMAGE_R */


/**COMPACT IMAGE_RG */

float executeFunctionLinearActiveImageHalfCompact(__read_only image2d_array_t c, int node, ExStackLinear* exStack){
    int i;
    float result, sum;
    int group_id = get_group_id(0);
    int local_id = get_group_id(0);
    
    uint4 pixelInt;
    float4 pixelFloat;

    pixelInt = read_imageui(c, sampler, (int4)(1,node+1,group_id,0));

    unsigned int inputs = MAX_ARITY;//c->nodes[node].maxInputs;
    //printf("func %d ", pixelInt.x);
    switch (pixelInt.x){

        #ifdef ONE

        case ONE:
            result = 1;
            break;
        #endif
        
        #ifdef ZERO

        case ZERO:
            result = 0;
            break;
        #endif
        
        #ifdef PI

        case PI:
            result = CONST_PI;
            break;
        #endif
        
        #ifdef WIRE

        case WIRE:
            result = popExLinear(exStack);
            break;
        #endif
        
        #ifdef SIG

        case SIG:
            sum = 0;
            for(i = 0; i < inputs/2; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+2,node+1,group_id,0));
                sum += (popExLinear(exStack) *  pixelFloat.x);
                sum += (popExLinear(exStack) *  pixelFloat.y);

            }
            result = 1.0f / (1 + exp(-sum));
            break;
        #endif

        default:
            break;
    }
    return result;
}

void evaluateCircuitTrainCompactImage_RG(__read_only image2d_array_t indiv,
                                            __global float* data, 
                                            __global float* out, 
                                            __local float* error,
                                            __global float* fitness) {
    

    int i, k, j = 0;
    unsigned int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;

    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;
    int indexCalc;
    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_TRAIN/LOCAL_SIZE_TRAIN) ; k++){

    #else
        for(k = 0; k < ceil( M_TRAIN/ (float)LOCAL_SIZE_TRAIN ) ; k++){
            
            if( k * LOCAL_SIZE_TRAIN + local_id < M_TRAIN){
    #endif

            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;
            int cont = 0;

            pixel = read_imageui(indiv, sampler, (int4)(0,0,group_id,0));
            int activenodes = pixel.x;

            for(i = 0; i < ceil(activenodes/(float)4); i++){
                uint4 pixelAux;
                pixelAux = read_imageui(indiv, sampler, (int4)(0, (i + 1), group_id, 0));

                currentActive0 = pixelAux.x >> 16;
                currentActive1 = pixelAux.x & (0xFFFF);

                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                activeInputs = ceil(MAX_ARITY/(float)4);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                }
                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageHalfCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

                activeInputs = ceil(MAX_ARITY/(float)4);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                   
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageHalfCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                currentActive0 = pixelAux.y >> 16;
                currentActive1 = pixelAux.y & (0xFFFF);

                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

                activeInputs = ceil(MAX_ARITY/(float)4);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageHalfCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

                activeInputs = ceil(MAX_ARITY/(float)4);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageHalfCompact(indiv, indexCalc, &exStack);
                cont++;

            }
            
            int index = 0;
            for( i = 0; i < MAX_OUTPUTS; i+=2) {
                pixel = read_imageui(indiv, sampler, (int4)(index+1, 0, group_id, 0));
                index++;
                //if(group_id == 0 && local_id == 0)
                //    printf("\n - %d \n", i);
                //unsigned int nodeIndex = pixel.x;
    
                if(alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_TRAIN + local_id + (M_TRAIN*i)] == 1.0)? i : correctClass;

                if(i+1 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.y;
                if(alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i+1;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_TRAIN + local_id + (M_TRAIN*(i+1))] == 1.0)? (i+1) : correctClass;

            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_TRAIN_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_TRAIN_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_TRAIN) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){

        fitness[group_id] = error[0] / M_TRAIN;

    }
}

void evaluateCircuitValidationCompactImage_RG(__read_only image2d_array_t indiv,
                                            __global float* data, 
                                            __global float* out, 
                                            __local float* error,
                                            __global float* fitness) {
    

    int i, k, j = 0;
    unsigned int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;

    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;
    int indexCalc;
    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_VALIDATION/LOCAL_SIZE_VALIDATION) ; k++){

    #else
        for(k = 0; k < ceil( M_VALIDATION/ (float)LOCAL_SIZE_VALIDATION ) ; k++){
            
            if( k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION){
    #endif

            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;
            int cont = 0;

            pixel = read_imageui(indiv, sampler, (int4)(0,0,group_id,0));
            int activenodes = pixel.x;

            for(i = 0; i < ceil(activenodes/(float)4); i++){
                uint4 pixelAux;
                pixelAux = read_imageui(indiv, sampler, (int4)(0, (i + 1), group_id, 0));

                currentActive0 = pixelAux.x >> 16;
                currentActive1 = pixelAux.x & (0xFFFF);

                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                activeInputs = ceil(MAX_ARITY/(float)4);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                }
                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageHalfCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

                activeInputs = ceil(MAX_ARITY/(float)4);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                   
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageHalfCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                currentActive0 = pixelAux.y >> 16;
                currentActive1 = pixelAux.y & (0xFFFF);

                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

                activeInputs = ceil(MAX_ARITY/(float)4);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageHalfCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

                activeInputs = ceil(MAX_ARITY/(float)4);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageHalfCompact(indiv, indexCalc, &exStack);
                cont++;
                

            }
            
            int index = 0;
            for( i = 0; i < MAX_OUTPUTS; i+=2) {
                pixel = read_imageui(indiv, sampler, (int4)(index+1, 0, group_id, 0));
                index++;
                //if(group_id == 0 && local_id == 0)
                //    printf("\n - %d \n", i);
                //unsigned int nodeIndex = pixel.x;
    
                if(alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*i)] == 1.0)? i : correctClass;

                if(i+1 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.y;
                if(alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i+1;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*(i+1))] == 1.0)? (i+1) : correctClass;

            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){

        fitness[group_id] = error[0] / M_VALIDATION;

    }
}
/**COMPACT IMAGE_RG */

/**COMPACT IMAGE_RGBA */

float executeFunctionLinearActiveImageQuarterCompact(__read_only image2d_array_t c, int node, ExStackLinear* exStack){
    int i;
    float result, sum;
    int group_id = get_group_id(0);
    int local_id = get_group_id(0);
    
    uint4 pixelInt;
    float4 pixelFloat;

    pixelInt = read_imageui(c, sampler, (int4)(1,node+1,group_id,0));

    unsigned int inputs = MAX_ARITY;//c->nodes[node].maxInputs;
    //printf("func %d ", pixelInt.x);
    switch (pixelInt.x){

        #ifdef ONE

        case ONE:
            result = 1;
            break;
        #endif
        
        #ifdef ZERO

        case ZERO:
            result = 0;
            break;
        #endif
        
        #ifdef PI

        case PI:
            result = CONST_PI;
            break;
        #endif
        
        #ifdef WIRE

        case WIRE:
            result = popExLinear(exStack);
            break;
        #endif
        
        #ifdef SIG

        case SIG:
            sum = 0;
            for(i = 0; i < inputs/4; i++){
                pixelFloat = read_imagef(c, sampler, (int4)(i+2,node+1,group_id,0));
                sum += (popExLinear(exStack) *  pixelFloat.x);
                sum += (popExLinear(exStack) *  pixelFloat.y);
                sum += (popExLinear(exStack) *  pixelFloat.z);
                sum += (popExLinear(exStack) *  pixelFloat.w);

            }
            result = 1.0f / (1 + exp(-sum));
            break;
        #endif

        default:
            break;
    }
    return result;
}

void evaluateCircuitTrainCompactImage_RGBA(__read_only image2d_array_t indiv,
                                                        __global float* data, 
                                                        __global float* out, 
                                                        __local float* error,
                                                        __global float* fitness) {
    

    int i, k, j = 0;
    unsigned int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;

    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;
    int indexCalc;
    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_TRAIN/LOCAL_SIZE_TRAIN) ; k++){

    #else
        for(k = 0; k < ceil( M_TRAIN/ (float)LOCAL_SIZE_TRAIN ) ; k++){
            
            if( k * LOCAL_SIZE_TRAIN + local_id < M_TRAIN){
    #endif

            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;
            int cont = 0;

            pixel = read_imageui(indiv, sampler, (int4)(0,0,group_id,0));
            int activenodes = pixel.x;

            for(i = 0; i < ceil(activenodes/(float)8); i++){
                uint4 pixelAux;
                pixelAux = read_imageui(indiv, sampler, (int4)(0, (i + 1), group_id, 0));

                currentActive0 = pixelAux.x >> 16;
                currentActive1 = pixelAux.x & (0xFFFF);

                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }
                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                currentActive0 = pixelAux.y >> 16;
                currentActive1 = pixelAux.y & (0xFFFF);
                /*if(group_id == 0 && local_id == 0)
                    printf("%d %d \n",currentActive0, currentActive1);
                */
                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

               activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                currentActive0 = pixelAux.z >> 16;
                currentActive1 = pixelAux.z & (0xFFFF);
                /*if(group_id == 0 && local_id == 0)
                    printf("%d %d \n",currentActive0, currentActive1);
                */
                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

               activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                currentActive0 = pixelAux.w >> 16;
                currentActive1 = pixelAux.w & (0xFFFF);

                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + ( M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);

            }
            
            int index = 0;
            for( i = 0; i < MAX_OUTPUTS; i+=4) {
                pixel = read_imageui(indiv, sampler, (int4)(index+1, 0, group_id, 0));
                index++;
                //if(group_id == 0 && local_id == 0)
                //    printf("\n - %d \n", i);
                //unsigned int nodeIndex = pixel.x;
    
                if(alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_TRAIN + local_id + (M_TRAIN*i)] == 1.0)? i : correctClass;

                if(i+1 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.y;
                if(alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i+1;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_TRAIN + local_id + (M_TRAIN*(i+1))] == 1.0)? (i+1) : correctClass;

                if(i+2 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.z;
                if(alreadyEvaluated[pixel.z] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.z];
                    predictedClass = i+2;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_TRAIN + local_id + (M_TRAIN*(i+2))] == 1.0)? (i+2) : correctClass;

                if(i+3 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.w;
                if(alreadyEvaluated[pixel.w] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.w];
                    predictedClass = i+3;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_TRAIN + local_id + (M_TRAIN*(i+3))] == 1.0)? (i+3) : correctClass;
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_TRAIN_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_TRAIN_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_TRAIN) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){

        fitness[group_id] = error[0] / M_TRAIN;

    }
}

void evaluateCircuitValidationCompactImage_RGBA(__read_only image2d_array_t indiv,
                                                        __global float* data, 
                                                        __global float* out, 
                                                        __local float* error,
                                                        __global float* fitness) {
    

    int i, k, j = 0;
    unsigned int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;

    int erro = 0;

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;
    int indexCalc;
    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
    for(k = 0; k < (M_VALIDATION/LOCAL_SIZE_VALIDATION) ; k++){

    #else
        for(k = 0; k < ceil( M_VALIDATION/ (float)LOCAL_SIZE_VALIDATION ) ; k++){
            
            if( k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION){
    #endif

            maxPredicted = -FLT_MAX ;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;
            int cont = 0;

            pixel = read_imageui(indiv, sampler, (int4)(0,0,group_id,0));
            int activenodes = pixel.x;
            /*if(group_id == 0 && local_id == 0)
                printf("%d %d \n",activenodes, activenodes);
    */

            for(i = 0; i < ceil(activenodes/(float)8); i++){
                uint4 pixelAux;
                pixelAux = read_imageui(indiv, sampler, (int4)(0, (i + 1), group_id, 0));

                currentActive0 = pixelAux.x >> 16;
                currentActive1 = pixelAux.x & (0xFFFF);
                /*if(group_id == 0 && local_id == 0)
                    printf("%d %d \n",currentActive0, currentActive1);
                */
                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                currentActive0 = pixelAux.y >> 16;
                currentActive1 = pixelAux.y & (0xFFFF);
                /*if(group_id == 0 && local_id == 0)
                    printf("%d %d \n",currentActive0, currentActive1);
                */
                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

               activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                currentActive0 = pixelAux.z >> 16;
                currentActive1 = pixelAux.z & (0xFFFF);
                /*if(group_id == 0 && local_id == 0)
                    printf("%d %d \n",currentActive0, currentActive1);
                */
                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

               activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                currentActive0 = pixelAux.w >> 16;
                currentActive1 = pixelAux.w & (0xFFFF);
                /*if(group_id == 0 && local_id == 0)
                    printf("%d %d \n",currentActive0, currentActive1);
                */
                indexCalc = (2*currentActive0)+1;

                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = read_imageui(indiv, sampler, (int4)(1, indexCalc, group_id, 0));

               activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = read_imageui(indiv, sampler, (int4)(j+2, indexCalc, group_id, 0));
                    
                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                        
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);


                
            }
            int index = 0;
            for( i = 0; i < MAX_OUTPUTS; i+=4) {
                pixel = read_imageui(indiv, sampler, (int4)(index+1, 0, group_id, 0));
                index++;
                //if(group_id == 0 && local_id == 0)
                //    printf("\n - %d \n", i);
                //unsigned int nodeIndex = pixel.x;
    
                if(alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*i)] == 1.0)? i : correctClass;

                if(i+1 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.y;
                if(alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i+1;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*(i+1))] == 1.0)? (i+1) : correctClass;

                if(i+2 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.z;
                if(alreadyEvaluated[pixel.z] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.z];
                    predictedClass = i+2;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*(i+2))] == 1.0)? (i+2) : correctClass;

                if(i+3 >= MAX_OUTPUTS) break;

                //nodeIndex = pixel.w;
                if(alreadyEvaluated[pixel.w] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.w];
                    predictedClass = i+3;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k*LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION*(i+3))] == 1.0)? (i+3) : correctClass;
            }

            erro += (predictedClass == correctClass)? 1.0 : 0.0;

            
        
        #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = erro;
    barrier(CLK_LOCAL_MEM_FENCE);

    ///redução erros por work group
    for(i =  LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 /2 ; i > 0; i>>=1){
        barrier(CLK_LOCAL_MEM_FENCE);


    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if( local_id < i )
    #else
        /* LOCAL_SIZE is not power of 2, so we need to perform an additional
        * check to ensure that no access beyond PE's range will occur. */ 
        if( (local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION) )
    #endif 
           error[local_id] += error[local_id + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    

    if(local_id == 0){

        fitness[group_id] = error[0] / M_VALIDATION;

    }
}


/**COMPACT IMAGE_RGBA */

/**PADRÃO */
__kernel void evolve(__global unsigned int* functionSet,
                     __global int *seeds,
                     __global Chromosome* population,
                     __global Chromosome* best){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    //printf("%d\n", local_id);
    int seed = seeds[global_id];
    //int seed = seeds[group_id];
    //printf("%d\n", local_id);
    //if(local_id == 0)
    //    population[group_id] = *best;

    //barrier(CLK_GLOBAL_MEM_FENCE);
    //mutateTopologyProbabilistic(&newBest[group_id], functionSet, &seed,  0);
    mutateTopologyProbabilistic2(&population[group_id], functionSet, &seed,  0);

    //barrier(CLK_GLOBAL_MEM_FENCE);

    //if(local_id == 0)
    seeds[global_id] = seed;
}

__kernel void evaluateTest(__global float* dataset,
                            __global float* outputs,
                            __global unsigned int* functionSet,
                            __global Chromosome* individual,
                            __local float* error,
                            __global float* fitness,
                            __global float* fitnessValidation){


    evaluateCircuitTest(individual, dataset, outputs, error, fitness);

}

__kernel void evaluateTrain(__global float* dataset,
                            __global float* outputs,
                            __global unsigned int* functionSet,
                            __global Chromosome* pop,
                            __local float* error,
                            __global float* fitness,
                            __global float* fitnessValidation){

    int group_id = get_group_id(0);
    evaluateCircuit(&pop[group_id], dataset, outputs, error, fitness);
}



__kernel void evaluateValidation(__global float* dataset,
                            __global float* outputs,
                            __global unsigned int* functionSet,
                            __global Chromosome* pop,
                            __local float* error,
                            __global float* fitness,
                            __global float* fitnessValidation){

    int group_id = get_group_id(0);
    evaluateCircuitValidation(&pop[group_id], dataset, outputs, error, fitnessValidation);

}

__kernel void evaluateTrainValidation(__global float* datasetTrain,
                                        __global float* outputsTrain,
                                        __global float* datasetValid,
                                        __global float* outputsValid,
                                        __global unsigned int* functionSet,
                                        __global Chromosome* pop,
                                        __global Chromosome* best,
                                        __local float* error,
                                        __global float* fitness,
                                        __global float* fitnessValidation){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    
    //pop[group_id] = *best;
    //barrier(CLK_GLOBAL_MEM_FENCE);

    evaluateCircuit(&pop[group_id], datasetTrain, outputsTrain, error, fitness);

    barrier(CLK_GLOBAL_MEM_FENCE);

    evaluateCircuitTrainValidation(&pop[group_id], datasetValid, outputsValid, error, fitnessValidation);

}
/**PADRÃO */


/**COMPACTO */
__kernel void evaluateTrainCompact(__global float* dataset,
                            __global float* outputs,
                            __global unsigned int* functionSet,
                            __global CompactChromosome* pop,
                            __local float* error,
                            __global float* fitness,
                            __global float* fitnessValidation){

    int group_id = get_group_id(0);
    
    evaluateCircuitTrainCompact(&pop[group_id], dataset, outputs, error, fitness);

}

__kernel void evaluateValidationCompact(__global float* dataset,
                            __global float* outputs,
                            __global unsigned int* functionSet,
                            __global CompactChromosome* pop,
                            __local float* error,
                            __global float* fitness,
                            __global float* fitnessValidation){

    int group_id = get_group_id(0);
    evaluateCircuitValidationCompact(&pop[group_id], dataset, outputs, error, fitnessValidation);

}
/**COMPACTO */


/**ATIVO */
__kernel void evaluateTrainValidationActive(__global float* datasetTrain,
                                            __global float* outputsTrain,
                                            __global float* datasetValid,
                                            __global float* outputsValid,
                                            __global unsigned int* functionSet,
                                            __global ActiveChromosome* pop,
                                            __local float* error,
                                            __global float* fitness,
                                            __global float* fitnessValidation){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    
    //pop[group_id] = *best;
    //barrier(CLK_GLOBAL_MEM_FENCE);

    evaluateCircuitActive(&pop[group_id], datasetTrain, outputsTrain, error, fitness);

    barrier(CLK_GLOBAL_MEM_FENCE);

    evaluateCircuitTrainValidationActive(&pop[group_id], datasetValid, outputsValid, error, fitnessValidation);

}
/**ATIVO */

/**IMAGE_R */
__kernel void evaluateTrainImage(__global float* data,
                                __global float* out,
                                __global unsigned int* functionSet,
                                __read_only image2d_array_t pop,
                                __local float* error,
                                __global float* fitness,
                                __global float* fitnessValidation){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    /*
    uint4 pixel;
    int4 coord = (100,0,0,0);

    pixel = read_imageui(pop, sampler, (int4)(1,1,group_id,0));
    if(local_id == 0)
    printf("%d pixel= %d, %d, %d, %d\n", group_id, pixel.x, pixel.y, pixel.z, pixel.w);
   */
    //evaluateCircuitParallelTrainLinearImage(&pop[group_id], dataset, outputs, error);


    evaluateCircuitImage_R(pop, data, out, error, fitness);

}

__kernel void evaluateValidationImage(__global float* data,
                                    __global float* out,
                                    __global unsigned int* functionSet,
                                    __read_only image2d_array_t pop,
                                    __local float* error,
                                    __global float* fitness,
                                    __global float* fitnessValidation){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    evaluateCircuitValidationImage_R(pop, data, out, error, fitnessValidation);

}
/**IMAGE_R */


/**IMAGE_RG */
__kernel void evaluateTrainImageHalf(__global float* data,
                                    __global float* out,
                                    __global unsigned int* functionSet,
                                    __read_only image2d_array_t pop,
                                    __local float* error,
                                    __global float* fitness,
                                    __global float* fitnessValidation){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    /*
    uint4 pixel;
    int4 coord = (100,0,0,0);

    pixel = read_imageui(pop, sampler, (int4)(0,0,group_id,0));
    if(local_id == 0 ){
        int i;
        for(i = 0; i < 11; i ++){
            pixel = read_imageui(pop, sampler, (int4)(i,603,group_id,0));
            printf("%d pixel= %d, %d, %d, %d\n", group_id, pixel.x, pixel.y, pixel.z, pixel.w);
        }
    }
    */
   
    //evaluateCircuitParallelTrainLinearImage(&pop[group_id], dataset, outputs, error);


    evaluateCircuitImage_RG(pop, data, out, error, fitness);

}

__kernel void evaluateValidationImageHalf(__global float* data,
                                        __global float* out,
                                        __global unsigned int* functionSet,
                                        __read_only image2d_array_t pop,
                                        __local float* error,
                                        __global float* fitness,
                                        __global float* fitnessValidation){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    evaluateCircuitValidationImage_RG(pop, data, out, error, fitnessValidation);

}
/**IMAGE_RG */



/**IMAGE_RGBA */
__kernel void evaluateTrainImageQuarter(__global float* data,
                                    __global float* out,
                                    __global unsigned int* functionSet,
                                    __read_only image2d_array_t pop,
                                    __local float* error,
                                    __global float* fitness,
                                    __global float* fitnessValidation){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    /*
    uint4 pixel;
    int4 coord = (100,0,0,0);

    pixel = read_imageui(pop, sampler, (int4)(0,0,group_id,0));
    if(local_id == 0 ){
        int i;
        for(i = 0; i < 11; i ++){
            pixel = read_imageui(pop, sampler, (int4)(i,603,group_id,0));
            printf("%d pixel= %d, %d, %d, %d\n", group_id, pixel.x, pixel.y, pixel.z, pixel.w);
        }
    }
    */
   
    //evaluateCircuitParallelTrainLinearImage(&pop[group_id], dataset, outputs, error);


    evaluateCircuitImage_RGBA(pop, data, out, error, fitness);

}

__kernel void evaluateValidationImageQuarter(__global float* data,
                                        __global float* out,
                                        __global unsigned int* functionSet,
                                        __read_only image2d_array_t pop,
                                        __local float* error,
                                        __global float* fitness,
                                        __global float* fitnessValidation){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    evaluateCircuitValidationImage_RGBA(pop, data, out, error, fitnessValidation);

}
/**IMAGE_RGBA */



/**COMPACT IMAGE_R */
__kernel void evaluateTrainImageCompact(__global float* data,
                                    __global float* out,
                                    __global unsigned int* functionSet,
                                    __read_only image2d_array_t pop,
                                    __local float* error,
                                    __global float* fitness,
                                    __global float* fitnessValidation){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    /*
    uint4 pixel;
    int4 coord = (100,0,0,0);

    pixel = read_imageui(pop, sampler, (int4)(0,0,group_id,0));
    if(local_id == 0 ){
        //printf("%d, %d\n", get_image_width(pop), get_image_height(pop));
        int i;
        //for(i = 0; i < 11; i ++){
            pixel = read_imageui(pop, sampler, (int4)(0,1,group_id,0));
            printf("%d pixel= %d, %d, %d, %d\n", group_id, pixel.x, pixel.y, pixel.z, pixel.w);
        //}
    }
    */
   
    //evaluateCircuitParallelTrainLinearImage(&pop[group_id], dataset, outputs, error);


    evaluateCircuitTrainCompactImage_R(pop, data, out, error, fitness);

}

__kernel void evaluateValidationImageCompact(__global float* data,
                                        __global float* out,
                                        __global unsigned int* functionSet,
                                        __read_only image2d_array_t pop,
                                        __local float* error,
                                        __global float* fitness,
                                        __global float* fitnessValidation){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    evaluateCircuitValidationCompactImage_R(pop, data, out, error, fitnessValidation);

}
/**COMPACT IMAGE_R */

/**COMPACT IMAGE_RG */
__kernel void evaluateTrainImageHalfCompact(__global float* data,
                                    __global float* out,
                                    __global unsigned int* functionSet,
                                    __read_only image2d_array_t pop,
                                    __local float* error,
                                    __global float* fitness,
                                    __global float* fitnessValidation){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    /*
    uint4 pixel;
    int4 coord = (100,0,0,0);

    pixel = read_imageui(pop, sampler, (int4)(0,0,group_id,0));
    if(local_id == 0 ){
        //printf("%d, %d\n", get_image_width(pop), get_image_height(pop));
        int i;
        //for(i = 0; i < 11; i ++){
            pixel = read_imageui(pop, sampler, (int4)(0,0,group_id,0));
            printf("%d pixel= %d, %d, %d, %d\n", group_id, pixel.x, pixel.y, pixel.z, pixel.w);
        //}
    }
    */
   
    //evaluateCircuitParallelTrainLinearImage(&pop[group_id], dataset, outputs, error);


    evaluateCircuitTrainCompactImage_RG(pop, data, out, error, fitness);

}

__kernel void evaluateValidationImageHalfCompact(__global float* data,
                                        __global float* out,
                                        __global unsigned int* functionSet,
                                        __read_only image2d_array_t pop,
                                        __local float* error,
                                        __global float* fitness,
                                        __global float* fitnessValidation){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    evaluateCircuitValidationCompactImage_RG(pop, data, out, error, fitnessValidation);

}
/**COMPACT IMAGE_RG */

/**COMPACT IMAGE_RGBA */
__kernel void evaluateTrainImageQuarterCompact(__global float* data,
                                    __global float* out,
                                    __global unsigned int* functionSet,
                                    __read_only image2d_array_t pop,
                                    __local float* error,
                                    __global float* fitness,
                                    __global float* fitnessValidation){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    /*
    uint4 pixel;
    int4 coord = (100,0,0,0);

    pixel = read_imageui(pop, sampler, (int4)(0,0,group_id,0));
    if(local_id == 0 ){
        //printf("%d, %d\n", get_image_width(pop), get_image_height(pop));
        int i;
        //for(i = 0; i < 11; i ++){
            pixel = read_imageui(pop, sampler, (int4)(1,0,group_id,0));
            printf("%d pixel= %d, %d, %d, %d\n", group_id, pixel.x, pixel.y, pixel.z, pixel.w);
        //}
    }
    */
   
    //evaluateCircuitParallelTrainLinearImage(&pop[group_id], dataset, outputs, error);


    evaluateCircuitTrainCompactImage_RGBA(pop, data, out, error, fitness);

}

__kernel void evaluateValidationImageQuarterCompact(__global float* data,
                                        __global float* out,
                                        __global unsigned int* functionSet,
                                        __read_only image2d_array_t pop,
                                        __local float* error,
                                        __global float* fitness,
                                        __global float* fitnessValidation){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    evaluateCircuitValidationCompactImage_RGBA(pop, data, out, error, fitnessValidation);

}
/**COMPACT IMAGE_RGBA */

__kernel void testMemory(__global int* test, __local float* error){

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);
     
    if(local_id == 0)
        printf("%d\n", test[group_id]);
}

