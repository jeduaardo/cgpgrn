//
// Created by bruno on 04/02/2020.
//

#ifndef PCGP_UTILS_H
#define PCGP_UTILS_H

/** C headers */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <random>

/** CPP headers */
#include <iostream>
#include <string>
#include <iomanip>
#include <vector>
#include <limits>
#include <climits>
#include <ctime>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "constants.h"


#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>


#define GPU_PLATFORM 0
#define CPU_PLATFORM 1

#define GPU_DEVICE 0
#define CPU_DEVICE 1


//unsigned short int randomFunction(Parameters *p, int *seed);
unsigned int randomFunction(Parameters *p, int *seed);
//unsigned short int randomInput(Parameters *p, unsigned short int index, int *seed);
unsigned int randomInput(Parameters *p, unsigned short int index, int *seed);
float randomConnectionWeight(Parameters *p, int *seed);
int randomInterval(int inf_bound, int sup_bound, int *seed);
float randomProb(int *seed);
//unsigned short int randomOutputIndex(int* seed);
unsigned int randomOutputIndex(int* seed);
//unsigned short int getFunctionInputs(unsigned short int function);
unsigned int getFunctionInputs(unsigned int function);

void readDataset(Parameters* params, Dataset* fulldata, std::string filename);

void printDataset(Dataset *data);


Dataset* generateFolds(Dataset* data, int* indexesData, int* indexesDataInFolds);
void calculateDatasetsSize(Dataset* data, int* trainSize, int* validationSize, int* testSize);
void shuffleData(Dataset* data, int* indexesData, int* seed);
void getIndexes(int* indices, int k, int excludeIndex, int* seed);
Dataset* getSelectedDataset(Dataset* folds, int* indexes, int index_start, int index_end);


std::string ToString( double t );
bool IsPowerOf2( int n );
//unsigned short int NextPowerOf2( unsigned short int n );
unsigned int NextPowerOf2( unsigned short int n );
bool stopCriteria(unsigned long int it);

#endif //PCGP_UTILS_H
