//
// Created by bruno on 25/06/2020.
//

#ifndef P_CGPDE_OCLCONFIG_H
#define P_CGPDE_OCLCONFIG_H

#include "constants.h"
#include "utils.h"

class OCLConfig {
public:
    OCLConfig();

    float fitness[NUM_INDIV];
    float fitnessValidation[NUM_INDIV];

    std::vector<cl::Platform> platforms;
    std::vector<std::vector<cl::Device>> devices;

    std::vector<cl::ImageFormat> imageFormats;
    cl::size_t<3> origin;
    cl::size_t<3> imgSize;

    cl::Context context;
    cl::CommandQueue cmdQueue;
    cl::Program program;

    ///Evento para controlar tempo gasto
    cl_ulong inicio, fim;
    cl::Event e_tempo, e_tempo_train, e_tempo_valid, e_tempo_test;

    cl::Kernel testKernel;


    cl::Kernel kernelCGP;
    //cl::Kernel kernelCGPDE;
    //cl::Kernel kernelDE;

    cl::Kernel kernelTrain;
    cl::Kernel kernelValid;
    cl::Kernel kernelTest;

    cl::Kernel kernelTrainCompact;
    cl::Kernel kernelValidCompact;
    cl::Kernel kernelTestCompact;

    cl::Kernel kernelEvolve;

    cl::Kernel kernelEvaluate;
    cl::Kernel kernelEvaluateActive;

    cl::Kernel kernelEvaluateImage;
    cl::Kernel kernelEvaluateImageValidation;

    cl::Kernel kernelEvaluateImageHalf;
    cl::Kernel kernelEvaluateImageValidationHalf;

    cl::Kernel kernelEvaluateImageQuarter;
    cl::Kernel kernelEvaluateImageValidationQuarter;

    cl::Kernel kernelEvaluateImageCompact;
    cl::Kernel kernelEvaluateImageValidationCompact;

    cl::Kernel kernelEvaluateImageHalfCompact;
    cl::Kernel kernelEvaluateImageValidationHalfCompact;

    cl::Kernel kernelEvaluateImageQuarterCompact;
    cl::Kernel kernelEvaluateImageValidationQuarterCompact;

    ///Buffers
    cl::Buffer bufferSeeds;

    cl::Buffer bufferDataOut;

    cl::Buffer bufferDatasetTrain;
    cl::Buffer bufferOutputsTrain;

    cl::Buffer bufferDatasetValid;
    cl::Buffer bufferOutputsValid;

    cl::Buffer bufferDatasetTest;
    cl::Buffer bufferOutputsTest;

    cl::Buffer bufferFunctions;

    cl::Buffer bufferBest;
    cl::Buffer bufferPopulation;
    //cl::Buffer bufferPopulationActive;
    cl::Buffer bufferPopulationCompact;

    cl::Buffer bufferFitness;
    cl::Buffer bufferFitnessValidation;

    cl::Image2DArray populationImage;

    unsigned int * populationImageObject;
    unsigned int * populationImageObjectHalf;
    unsigned int * populationImageObjectQuarter;



    cl::ImageFormat image_format;
    cl_image_desc image_desc;

    size_t numPoints;
    size_t numPointsTrain;
    size_t numPointsValid;
    size_t numPointsTest;

    size_t maxLocalSize;

    size_t localSizeTrain;
    size_t localSizeValid;
    size_t localSizeTest;

    size_t globalSizeTrain;
    size_t globalSizeValid;
    size_t globalSizeTest;

    size_t localSizeAval;
    size_t globalSizeAval;

    size_t localSizeEvol;
    size_t globalSizeEvol;

    std::string compileFlags;

    cl_command_queue_properties commandQueueProperties;

    float* transposeDatasetOutput;

    float* transposeDatasetTrain;
    float* transposeOutputsTrain;

    float* transposeDatasetValid;
    float* transposeOutputsValid;

    float* transposeDatasetTest;
    float* transposeOutputsTest;




    void allocateBuffers(Parameters* p, int sizeTrain, int sizeValid, int sizeTest);
    void setNDRages();
    void setCompileFlags();
    void releaseAll();
    std::string setProgramSource(Parameters* p, Dataset* fullData);
    void buildProgram(Parameters* p, Dataset* fullData, std::string sourceFileStr);
    void buildKernels();
    //void writeReadOnlyBufers(Parameters* p, int* seeds);
    void writeReadOnlyBufers(Parameters* p);
    void transposeDatasets(Dataset* train, Dataset* valid, Dataset* test);
    void transposeDatasets(Dataset* train);

    void writeBestBuffer(Chromosome* best);
    void writePopulationBuffer(Chromosome* population);
    //void writePopulationActiveBuffer(ActiveChromosome* population);
    void writePopulationCompactBuffer(CompactChromosome* population);

    void readBestBuffer(Chromosome* best);
    void readPopulationBuffer(Chromosome* population);
    //void readPopulationActiveBuffer(ActiveChromosome* population);
    void readPopulationCompactBuffer(CompactChromosome* population);


    void readSeedsBuffer(int* seeds);
    void readFitnessBuffer();
    void readFitnessValidationBuffer();

    void setupImageBuffers();
    void setupImageBuffersHalf();
    void setupImageBuffersQuarter();
    void setupImageBuffersCompact();
    void setupImageBuffersHalfCompact();
    void setupImageBuffersQuarterCompact();


    void finishCommandQueue();

    void enqueueCGPKernel();

    void enqueueTrainKernel();
    void enqueueValidationKernel();
    void enqueueTestKernel();

    void enqueueTrainCompactKernel();
    void enqueueValidationCompactKernel();

    void enqueueEvolveKernel();

    void enqueueEvaluationKernel();
    void enqueueEvaluationActiveKernel();

    void enqueueEvaluationImageKernel();
    void enqueueEvaluationImageValidationKernel();

    void enqueueEvaluationImageHalfKernel();
    void enqueueEvaluationImageValidationHalfKernel();

    void enqueueEvaluationImageQuarterKernel();
    void enqueueEvaluationImageValidationQuarterKernel();

    void enqueueEvaluationImageCompactKernel();
    void enqueueEvaluationImageValidationCompactKernel();

    void enqueueEvaluationImageHalfCompactKernel();
    void enqueueEvaluationImageValidationHalfCompactKernel();

    void enqueueEvaluationImageQuarterCompactKernel();
    void enqueueEvaluationImageValidationQuarterCompactKernel();


    void writeImageBuffer(Chromosome* population);
    void writeImageBufferHalf(Chromosome* population);
    void writeImageBufferQuarter(Chromosome *population);
    void writeImageBufferCompact(Chromosome *population);
    void writeImageBufferHalfCompact(Chromosome *population);
    void writeImageBufferQuarterCompact(Chromosome *population);


    double getKernelElapsedTime();
    double getKernelElapsedTimeTrain();
    double getKernelElapsedTimeValid();
    double getKernelElapsedTimeTest();

    void compactChromosome(Chromosome* population, CompactChromosome* compactPopulation);


private:
    void printOpenclDeviceInfo();
    void checkError(cl_int result);
    void transposeData(Dataset* data, float** transposeDataset, float** transposeOutputs);
    void transposeDataOut(Dataset* data, float** transposeDatasetOutput);
    const char *getErrorString(cl_int error);




    unsigned int return_function_inputs_active(unsigned int function, unsigned int inputs, unsigned int active);
    unsigned int return_compact_inputs(unsigned int in0, unsigned int in1);
    unsigned int return_compact_inputs_weights(float in0, float in1);


};


#endif //P_CGPDE_OCLCONFIG_H
