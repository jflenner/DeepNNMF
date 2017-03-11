

#ifndef VideoTopicModels_Functions_h
#define VideoTopicModels_Functions_h

#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <list>
#include <cmath>
#include "boost/math/special_functions/digamma.hpp"
#include "Setup.h"
#include "RandomSimulation.h"


namespace TopicModels{
    
    void createDocList(const std::string inputFile, std::list<std::string>& docList){
        // Create the documents.
        std::ifstream myfile;
        myfile.open(inputFile.data());
        
        if (myfile.is_open()) {
            std::string tmpstring;
            while (!myfile.eof()) {
                std::getline(myfile, tmpstring);
                docList.push_back(tmpstring);
            }
            myfile.close();
            std::cout << "File " << inputFile << " read." << std::endl;
        }
        else{
            std::cout << "Could not open corpus file" << std::endl;
        }
    };
    
    void randomFidelity(const int numClasses, const int numFidelity, const RVectorXf& trueValues, RVectorXf& fidelity, RMatrixXf& fidelityValue){
        
        int numSamples = static_cast<int>(fidelity.cols());
        fidelity.fill(0.0f);
        fidelityValue.fill(0.0f);
        // Use stl to create a random permutation.
        std::vector<int> rndPerm(numSamples);
        for (int i = 0; i < numSamples; ++i){
            rndPerm[i] = i;
        }
        std::srand(std::time(0));
        std::random_shuffle (rndPerm.begin(), rndPerm.end());
        
//        std::cout << trueValues.cols() << std::endl;
//        std::cout << fidelityValue.rows() << std::endl;
//        std::cout << fidelityValue.cols() << std::endl;
//        std::cout << numSamples << std::endl;
        for (int n : boost::irange(0, numFidelity)){
//            std::cout << trueValues[rndPerm[n]] << std::endl;
            fidelity[rndPerm[n]] = 1.0f;
            fidelityValue.row(rndPerm[n])[trueValues[rndPerm[n]]] = 1.0f;
        }
        
    };
    
    void uniformFidelity(const int numClasses,
                         const int fidelitySkip,
                         const RVectorXf& trueValues,
                         RVectorXf& fidelity,
                         RMatrixXf& fidelityValue){
        
        int numSamples = static_cast<int>(fidelity.cols());
        fidelity.fill(0.0f);
        fidelityValue.fill(0.0f);
        
        for (int n = 0; n < numSamples; n += fidelitySkip){
            fidelity[n] = 1.0f;
            fidelityValue(n, trueValues[n]) = 1.0f;
        }
        
    };


    template<class C>
    class DataPartition{
    public:
        DataPartition(C& data){
            int numSamples = static_cast<int>(data.rows());
            numInPartition = numSamples / numProcessors;
            remainder = numSamples % numProcessors;
            numPartitions = numProcessors;
        };
        ~DataPartition(){};
        int getNumPartitions(){return numPartitions;};
        int getNumInPartition(){return numInPartition;};
        int getRemainder(){return remainder;};
        
    private:
        int numPartitions;
        int numInPartition;
        int remainder;
    };
    
    template<class T>
    void calcOuterProd(T vector1, RMatrixXf& result);
    
    template<class T>
    void saveMatrix(std::string outputFile, T matrix){
        std::ofstream myfile;
        myfile.open(outputFile.data());
        myfile << matrix << std::endl;
        myfile.close();
    }
    
    // Make sure you use matrices in template.
    template<class T>
    void loadMatrix(std::string inputFile, const int inputRows, const int inputColumns, T& matrix){
        matrix.resize(inputRows, inputColumns);
        matrix.fill(0);
        std::ifstream myfile;
        myfile.open(inputFile.data());
        
        for (int i = 0; i < inputRows; ++i){
            for (int j = 0; j < inputColumns; ++j){
                myfile >> matrix(i,j);
            }
        }
        
        myfile.close();
    }
    
    template<class T>
    void loadMatrix(std::string inputFile, T matrix){
        int inputRows;
        int inputCols;
        
        std::ifstream myfile;
        myfile.open(inputFile.data());
        
        myfile >> inputRows;
        myfile >> inputCols;
        
        matrix.resize(inputRows, inputCols);
        matrix.fill(0);
        for (int i : boost::irange(0, inputRows)){
            for (int j : boost::irange(0, inputCols)){
                myfile >> matrix(i,j);
            }
        }
        
        myfile.close();
    };
    
    template<class T>
    void loadVector(std::string inputFile, T matrix){
        int inputCols;
        
        std::ifstream myfile;
        myfile.open(inputFile.data());
        
        myfile >> inputCols;
        
        matrix.resize(inputCols);
        matrix.fill(0);
        for (int j : boost::irange(0, inputCols)){
            myfile >> matrix(j);
        }
        
        myfile.close();
    };
    
    template<class T>
    void normalizeL2(T matrix){
        
        for (int n : boost::irange(0, static_cast<int>(matrix.rows()))){
            float s = matrix.row(n).array().square().sum();
            s = sqrt(s);
            if (s > 0){
                matrix.row(n) = matrix.row(n)/s;
            }
        }
    };
    
    void subtractMean(RMatrixXf& matrix){
        float d = static_cast<float>(matrix.cols());
        
        for (int n : boost::irange(0, static_cast<int>(matrix.rows()))){
            matrix.row(n) = matrix.row(n).array() - matrix.row(n).array().sum()/d;
        }
    };
     
    float percentCorrect(const RVectorXf& predicted, const RVectorXf& correct){
        assert(predicted.cols() == correct.cols());
        float numCorrect(0);
        int N = static_cast<int>(predicted.cols());
        
        for (int n : boost::irange(0,N)){
            if (predicted(n) == correct(n)){
                ++numCorrect;
            }
        }
        
        return numCorrect/static_cast<float>(N);
    }
    
    template<class T>
    void calculateReconstructed(const RMatrixXf& dict, const RMatrixXf::RowXpr weight, T recVec){
        int numDictElements = static_cast<int>(dict.rows());
        // Loop through the dictionary elements.
        recVec.fill(0);
        for (int l = 0; l < numDictElements; ++l){
            recVec += dict.row(l)*(weight(l));
        }
    };
    
    template<class T>
    void calculateReconstructed(const RMatrixXf& dict, const RVectorXf& weight, T recVec){
        int numDictElements = static_cast<int>(dict.rows());
        // Loop through the dictionary elements.
        recVec.fill(0);
        for (int l = 0; l < numDictElements; ++l){
            recVec += dict.row(l)*(weight(l));
        }
    };

    void createCorrClss(const RVectorXi& inputCnt, RVectorXf& corrClss){
        
        int s{inputCnt.sum()};
        corrClss.resize(s);
        int base{0};
        for (int n : boost::irange(0, static_cast<int>(inputCnt.cols()) )){
            for (int k : boost::irange(0, inputCnt[n])){
                corrClss[base + k] = n;
            }
            base += inputCnt[n];
        }
    };
    
}

#endif
