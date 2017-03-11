

#ifndef MachineLearning_Dictionary_h
#define MachineLearning_Dictionary_h

#include "Setup.h"
#include "KMeans.h"
#include "Functions.h"

namespace TopicModels {
    
    template<class T>
    class Dictionary{
    public:
        Dictionary(T& inputData, int inputNumDictElements)
        : data(inputData),
        weight(static_cast<int>(inputData.rows()), inputNumDictElements),
        numDictElements(inputNumDictElements),
        dict(inputNumDictElements, static_cast<int>(inputData.cols())),
        numSamples(static_cast<int>(inputData.rows())){
            dict.fill(0.0f);
            weight.fill(0.0f);
        };
        virtual ~Dictionary(){};
        RMatrixXf& getDict(){return dict;};
        RMatrixXf& getWeight(){return weight;};
        void saveDict(std::string outputFile){
            saveMatrix<RMatrixXf&>(outputFile, dict);
        };
        void saveWeight(std::string outputFile){
            saveMatrix<RMatrixXf&>(outputFile, weight);
        };
        
    protected:
        void initializeKMeans(T& data, RMatrixXf& dict){
            int numSamples = static_cast<int>(data.rows());
            int numDictElements = static_cast<int>(dict.rows());
            RVectorXi kmInd(numSamples);
            int maxIter = 100;
            
            RMatrixXf dataf{data.template cast<float>()};
            kmeans(dataf, numDictElements, kmInd, maxIter, 0);
            
            dict.fill(0);
            for (int n : boost::irange(0, numSamples)){
                dict.row(kmInd[n]) += dataf.row(n);
            }
        };
        void initializeUniform(){
            int numSamples = static_cast<int>(data.rows());
            int numDictElements = static_cast<int>(dict.rows());
            // Use stl to create a random permutation.
            std::vector<int> rndPerm(numSamples);
            for (int i = 0; i < numSamples; ++i){
                rndPerm[i] = i;
            }
            std::srand(std::time(0));
            std::random_shuffle (rndPerm.begin(), rndPerm.end());
            
            for (int i = 0; i < numDictElements; ++i){
                dict.row(i) = data.row(rndPerm[i]).template cast<float>();
            }
        };
        void normalizeDictionary(){
            int numDictElements = static_cast<int>(dict.rows());
            for (int i = 0; i < numDictElements; ++i){
                float s = sqrt(dict.row(i)*dict.row(i).transpose());
                if (s > 0){
                    dict.row(i) = dict.row(i)/s;
                }
            }
        };
        
        T& data;
        RMatrixXf dict;
        RMatrixXf weight;
        
        int numSamples;
        int numDictElements;
    };
    
}

#endif
