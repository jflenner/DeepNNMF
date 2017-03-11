

#ifndef MachineLearning_CNN_h
#define MachineLearning_CNN_h

#include "Setup.h"
#include "RandomSimulation.h"
#include "Functions.h"
#include <thread>
#include <ctime>

namespace TopicModels {
    
    struct lowerToUpperInfo{
        int center{0};
        int shift{0};
        int mappedLoc{0};
        float newLpNrm{0};
    };
    typedef std::list<lowerToUpperInfo>* MapMatrix;  // The first int is the mapping to the next layer and the second int is the relationship to center pixel.
    typedef std::vector<MapMatrix> MapTensor;
    
    template<class T>
    T min(T in1, T in2){
        if (in1 < in2){
            return in1;
        }
        
        return in2;
    }
    
    template<class T>
    T max(T in1, T in2){
        if (in1 > in2){
            return in1;
        }
        
        return in2;
    }
    
    
    class DeepNetworkBase_1D{

    public:
        DeepNetworkBase_1D(RMatrixXf& inputData,
                               int inputNumLayers,
                               int inputPoolSize,
                               int inputPoolSkip,
                               int inputNumBasis,
                               int inputBasisDim)
        : data(inputData),
          numLayers(inputNumLayers),
          poolSize(inputPoolSize),
          poolSkip(inputPoolSkip),
          numBasis(inputNumBasis),
          numSamples(static_cast<int>(inputData.rows())),
          sValues(inputNumLayers),
          kValues(inputNumLayers),
          basisDim(inputBasisDim){};
        virtual ~DeepNetworkBase_1D(){
            for (auto maps : lowerToUpperMaps){
                if (maps != NULL){delete [] maps;}
            }
        };
        
    protected:
        // Pooling
        void maxPool(int l, RMatrixXf& weightPrevl){
            assert(l >= 0);
            
            RMatrixXf& datal = layerData[l+1];
            RMatrixXf& locMax = maxLocation[l+1];
            int N{static_cast<int>(weightPrevl.rows())};
            int s = sValues[l-1];
            
            for (int n : boost::irange(0, N)){
                int cnt{0};
                for (int p : boost::irange(0, numBasis)){
                    int baseLoc{p*s};
                    for (int h = poolSize; h < s - poolSize; h += poolSkip){
                        float m{0};
                        float mnew{0};
                        int lm{0};
                        for (int r : boost::irange(-poolSize, poolSize)){
                            mnew = max(std::abs(weightPrevl(n, baseLoc + h + r)), m);
                            if (mnew > m){
                                lm = baseLoc + h + r;
                                m = mnew;
                            }
                        }
                        
                        locMax(n, cnt) = lm;
                        datal(n, cnt) = m;
                        ++cnt;
                    }
                }
            };
        };
        void lpPool(int l, RMatrixXf& weightPrevl){
            assert(l >= 0);
            
            RMatrixXf& datal = layerData[l+1];
//            RMatrixXf& weightPrevl = weightsz[l];
            int N{static_cast<int>(weightPrevl.rows())};
            int s = sValues[l];
            
            for (int n : boost::irange(0, N)){
                int cnt{0};
                for (int k : boost::irange(0, numBasis)){
                    int baseLoc{k*s};
                    for (int h = poolSize; h < s - poolSize; h += poolSkip){
                        datal(n, cnt) = lpPool(weightPrevl, n, baseLoc, h);
                        ++cnt;
                    }
                }
            };
        }
        float lpPool(const RMatrixXf& weightPrevl, int n, int baseLoc, int h){
            float mnew{0};
            for (int r : boost::irange(-poolSize, poolSize)){
                mnew += std::pow(std::abs(weightPrevl(n, baseLoc + h + r)), pval);
            }
            mnew = std::pow(mnew, 1.0f/pval);
            
            return mnew;
        }
        Tensor layerData;    // The post pooling data.
        Tensor weights;      // A new set of weights per layer.
        MapTensor lowerToUpperMaps;  // Information on what lower level pixels rely on upper level values.
        RMatrixXf& data;
        RVectorXi sValues;   // Records the number of variables along the pool direction.
        RVectorXi kValues;   // Records the number of variables perp to the pool direction.
        Tensor maxLocation;  // The location of the maximums.
        Tensor basis;        // A new basis per layer.
        Tensor recon;        // The reconstruction at each level.
        Tensor residue;      // The residue per layer.        
        float pval{11};
        int numLayers;
        int poolSize;
        int poolSkip;
        int numBasis;
        int numSamples;
        int basisDim;
        int dataDim;
        int maxThreads{24};
    };
    
}


#endif
