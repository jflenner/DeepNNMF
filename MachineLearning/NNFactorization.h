

#ifndef NNFactorization_h
#define NNFactorization_h

#include "Setup.h"
#include "Dictionary.h"
#include "RandomSimulation.h"
#include "CNN.h"

namespace TopicModels {
    
    void multUpdate(const RMatrixXf& num, const RMatrixXf& denom, RMatrixXf& updateVar){
        updateVar.array() = updateVar.array()*(num.array()/denom.array()).array();
    };
    
    void regDenom(RMatrixXf& denom){
        float smallNum{0.0000000000001f};
        denom.array() = denom.array() + smallNum;
    };
    
    void addMissingLabelsRandom(RMatrixXf& classMatrix, RMatrixXf& knownLabels, float fracMissing){
        knownLabels.resize(classMatrix.rows(), classMatrix.cols());
        knownLabels.fill(1.0f);

        std::vector<int> rndPerm(classMatrix.rows());
        for (int i : boost::irange(0,static_cast<int>(classMatrix.rows()))){
            rndPerm[i] = i;
        }
        std::srand(std::time(0));
        std::random_shuffle(rndPerm.begin(), rndPerm.end());
        
        int numMissing{static_cast<int>(floor(fracMissing*static_cast<float>(classMatrix.rows())))};
        for (int k : boost::irange(0, numMissing)){
            knownLabels.row(rndPerm[k]) = RVectorXf::Zero(classMatrix.cols());
        }
    };
    
    void addMissingLabelsRandomBoundaries(const RVectorXi& boundaries, RMatrixXf& classMatrix, RMatrixXf& knownLabels, float fracMissing){
        knownLabels.resize(classMatrix.rows(), classMatrix.cols());
        knownLabels.fill(1.0f);
        std::vector<int> rndPerm(boundaries.cols()-1);
        for (int i : boost::irange(0,static_cast<int>(rndPerm.size()) )){
            rndPerm[i] = i;
        }
        std::srand(std::time(0));
        std::random_shuffle(rndPerm.begin(), rndPerm.end());
        
        int numMissing{static_cast<int>(floor(fracMissing*static_cast<float>(boundaries.cols())))};
        if (numMissing < 0) numMissing = 0;
        if (numMissing > rndPerm.size()){
            numMissing = static_cast<int>(rndPerm.size());
        }
        for (int i : boost::irange(0, numMissing)){
//            std::cout << rndPerm[i] << " " << boundaries.cols() << std::endl;            
            int begin{boundaries[rndPerm[i]]};
//            std::cout << rndPerm[i] << " " << boundaries.cols() << std::endl;            
            int end{boundaries[rndPerm[i]+1]};
            for (int j : boost::irange(begin, end)){
                knownLabels.row(j) = RVectorXf::Zero(classMatrix.cols());
            }
        }
    };
    
    class NNMF_Frob : public Dictionary<RMatrixXf>{
    public:
        NNMF_Frob(RMatrixXf& inputData, int inputNumDictElements)
        : Dictionary<RMatrixXf>(inputData, inputNumDictElements),
          recon(inputData.rows(), inputData.cols()),
          tmpNumWeight(inputData.rows(), inputNumDictElements),
          tmpDenomWeight(inputData.rows(), inputNumDictElements),
          tmpNumDict(inputNumDictElements, inputData.cols()),
          tmpDenomDict(inputNumDictElements, inputData.cols()){
            initializeUniform();
              normalizeDictionary();
        };
        ~NNMF_Frob(){};
        void run(int numiter){
            weight.fill(1.0f);
            for (int n : boost::irange(0, numiter)){
                updateWeight();
                updateDict();
            }
        };
        
    private:
        void updateWeight(){
            calcRecon();
            tmpNumDict = data*dict.transpose();
            tmpDenomDict = recon*dict.transpose();
            regDenom(tmpDenomDict);
            
            multUpdate(tmpNumDict, tmpDenomDict, weight);
        };
        void updateDict(){
            calcRecon();
            
            tmpNumWeight = weight.transpose()*data;
            tmpDenomWeight = weight.transpose()*recon;
            regDenom(tmpDenomWeight);
            
            multUpdate(tmpNumWeight, tmpDenomWeight, dict);
        };
        void calcRecon(){
            recon = weight*dict;
        };
        
        RMatrixXf recon;
        RMatrixXf tmpNumWeight;
        RMatrixXf tmpDenomWeight;
        RMatrixXf tmpNumDict;
        RMatrixXf tmpDenomDict;
    };
    
    void createClassBoundaries(const RMatrixXi& numInClass, RVectorXi& classBoundaries){
        classBoundaries.resize(numInClass.cols() + 1);
        
        classBoundaries[0] = 0;
        for (int n : boost::irange(0, static_cast<int>(numInClass.cols()))){
            classBoundaries[n+1] = classBoundaries[n] + numInClass(0,n);
        }
    };
    
    class SemiSupNNMF : public Dictionary<RMatrixXf>{
    public:
        SemiSupNNMF(RMatrixXf& inputData,
                    RMatrixXf& inputClassMatrix,
                    RMatrixXf& inputKnownLabels,
                    int inputNumDictElements,
                    float inputLambda = 100)
        : Dictionary<RMatrixXf>(inputData, inputNumDictElements),
          classMatrix(inputClassMatrix),
          knownLabels(inputKnownLabels),
          recon(inputData.rows(), inputData.cols()),
          tmpNumWeight(inputData.rows(), inputNumDictElements),
          tmpDenomWeight(inputData.rows(), inputNumDictElements),
          tmpNumDict(inputNumDictElements, inputData.cols()),
          tmpDenomDict(inputNumDictElements, inputData.cols()),
          lambda(inputLambda){
              labelDict = RMatrixXf::Random(inputNumDictElements, inputClassMatrix.cols());
              labelDict = labelDict.array().abs();
              classRecon = RMatrixXf::Zero(classMatrix.rows(), classMatrix.cols());
              tmpClassMatrix = RMatrixXf::Ones(classMatrix.rows(), classMatrix.cols());
            
              initializeUniform();
              normalizeDictionary();
        };
        ~SemiSupNNMF(){};
        void run(int numiter){
            weight.fill(1.0f);
            for (int n : boost::irange(0, numiter)){
                updateWeight();
                updateDataDict();
                updateLabelDict();
            }
        };
        RMatrixXf& getLabelRecon(){
            calcLabelRecon();
            return classRecon;
        };
        void saveAll(){
            calcDataRecon();
            calcLabelRecon();
            saveMatrix(machineLearningPath + "ssnmf_firstRecon.txt", recon);
            saveMatrix(machineLearningPath + "ssnmf_weight.txt", weight);
            saveMatrix(machineLearningPath + "ssnmf_dict.txt", dict);
            saveMatrix(machineLearningPath + "ssnmf_labelBasis.txt", labelDict);
            saveMatrix(machineLearningPath + "ssnmf_classMatrix.txt", classMatrix);
            saveMatrix(machineLearningPath + "ssnmf_knownLabels.txt", knownLabels);
            saveMatrix(machineLearningPath + "ssnmf_classRecon.txt", classRecon);
        }
        
    private:
        void updateWeight(){
            calcDataRecon();
            calcLabelRecon();
            
            tmpNumDict = data*dict.transpose();
            tmpDenomDict = recon*dict.transpose();
            tmpClassMatrix = classMatrix.array()*knownLabels.array();
            tmpNumDict += lambda*tmpClassMatrix*labelDict.transpose();
            tmpClassMatrix = weight*labelDict;
            tmpClassMatrix = tmpClassMatrix.array()*knownLabels.array();
            tmpDenomDict += lambda*tmpClassMatrix*labelDict.transpose();
            
            regDenom(tmpDenomDict);
            multUpdate(tmpNumDict, tmpDenomDict, weight);
        };
        void updateDataDict(){
            calcDataRecon();
            
            tmpNumWeight = weight.transpose()*data;
            tmpDenomWeight = weight.transpose()*recon;
            regDenom(tmpDenomWeight);
            
            multUpdate(tmpNumWeight, tmpDenomWeight, dict);
        };
        void updateLabelDict(){
            calcLabelRecon();
            
            tmpLBNum = weight.transpose()*classMatrix;
            tmpLBDenom = weight.transpose()*(weight*labelDict);
            regDenom(tmpLBDenom);
            
            multUpdate(tmpLBNum, tmpLBDenom, labelDict);
        };
        void calcLabelRecon(){
            classRecon = weight*labelDict;
        };
        void calcDataRecon(){
            recon = weight*dict;
        };
        RMatrixXf labelDict;
        RMatrixXf classMatrix;
        RMatrixXf knownLabels;
        RMatrixXf classRecon;
        RMatrixXf recon;
        RMatrixXf tmpNumWeight;
        RMatrixXf tmpDenomWeight;
        RMatrixXf tmpNumDict;
        RMatrixXf tmpDenomDict;
        RMatrixXf tmpClassMatrix;
        RMatrixXf tmpLBNum;
        RMatrixXf tmpLBDenom;
        float lambda;
    };
    
    class DeepNNMF : public DeepNetworkBase_1D{
    public:
        DeepNNMF(RMatrixXf& inputData,
                 RMatrixXf& inputClassMatrix,
                 RVectorXi& inputClassBoundaries,
                 int inputNumLayers,
                 int inputPoolSize,
                 int inputPoolSkip,
                 int inputNumBasis,
                 int inputLambda = 100)
        : DeepNetworkBase_1D(inputData, inputNumLayers, inputPoolSize, inputPoolSkip, inputNumBasis, static_cast<int>(inputData.cols())),
          classMatrix(inputClassMatrix),
          classBoundaries(inputClassBoundaries),
          lambda(inputLambda){
              weights.resize(numLayers);
              basis.resize(numLayers);
              layerData.resize(numLayers);
              maxLocation.resize(numLayers);
              recon.resize(numLayers);
              poolDeriv.resize(numLayers-1);
              tmpNumWeight.resize(numLayers);
              tmpDenomWeight.resize(numLayers);
              tmpNumDict.resize(numLayers);
              tmpDenomDict.resize(numLayers);
              int s{static_cast<int>(data.rows())};
              
              for (int l : boost::irange(0, numLayers)){
                  kValues[l] = numBasis;
              }
              tmpClassMatrix = RMatrixXf::Ones(classMatrix.rows(), classMatrix.cols());
              labelBasis = RMatrixXf::Random(kValues[numLayers-1], classMatrix.cols());
              knownLabels = RMatrixXf::Ones(classMatrix.rows(), classMatrix.cols());
              labelBasis = labelBasis.array().abs();
              labelRecon.resize(classMatrix.rows(), classMatrix.cols());
              
              for (int l : boost::irange(0, numLayers)){
                  sValues[l] = s;
                  
                  maxLocation[l] = RMatrixXf::Zero(s, kValues[l]);
                  weights[l] = RMatrixXf::Ones(s, kValues[l]);
                  if (l == 0){
                      layerData[l] = RMatrixXf::Zero(s, data.cols());
                      recon[l] = RMatrixXf::Zero(s, data.cols());
                  }
                  else{
                      layerData[l] = RMatrixXf::Zero(s, kValues[l]);
                      recon[l] = RMatrixXf::Zero(s, kValues[l]);
                  }
                  basis[l] = RMatrixXf::Ones(kValues[l], layerData[l].cols());
                  tmpNumWeight[l] = RMatrixXf::Zero(s, kValues[l]);
                  tmpDenomWeight[l] = RMatrixXf::Zero(s, kValues[l]);
                  tmpNumDict[l] = RMatrixXf::Zero(s, kValues[l]);
                  tmpDenomDict[l] = RMatrixXf::Zero(s, kValues[l]);
                  s = s/poolSkip;  // Must zero pad for pooling.
              }
              
              for (int l : boost::irange(0, numLayers-1)){
                  poolDeriv[l] = RMatrixXf::Zero(sValues[l], sValues[l+1]);
              }
              
              layerData[0] = data;
        };
        virtual ~DeepNNMF(){};
        void initialize(){
            weights.resize(numLayers);
            basis.resize(numLayers);
            layerData.resize(numLayers);
            maxLocation.resize(numLayers);
            recon.resize(numLayers);
            poolDeriv.resize(numLayers-1);
            tmpNumWeight.resize(numLayers);
            tmpDenomWeight.resize(numLayers);
            tmpNumDict.resize(numLayers);
            tmpDenomDict.resize(numLayers);
            int s{static_cast<int>(data.rows())};
            
            for (int l : boost::irange(0, numLayers)){
                kValues[l] = numBasis;
            }
            labelBasis = RMatrixXf::Random(kValues[numLayers-1], classMatrix.cols());
            knownLabels = RMatrixXf::Ones(classMatrix.rows(), classMatrix.cols());
            labelBasis = labelBasis.array().abs();
            labelRecon.resize(classMatrix.rows(), classMatrix.cols());
            
            for (int l : boost::irange(0, numLayers)){
                sValues[l] = s;
                
                maxLocation[l] = RMatrixXf::Zero(s, kValues[l]);
                weights[l] = RMatrixXf::Ones(s, kValues[l]);
                if (l == 0){
                    layerData[l] = RMatrixXf::Zero(s, data.cols());
                    recon[l] = RMatrixXf::Zero(s, data.cols());
                }
                else{
                    layerData[l] = RMatrixXf::Zero(s, kValues[l]);
                    recon[l] = RMatrixXf::Zero(s, kValues[l]);
                }
                basis[l] = RMatrixXf::Ones(kValues[l], layerData[l].cols());
                tmpNumWeight[l] = RMatrixXf::Zero(s, kValues[l]);
                tmpDenomWeight[l] = RMatrixXf::Zero(s, kValues[l]);
                tmpNumDict[l] = RMatrixXf::Zero(s, kValues[l]);
                tmpDenomDict[l] = RMatrixXf::Zero(s, kValues[l]);
                s = s/poolSkip;  // Must zero pad for pooling.
                //std::cout << "s = " << s << std::endl;
            }
            
            for (int l : boost::irange(0, numLayers-1)){
                poolDeriv[l] = RMatrixXf::Zero(sValues[l], sValues[l+1]);
            }
            
            layerData[0] = data;
            first = true;
        };
        void addMissingLabelsRandom(float fracMissing){
            knownLabels.fill(1.0f);
            std::srand(std::time(0));
            std::vector<int> rndPerm(classMatrix.rows());
            for (int i : boost::irange(0,static_cast<int>(classMatrix.rows()))){
                rndPerm[i] = i;
            }
            std::srand(std::time(0));
            std::random_shuffle(rndPerm.begin(), rndPerm.end());
            
            int numMissing{static_cast<int>(floor(fracMissing*static_cast<float>(classMatrix.rows())))};
            if (numMissing < 0) numMissing = 0;
            if (numMissing > rndPerm.size()) numMissing = rndPerm.size();
            for (int k : boost::irange(0, numMissing)){
                knownLabels.row(rndPerm[k]) = RVectorXf::Zero(classMatrix.cols());
            }
        };
        void addMissingLabelsRandomBoundaries(const RVectorXi& boundaries, float fracMissing){
            knownLabels.fill(1.0f);
            std::vector<int> rndPerm(boundaries.cols()-1);
            for (int i : boost::irange(0,static_cast<int>(rndPerm.size()) )){
                rndPerm[i] = i;
            }
            std::srand(std::time(0));
            std::random_shuffle(rndPerm.begin(), rndPerm.end());
            
            int numMissing{static_cast<int>(floor(fracMissing*static_cast<float>(boundaries.cols())))};
            if (numMissing < 0) numMissing = 0;
            if (numMissing > static_cast<int>(rndPerm.size())) numMissing = static_cast<int>(rndPerm.size());
            for (int i : boost::irange(0, numMissing)){
                int begin{boundaries[rndPerm[i]]};
                int end{boundaries[rndPerm[i]+1]};
                for (int j : boost::irange(begin, end)){
                    knownLabels.row(j) = RVectorXf::Zero(classMatrix.cols());
                }
            }
        };
        void forwardProp(int numiter){
            initializeRandDict(0);
            for (int n : boost::irange(0, numiter)){
                updateWeight(0);
                updateDict(0);
            }
            maxPool(0);
            calcRecon(0);
            
            for (int layer : boost::irange(1, numLayers)){
                initializeRandDict(layer);
                for (int n : boost::irange(0, numiter)){
                    updateWeight(layer);
                    updateDict(layer);
                }
                if (layer < numLayers - 1){
                    maxPool(layer);
                }
                calcRecon(layer);
            }
            updateLabelBasis();
            first = false;
        };
        void backProp(int numiter){
            if (first == true){
                std::cout << "Must Forward Propigate to Initialize" << std::endl;
            }
            else{
                for (int n : boost::irange(0, numiter)){
                    
                    for (int layer : boost::irange(0, numLayers)){
                        updateWeight(layer);
                        updateDict(layer);
                        if (layer < numLayers - 1){
                            maxPool(layer);
                        }
                    }
                    updateLabelBasis();
                }
            }

        };
        RMatrixXf& getLabelRecon(){
            labelRecon = weights[numLayers - 1]*labelBasis;
            return labelRecon;
        };
        void saveAll(std::string addstring, int layer){
            assert(0 <= layer < numLayers);
            calcRecon(0);
            calcRecon(layer);
            saveMatrix(machineLearningPath + addstring + "firstRecon.txt", layerData[0]);
            saveMatrix(machineLearningPath + addstring + "pooled.txt", layerData[layer]);
            saveMatrix(machineLearningPath + addstring + "weight.txt", weights[layer]);
            saveMatrix(machineLearningPath + addstring + "dict.txt", basis[layer]);
            saveMatrix(machineLearningPath + addstring + "recon.txt", recon[layer]);
            saveMatrix(machineLearningPath + addstring + "labelBasis.txt", labelBasis);
            saveMatrix(machineLearningPath + addstring + "classMatrix.txt", classMatrix);
            saveMatrix(machineLearningPath + addstring + "knownLabels.txt", knownLabels);
        };
        void saveBackRecon(){
            RMatrixXf br;
            backRecon(br);
            saveMatrix(machineLearningPath + "backRecon.txt", br);
        };
        
    private:
        void backRecon(RMatrixXf& output){
            output.resize(data.rows(), data.cols());
            
            calcRecon(numLayers - 1);
            for (int l = numLayers - 1; l > 0; --l){
                tmpNumDict[l-1].fill(0);
                for (int k : boost::irange(0, kValues[l-1])){
                    calcMaxPoolDeriv(l-1, k);
                    tmpNumDict[l-1].col(k) += poolDeriv[l-1]*recon[l].col(k);
                }
                tmpNumDict[l-1] += weights[l-1];
//                std::cout << weights[l-1].cols() << std::endl;
//                std::cout << weights[l-1].rows() << std::endl;
//                std::cout << tmpNumDict[l-1].cols() << std::endl;
//                std::cout << tmpNumDict[l-1].rows() << std::endl;
//                std::cout << recon[l-1].cols() << std::endl;
//                std::cout << recon[l-1].rows() << std::endl;
                recon[l-1] = tmpNumDict[l-1]*basis[l-1];
            }

            output = recon[0];
        };
        void initializeRandDict(int layer){
            std::vector<int> rndPerm(sValues[layer]);
            for (int i = 0; i < sValues[layer]; ++i){
                rndPerm[i] = i;
            }
            std::srand(std::time(0));
            std::random_shuffle(rndPerm.begin(), rndPerm.end());
            
            for (int k : boost::irange(0, kValues[layer])){
                basis[layer].row(k) = layerData[layer].row(rndPerm[k] );
            }
            
        }
        void normalizeDict(int layer){
            for (int k : boost::irange(0, kValues[layer])){
                float tmp = basis[layer].row(k).dot(basis[layer].row(k));
                basis[layer].row(k) = basis[layer].row(k)/sqrt(tmp);
            }
        }
        void maxPool(int layer){
            assert(layer >= 0);
            assert(layer < numLayers);
            
            int s = sValues[layer];
            
            // Pool along time or the samples.
            for (int k : boost::irange(0, numBasis)){
                int cnt{0};
                for (int l : boost::irange(0, static_cast<int>(classBoundaries.cols() - 1) )){
                    for (int h = classBoundaries[l]; h < classBoundaries[l+1]; h += poolSkip){
                        float m{0};
                        float mnew{0};
                        int lm{0};
                        if (poolSize == 0){
                            lm = h;
                            m = std::abs(weights[layer](h, k));
                        }
                        else{
                            for (int r : boost::irange(-poolSize, poolSize)){
                                if (h + r >= classBoundaries[l] && h + r < classBoundaries[l+1]){
                                    mnew = max(std::abs(weights[layer](h + r, k)), m);
                                }
                                if (mnew > m){
                                    lm = h + r;
                                    m = mnew;
                                }
                            }
                        }
                        
                        maxLocation[layer + 1](cnt, k) = lm;
                        layerData[layer+1](cnt, k) = m;
                        ++cnt;
                    }
                }
            };
        };
        void calcMaxPoolDeriv(int layer, int basis){
            assert(layer < numLayers - 1);
            poolDeriv[layer].fill(0.0f);
            if (!first){
                for (int n : boost::irange(0, sValues[layer + 1])){
                    poolDeriv[layer](maxLocation[layer+1](n, basis), n) = 1.0f;
                }
            }
        };
        void updateWeight(int layer){
            calcRecon(layer);
            tmpNumDict[layer] = layerData[layer]*basis[layer].transpose();
            tmpDenomDict[layer] = recon[layer]*basis[layer].transpose();
            if (layer != numLayers - 1){
                calcRecon(layer + 1);
                for (int k : boost::irange(0, kValues[layer])){
                    calcMaxPoolDeriv(layer, k);
                    tmpNumDict[layer].col(k) += poolDeriv[layer]*recon[layer+1].col(k);
                    tmpDenomDict[layer].col(k) += poolDeriv[layer]*layerData[layer+1].col(k);
                }
            }
            else{
                tmpClassMatrix = classMatrix.array()*knownLabels.array();
                tmpNumDict[layer] += lambda*tmpClassMatrix*labelBasis.transpose();
                tmpClassMatrix = weights[numLayers - 1]*labelBasis;
                tmpClassMatrix = tmpClassMatrix.array()*knownLabels.array();
                tmpDenomDict[layer] += lambda*tmpClassMatrix*labelBasis.transpose();
            }
            regDenom(tmpDenomDict[layer]);
            
            multUpdate(tmpNumDict[layer], tmpDenomDict[layer], weights[layer]);
        };
        void updateDict(int layer){
            calcRecon(layer);
            
            tmpNumWeight[layer] = weights[layer].transpose()*layerData[layer];
            tmpDenomWeight[layer] = weights[layer].transpose()*recon[layer];
            regDenom(tmpDenomWeight[layer]);
            
            multUpdate(tmpNumWeight[layer], tmpDenomWeight[layer], basis[layer]);
        };
        void updateLabelBasis(){
            
            tmpLBNum = weights[numLayers - 1].transpose()*classMatrix;
            tmpLBDenom = weights[numLayers - 1].transpose()*(weights[numLayers - 1]*labelBasis);
            regDenom(tmpLBDenom);
            
            multUpdate(tmpLBNum, tmpLBDenom, labelBasis);
        };
        void calcRecon(int layer){
            recon[layer] = weights[layer]*basis[layer];
        };
        Tensor poolDeriv;
        Tensor tmpNumWeight;
        Tensor tmpDenomWeight;
        Tensor tmpNumDict;
        Tensor tmpDenomDict;
        RMatrixXf labelRecon;
        RMatrixXf tmpClassMatrix;
        RMatrixXf knownLabels;
        RMatrixXf tmpLBNum;
        RMatrixXf tmpLBDenom;
        RMatrixXf labelBasis;
        RMatrixXf classMatrix;
        RVectorXi classBoundaries;
        float lambda;
        bool first{true};
    };
    
    class DeepNNMF_NoPool : public DeepNetworkBase_1D{
    public:
        DeepNNMF_NoPool(RMatrixXf& inputData,
                 RMatrixXf& inputClassMatrix,
                 RVectorXi& inputClassBoundaries,
                 int inputNumLayers,
                 int inputNumBasis,
                 int inputLambda = 100)
        : DeepNetworkBase_1D(inputData, inputNumLayers, 1, 1, inputNumBasis, static_cast<int>(inputData.cols())),
        classMatrix(inputClassMatrix),
        classBoundaries(inputClassBoundaries),
        lambda(inputLambda){
            weights.resize(numLayers);
            basis.resize(numLayers);
            layerData.resize(numLayers);
            maxLocation.resize(numLayers);
            recon.resize(numLayers);
            tmpNumWeight.resize(numLayers);
            tmpDenomWeight.resize(numLayers);
            tmpNumDict.resize(numLayers);
            tmpDenomDict.resize(numLayers);
            int s{static_cast<int>(data.rows())};
            
            for (int l : boost::irange(0, numLayers)){
                kValues[l] = numBasis;
            }
            tmpClassMatrix = RMatrixXf::Ones(classMatrix.rows(), classMatrix.cols());
            labelBasis = RMatrixXf::Random(kValues[numLayers-1], classMatrix.cols());
            knownLabels = RMatrixXf::Ones(classMatrix.rows(), classMatrix.cols());
            labelBasis = labelBasis.array().abs();
            labelRecon.resize(classMatrix.rows(), classMatrix.cols());
            
            for (int l : boost::irange(0, numLayers)){
                sValues[l] = s;
                
                maxLocation[l] = RMatrixXf::Zero(s, kValues[l]);
                weights[l] = RMatrixXf::Ones(s, kValues[l]);
                if (l == 0){
                    layerData[l] = RMatrixXf::Zero(s, data.cols());
                    recon[l] = RMatrixXf::Zero(s, data.cols());
                }
                else{
                    layerData[l] = RMatrixXf::Zero(s, kValues[l]);
                    recon[l] = RMatrixXf::Zero(s, kValues[l]);
                }
                basis[l] = RMatrixXf::Ones(kValues[l], layerData[l].cols());
                tmpNumWeight[l] = RMatrixXf::Zero(s, kValues[l]);
                tmpDenomWeight[l] = RMatrixXf::Zero(s, kValues[l]);
                tmpNumDict[l] = RMatrixXf::Zero(s, kValues[l]);
                tmpDenomDict[l] = RMatrixXf::Zero(s, kValues[l]);
                s = s/poolSkip;  // Must zero pad for pooling.
            }
            
            layerData[0] = data;
        };
        ~DeepNNMF_NoPool(){};
        void addMissingLabelsRandom(float fracMissing){
            knownLabels.fill(1.0f);
            std::srand(std::time(0));
            std::vector<int> rndPerm(classMatrix.rows());
            for (int i : boost::irange(0,static_cast<int>(classMatrix.rows()))){
                rndPerm[i] = i;
            }
            std::srand(std::time(0));
            std::random_shuffle(rndPerm.begin(), rndPerm.end());
            
            int numMissing{static_cast<int>(floor(fracMissing*static_cast<float>(classMatrix.rows())))};
            for (int k : boost::irange(0, numMissing)){
                knownLabels.row(rndPerm[k]) = RVectorXf::Zero(classMatrix.cols());
            }
        };
        void addMissingLabelsRandomBoundaries(const RVectorXi& boundaries, float fracMissing){
            knownLabels.fill(1.0f);
            std::vector<int> rndPerm(boundaries.cols()-1);
            for (int i : boost::irange(0,static_cast<int>(boundaries.cols()) )){
                rndPerm[i] = i;
            }
            std::srand(std::time(0));
            std::random_shuffle(rndPerm.begin(), rndPerm.end());
            
            int numMissing{static_cast<int>(floor(fracMissing*static_cast<float>(boundaries.cols())))};
            for (int i : boost::irange(0, numMissing)){
                int begin{boundaries[rndPerm[i]]};
                int end{boundaries[rndPerm[i]+1]};
                for (int j : boost::irange(begin, end)){
                    knownLabels.row(j) = RVectorXf::Zero(classMatrix.cols());
                }
            }
        };
        void setLayerData(int layer){
            layerData[layer+1] = weights[layer];
        };
        void forwardProp(int numiter){
            initializeRandDict(0);
            for (int n : boost::irange(0, numiter)){
                updateWeight(0);
                updateDict(0);
            }
            setLayerData(0);
            calcRecon(0);
            
            for (int layer : boost::irange(1, numLayers)){
                initializeRandDict(layer);
                for (int n : boost::irange(0, numiter)){
                    updateWeight(layer);
                    updateDict(layer);
                }
                if (layer < numLayers - 1){
                    setLayerData(layer);
                }
                calcRecon(layer);
            }
            updateLabelBasis();
            first = false;
        };
        void backProp(int numiter){
            if (first == true){
                std::cout << "Must Forward Propigate to Initialize" << std::endl;
            }
            else{
                for (int n : boost::irange(0, numiter)){
                    
                    for (int layer : boost::irange(0, numLayers)){
                        updateWeight(layer);
                        updateDict(layer);
                        if (layer < numLayers - 1){
                            setLayerData(layer);
                        }
                    }
                    updateLabelBasis();
                }
            }
            
        };
        RMatrixXf& getLabelRecon(){
            labelRecon = weights[numLayers - 1]*labelBasis;
            return labelRecon;
        };
        void saveAll(int layer){
            assert(0 <= layer < numLayers);
            calcRecon(0);
            calcRecon(layer);
            saveMatrix(machineLearningPath + "firstRecon.txt", layerData[0]);
            saveMatrix(machineLearningPath + "pooled.txt", layerData[layer]);
            saveMatrix(machineLearningPath + "weight.txt", weights[layer]);
            saveMatrix(machineLearningPath + "dict.txt", basis[layer]);
            saveMatrix(machineLearningPath + "recon.txt", recon[layer]);
            saveMatrix(machineLearningPath + "labelBasis.txt", labelBasis);
            saveMatrix(machineLearningPath + "classMatrix.txt", classMatrix);
            saveMatrix(machineLearningPath + "knownLabels.txt", knownLabels);
        };
        void saveBackRecon(){
            RMatrixXf br;
            backRecon(br);
            saveMatrix(machineLearningPath + "backRecon.txt", br);
        };
        
    private:
        void backRecon(RMatrixXf& output){
            output.resize(data.rows(), data.cols());
            
            calcRecon(numLayers - 1);
            for (int l = numLayers - 1; l > 0; --l){
                tmpNumDict[l-1].fill(0);
                for (int k : boost::irange(0, kValues[l-1])){
                    calcMaxPoolDeriv(l-1, k);
                    tmpNumDict[l-1].col(k) += recon[l].col(k);
                }
                tmpNumDict[l-1] += weights[l-1];
                recon[l-1] = tmpNumDict[l-1]*basis[l-1];
            }
            
            output = recon[0];
        };
        void initializeRandDict(int layer){
            std::vector<int> rndPerm(sValues[layer]);
            for (int i = 0; i < sValues[layer]; ++i){
                rndPerm[i] = i;
            }
            std::srand(std::time(0));
            std::random_shuffle(rndPerm.begin(), rndPerm.end());
            
            for (int k : boost::irange(0, kValues[layer])){
                basis[layer].row(k) = layerData[layer].row(rndPerm[k] );
            }
            
        }
        void normalizeDict(int layer){
            for (int k : boost::irange(0, kValues[layer])){
                float tmp = basis[layer].row(k).dot(basis[layer].row(k));
                basis[layer].row(k) = basis[layer].row(k)/sqrt(tmp);
            }
        }
        void calcMaxPoolDeriv(int layer, int basis){
            
        };
        void updateWeight(int layer){
            calcRecon(layer);
            tmpNumDict[layer] = layerData[layer]*basis[layer].transpose();
            tmpDenomDict[layer] = recon[layer]*basis[layer].transpose();
            if (layer != numLayers - 1){
                calcRecon(layer + 1);
                for (int k : boost::irange(0, kValues[layer])){
                    calcMaxPoolDeriv(layer, k);
                    tmpNumDict[layer].col(k) += recon[layer+1].col(k);
                    tmpDenomDict[layer].col(k) += layerData[layer+1].col(k);
                }
            }
            else{
                tmpClassMatrix = classMatrix.array()*knownLabels.array();
                tmpNumDict[layer] += lambda*tmpClassMatrix*labelBasis.transpose();
                tmpClassMatrix = weights[numLayers - 1]*labelBasis;
                tmpClassMatrix = tmpClassMatrix.array()*knownLabels.array();
                tmpDenomDict[layer] += lambda*tmpClassMatrix*labelBasis.transpose();
            }
            regDenom(tmpDenomDict[layer]);
            
            multUpdate(tmpNumDict[layer], tmpDenomDict[layer], weights[layer]);
        };
        void updateDict(int layer){
            calcRecon(layer);
            
            tmpNumWeight[layer] = weights[layer].transpose()*layerData[layer];
            tmpDenomWeight[layer] = weights[layer].transpose()*recon[layer];
            regDenom(tmpDenomWeight[layer]);
            
            multUpdate(tmpNumWeight[layer], tmpDenomWeight[layer], basis[layer]);
        };
        void updateLabelBasis(){
            
            tmpLBNum = weights[numLayers - 1].transpose()*classMatrix;
            tmpLBDenom = weights[numLayers - 1].transpose()*(weights[numLayers - 1]*labelBasis);
            regDenom(tmpLBDenom);
            
            multUpdate(tmpLBNum, tmpLBDenom, labelBasis);
        };
        void calcRecon(int layer){
            recon[layer] = weights[layer]*basis[layer];
        };
        Tensor tmpNumWeight;
        Tensor tmpDenomWeight;
        Tensor tmpNumDict;
        Tensor tmpDenomDict;
        RMatrixXf labelRecon;
        RMatrixXf tmpClassMatrix;
        RMatrixXf knownLabels;
        RMatrixXf tmpLBNum;
        RMatrixXf tmpLBDenom;
        RMatrixXf labelBasis;
        RMatrixXf classMatrix;
        RVectorXi classBoundaries;
        float lambda;
        bool first{true};
    };
    
    class DeepNNMF_AllSup : public DeepNetworkBase_1D{
    public:
        DeepNNMF_AllSup(RMatrixXf& inputData,
                 RMatrixXf& inputClassMatrix,
                 RVectorXi& inputClassBoundaries,
                 int inputNumLayers,
                 int inputPoolSize,
                 int inputPoolSkip,
                 int inputNumBasis,
                 int inputLambda = 100)
        : DeepNetworkBase_1D(inputData, inputNumLayers, inputPoolSize, inputPoolSkip, inputNumBasis, static_cast<int>(inputData.cols())),
        classMatrix(inputClassMatrix),
        classBoundaries(inputClassBoundaries),
        lambda(inputLambda){
            weights.resize(numLayers);
            basis.resize(numLayers);
            layerData.resize(numLayers);
            maxLocation.resize(numLayers);
            recon.resize(numLayers);
            poolDeriv.resize(numLayers-1);
            tmpNumWeight.resize(numLayers);
            tmpDenomWeight.resize(numLayers);
            tmpNumDict.resize(numLayers);
            tmpDenomDict.resize(numLayers);
            labelBasis.resize(numLayers);
            knownLabels.resize(numLayers);
            labelRecon.resize(numLayers);
            
            int s{static_cast<int>(data.rows())};
            
            for (int l : boost::irange(0, numLayers)){
                kValues[l] = numBasis;
            }
            
            for (int l : boost::irange(0, numLayers)){
                sValues[l] = s;
                
                maxLocation[l] = RMatrixXf::Zero(s, kValues[l]);
                weights[l] = RMatrixXf::Ones(s, kValues[l]);
                if (l == 0){
                    layerData[l] = RMatrixXf::Zero(s, data.cols());
                    recon[l] = RMatrixXf::Zero(s, data.cols());
                }
                else{
                    layerData[l] = RMatrixXf::Zero(s, kValues[l]);
                    recon[l] = RMatrixXf::Zero(s, kValues[l]);
                }
                basis[l] = RMatrixXf::Ones(kValues[l], layerData[l].cols());
                tmpNumWeight[l] = RMatrixXf::Zero(s, kValues[l]);
                tmpDenomWeight[l] = RMatrixXf::Zero(s, kValues[l]);
                tmpNumDict[l] = RMatrixXf::Zero(s, kValues[l]);
                tmpDenomDict[l] = RMatrixXf::Zero(s, kValues[l]);
                labelBasis[l] = RMatrixXf::Random(kValues[l], classMatrix.cols());
                labelBasis[l] = labelBasis[l].array().abs();
                labelRecon[l] = RMatrixXf::Zero(classMatrix.rows(), classMatrix.cols());
                knownLabels[l] = RMatrixXf::Ones(classMatrix.rows(), classMatrix.cols());
                s = s/poolSkip;  // Must zero pad for pooling.
            }
            
            for (int l : boost::irange(0, numLayers-1)){
                poolDeriv[l] = RMatrixXf::Zero(sValues[l], sValues[l+1]);
            }
            
            layerData[0] = data;
        };
        ~DeepNNMF_AllSup(){};
        void addMissingLabelsRandom(float fracMissing){
            for (int l : boost::irange(0, numLayers)){
                knownLabels[l].fill(1.0f);
            }
            std::srand(std::time(0));
            std::vector<int> rndPerm(classMatrix.rows());
            for (int i : boost::irange(0,static_cast<int>(classMatrix.rows()))){
                rndPerm[i] = i;
            }

            std::random_shuffle(rndPerm.begin(), rndPerm.end());
            
            int numMissing{static_cast<int>(floor(fracMissing*static_cast<float>(classMatrix.rows())))};
            for (int k : boost::irange(0, numMissing)){
                for (int l : boost::irange(0, numLayers)){
                    knownLabels[l].row(rndPerm[k]) = RVectorXf::Zero(classMatrix.cols());
                }
            }
        };
        void addMissingLabelsRandomBoundaries(const RVectorXi& boundaries, float fracMissing){
            for (int l : boost::irange(0, numLayers)){
                knownLabels[l].fill(1.0f);
            }
            std::vector<int> rndPerm(boundaries.cols()-1);
            for (int i : boost::irange(0,static_cast<int>(boundaries.cols()) )){
                rndPerm[i] = i;
            }
            std::srand(std::time(0));
            std::random_shuffle(rndPerm.begin(), rndPerm.end());
            
            int numMissing{static_cast<int>(floor(fracMissing*static_cast<float>(boundaries.cols())))};
            for (int i : boost::irange(0, numMissing)){
                int begin{boundaries[rndPerm[i]]};
                int end{boundaries[rndPerm[i]+1]};
                for (int j : boost::irange(begin, end)){
                    for (int l : boost::irange(0, numLayers)){
                        knownLabels[l].row(j) = RVectorXf::Zero(classMatrix.cols());
                    }
                }
            }
        };
        void forwardProp(int numiter){
            initializeRandDict(0);
            for (int n : boost::irange(0, numiter)){
                updateWeight(0);
                updateDict(0);
            }
            maxPool(0);
            calcRecon(0);
            
            for (int layer : boost::irange(1, numLayers)){
                initializeRandDict(layer);
                for (int n : boost::irange(0, numiter)){
                    updateWeight(layer);
                    updateDict(layer);
                }
                if (layer < numLayers - 1){
                    maxPool(layer);
                }
                calcRecon(layer);
                updateLabelBasis(layer);
            }
            first = false;
        };
        void backProp(int numiter){
            if (first == true){
                std::cout << "Must Forward Propigate to Initialize" << std::endl;
            }
            else{
                for (int n : boost::irange(0, numiter)){
                    
                    for (int layer : boost::irange(0, numLayers)){
                        updateWeight(layer);
                        updateDict(layer);
                        if (layer < numLayers - 1){
                            maxPool(layer);
                        }
                    }
                }
            }
            
        };
        RMatrixXf& getLabelRecon(int l){
            labelRecon[l] = weights[l]*labelBasis[l];
            return labelRecon[l];
        };
        void saveAll(int layer){
            assert(0 <= layer < numLayers);
            calcRecon(0);
            calcRecon(layer);
            saveMatrix(machineLearningPath + "firstRecon.txt", layerData[0]);
            saveMatrix(machineLearningPath + "pooled.txt", layerData[layer]);
            saveMatrix(machineLearningPath + "weight.txt", weights[layer]);
            saveMatrix(machineLearningPath + "dict.txt", basis[layer]);
            saveMatrix(machineLearningPath + "recon.txt", recon[layer]);
            saveMatrix(machineLearningPath + "labelBasis.txt", labelBasis[layer]);
            saveMatrix(machineLearningPath + "classMatrix.txt", classMatrix);
            saveMatrix(machineLearningPath + "knownLabels.txt", knownLabels[layer]);
        };
        void saveBackRecon(){
            RMatrixXf br;
            backRecon(br);
            saveMatrix(machineLearningPath + "backRecon.txt", br);
        };
        
    private:
        void backRecon(RMatrixXf& output){
            output.resize(data.rows(), data.cols());
            
            calcRecon(numLayers - 1);
            for (int l = numLayers - 1; l > 0; --l){
                tmpNumDict[l-1].fill(0);
                for (int k : boost::irange(0, kValues[l-1])){
                    calcMaxPoolDeriv(l-1, k);
                    tmpNumDict[l-1].col(k) += poolDeriv[l-1]*recon[l].col(k);
                }
                //                std::cout << weights[l-1].cols() << std::endl;
                //                std::cout << weights[l-1].rows() << std::endl;
                //                std::cout << tmpNumDict[l-1].cols() << std::endl;
                //                std::cout << tmpNumDict[l-1].rows() << std::endl;
                //                std::cout << recon[l-1].cols() << std::endl;
                //                std::cout << recon[l-1].rows() << std::endl;
                recon[l-1] = tmpNumDict[l-1]*basis[l-1];
            }
            
            output = recon[0];
        };
        void initializeRandDict(int layer){
            std::vector<int> rndPerm(sValues[layer]);
            for (int i = 0; i < sValues[layer]; ++i){
                rndPerm[i] = i;
            }
            std::srand(std::time(0));
            std::random_shuffle(rndPerm.begin(), rndPerm.end());
            
            for (int k : boost::irange(0, kValues[layer])){
                basis[layer].row(k) = layerData[layer].row(rndPerm[k] );
            }
            
        }
        void normalizeDict(int layer){
            for (int k : boost::irange(0, kValues[layer])){
                float tmp = basis[layer].row(k).dot(basis[layer].row(k));
                basis[layer].row(k) = basis[layer].row(k)/sqrt(tmp);
            }
        }
        void maxPool(int layer){
            assert(layer >= 0);
            assert(layer < numLayers - 1);
            
            int s = sValues[layer];
            
            // Pool along time or the samples.
            for (int k : boost::irange(0, numBasis)){
                int cnt{0};
                for (int l : boost::irange(0, static_cast<int>(classBoundaries.cols() - 1) )){
                    for (int h = classBoundaries[l]; h < classBoundaries[l+1]; h += poolSkip){
                        float m{0};
                        float mnew{0};
                        int lm{0};
                        for (int r : boost::irange(-poolSize, poolSize)){
                            if (h + r >= classBoundaries[l] && h + r < classBoundaries[l+1]){
                                mnew = max(std::abs(weights[layer](h + r, k)), m);
                            }
                            if (mnew > m){
                                lm = h + r;
                                m = mnew;
                            }
                        }
                        
                        maxLocation[layer + 1](cnt, k) = lm;
                        layerData[layer+1](cnt, k) = m;
                        ++cnt;
                    }
                }
            };
        };
        void calcMaxPoolDeriv(int layer, int basis){
            assert(layer < numLayers - 1);
            poolDeriv[layer].fill(0.0f);
            if (!first){
                for (int n : boost::irange(0, sValues[layer + 1])){
                    poolDeriv[layer](maxLocation[layer+1](n, basis), n) = 1.0f;
                }
            }
        };
        void updateWeight(int layer){
            calcRecon(layer);
            tmpNumDict[layer] = layerData[layer]*basis[layer].transpose();
            tmpDenomDict[layer] = recon[layer]*basis[layer].transpose();
            if (layer != numLayers - 1){
                calcRecon(layer + 1);
                for (int k : boost::irange(0, kValues[layer])){
                    calcMaxPoolDeriv(layer, k);
                    tmpNumDict[layer].col(k) += poolDeriv[layer]*recon[layer+1].col(k);
                    tmpDenomDict[layer].col(k) += poolDeriv[layer]*layerData[layer+1].col(k);
                }
            }
            else{
                tmpClassMatrix[layer] = classMatrix.array()*knownLabels[layer].array();
                tmpNumDict[layer] += lambda*tmpClassMatrix[layer]*labelBasis[layer].transpose();
                tmpClassMatrix[layer] = weights[numLayers - 1]*labelBasis[layer];
                tmpClassMatrix[layer] = tmpClassMatrix[layer].array()*knownLabels[layer].array();
                tmpDenomDict[layer] += lambda*tmpClassMatrix[layer]*labelBasis[layer].transpose();
            }
            regDenom(tmpDenomDict[layer]);
            
            multUpdate(tmpNumDict[layer], tmpDenomDict[layer], weights[layer]);
        };
        void updateDict(int layer){
            calcRecon(layer);
            
            tmpNumWeight[layer] = weights[layer].transpose()*layerData[layer];
            tmpDenomWeight[layer] = weights[layer].transpose()*recon[layer];
            regDenom(tmpDenomWeight[layer]);
            
            multUpdate(tmpNumWeight[layer], tmpDenomWeight[layer], basis[layer]);
        };
        void updateLabelBasis(int layer){
            
            tmpLBNum[layer] = weights[layer].transpose()*classMatrix;
            tmpLBDenom[layer] = weights[layer].transpose()*(weights[layer]*labelBasis[layer]);
            regDenom(tmpLBDenom[layer]);
            
            multUpdate(tmpLBNum[layer], tmpLBDenom[layer], labelBasis[layer]);
        };
        void calcRecon(int layer){
            recon[layer] = weights[layer]*basis[layer];
        };
        Tensor poolDeriv;
        Tensor tmpNumWeight;
        Tensor tmpDenomWeight;
        Tensor tmpNumDict;
        Tensor tmpDenomDict;
        Tensor labelBasis;
        Tensor labelRecon;
        Tensor tmpClassMatrix;
        Tensor knownLabels;
        Tensor tmpLBNum;
        Tensor tmpLBDenom;
        RMatrixXf classMatrix;
        RVectorXi classBoundaries;
        float lambda;
        bool first{true};
    };
    
    float classify(const RMatrixXf& trueLabelMatrix, const RMatrixXf& learnedLabelMatrix){
        float percentCorrect{0.0f};
        
        int numClasses{static_cast<int>(trueLabelMatrix.cols())};
        int numSamples{static_cast<int>(trueLabelMatrix.rows())};
        assert(numSamples == static_cast<int>(learnedLabelMatrix.rows()));
        
        RVectorXf::Index maxLocTrue;
        RVectorXf::Index maxLocLearned;
        RVectorXf dummyVector(numClasses);
        for (int i : boost::irange(0, numSamples)){
            dummyVector = trueLabelMatrix.row(i);
            dummyVector.maxCoeff(&maxLocTrue);
            dummyVector = learnedLabelMatrix.row(i);
            dummyVector.maxCoeff(&maxLocLearned);
            
            if (maxLocTrue == maxLocLearned){
                ++percentCorrect;
            }
        }
        
        return percentCorrect/static_cast<float>(numSamples);
    };
    
    float confusionMatrix(const RMatrixXf& trueLabelMatrix, const RMatrixXf& learnedLabelMatrix, RMatrixXf& cm){
        float percentCorrect{0.0f};
        
        int numClasses{static_cast<int>(trueLabelMatrix.cols())};
        int numSamples{static_cast<int>(trueLabelMatrix.rows())};
        assert(numSamples == static_cast<int>(learnedLabelMatrix.rows()));
        
        cm.resize(numClasses, numClasses);
        cm.fill(0);
        
        RVectorXf::Index maxLocTrue;
        RVectorXf::Index maxLocLearned;
        RVectorXf dummyVector(numClasses);
        for (int i : boost::irange(0, numSamples)){
            dummyVector = trueLabelMatrix.row(i);
            dummyVector.maxCoeff(&maxLocTrue);
            dummyVector = learnedLabelMatrix.row(i);
            dummyVector.maxCoeff(&maxLocLearned);
            
            cm(maxLocTrue, maxLocLearned) = cm(maxLocTrue, maxLocLearned) + 1;
            
            if (maxLocTrue == maxLocLearned){
                ++percentCorrect;
            }
        }
        
        return percentCorrect/static_cast<float>(numSamples);        
    };
    
    class CreateLabelMatrix{
    public:
        CreateLabelMatrix(RMatrixXi& numInClass){
            int rows{numInClass.sum()};
            classMatrix.resize(rows, numInClass.cols());
            
            classMatrix.fill(0.0f);
            int cnt{0};
            for (int i : boost::irange(0, static_cast<int>(numInClass.cols()))){
                for (int j : boost::irange(0, numInClass(0,i))){
                    classMatrix(cnt, i) = 1.0f;
                    ++cnt;
                }
            }
            
        };
        ~CreateLabelMatrix(){};
        RMatrixXf& getClassMatrix(){return classMatrix;};
        
    private:
        RMatrixXf classMatrix;
    };
    
    class CreateKnownLabelMatrix{
    public:
        CreateKnownLabelMatrix(RMatrixXi& numInClass, RMatrixXi& numKnown){

        }
    };
    
    class Create2DTestDataNN{
    public:
        Create2DTestDataNN(int numSamples)
        : output(2*numSamples, 2){
            boost::random::uniform_01<float> rv;
            boost::random::lagged_fibonacci607 gen;
            
            for (int n : boost::irange(0, numSamples)){
                output(n,0) = 0.25 + 0.3*rv(gen);
                output(n,1) = 0.25 + 0.3*rv(gen);
                output(numSamples + n,0) = 0.5 + 0.3*rv(gen);
                output(numSamples + n,1) = 0.5 + 0.3*rv(gen);
            }
        };
        ~Create2DTestDataNN(){
            
        };
        RMatrixXf& getOutput(){return output;};
        
    private:
        RMatrixXf output;
    };
    
    class CreateTestDictNN{
    public:
        CreateTestDictNN(const int numSamples, const int dataDim, const int numTopics)
        : data(numSamples, dataDim),
        basis(numTopics, dataDim),
        weights(numSamples, numTopics),
        corrClss(numSamples){
            assert(dataDim > numTopics);
            // Create nearly orthogonal vectors.
            float epsilon = 0;
            basis.fill(epsilon);
            data.fill(0.0f);
            weights.fill(0.0f);
            
            // Create vectors
            for (int k : boost::irange(0, numTopics)){
                basis(k,k) = 1.0f;
            }
            
            // Normalize vectors.
            for (int k : boost::irange(0, numTopics)){
                basis.row(k) = basis.row(k)/basis.row(k).sum();
            }
            
            // Create the samples.
            boost::random::normal_distribution<float> nrv(1, .5);
            boost::random::lagged_fibonacci607 gen;
            RandomSimulation::Multinomial mult(numTopics);
            Eigen::VectorXf pi(numTopics);
            pi.fill(1.0f/static_cast<float>(numTopics));
            Eigen::VectorXf mean =Eigen::VectorXf::Zero(numTopics);
            for (int n : boost::irange(0, numSamples)){
                int loc = mult.simulate(pi);
                corrClss[n] = loc;
                weights.row(n)[loc] = std::abs(nrv(gen));
                for (int k : boost::irange(0, numTopics)){
                    data.row(n) += weights(n,k)*basis.row(k);
                }
            }
        };
        ~CreateTestDictNN(){};
        RMatrixXf& getData(){return data;};
        RMatrixXf& getWeights(){return weights;};
        RMatrixXf& getBasis(){return basis;};
        RVectorXf& getCorrClss(){return corrClss;};
        
    protected:
        RMatrixXf data;
        RMatrixXf weights;
        RMatrixXf basis;
        RVectorXf corrClss;
    };

    
}

#endif /* NNFactorization_h */
