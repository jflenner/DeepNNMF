

#ifndef VideoTopicModels_RandomSimulation_h
#define VideoTopicModels_RandomSimulation_h

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/taus88.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/mersenne_twister.hpp> // Use mt11213b for more memory but longer cycle, while rand48 has much shorter cycle and less memory.
#include <boost/random/lagged_fibonacci.hpp> // The lagged fibonacci has the most memory requirements, but fastest speed and longest cycle.
#include <boost/range/irange.hpp>
#include <time.h>
#include "Dense"
#include "Cholesky"
#include "Setup.h"

// Always use references for template arguments.
namespace RandomSimulation {
    
    //
    
    class BoxMueller{
    public:
        BoxMueller(int inputDim, int seed = static_cast<int>(std::time(0))) :
        dim(static_cast<int>(floor(static_cast<float>(inputDim)/2.0))),
        odd(false){
            gen.seed(seed);
            if (static_cast<float>(dim) - ceil(static_cast<float>(inputDim)/2.0) < 0){
                odd = true;
            }
        };
        template<class T>        
        void simulate(T input){
            float var1(0.0);
            float var2(0.0);
            float tmp(0.0);
            float s(0.0);
            
            for (int i = 0; i < dim; ++i) {
                
                s = 2;
                while (s > 1) {
                    var1 = 2.0*(dist(gen) - .5);
                    var2 = 2.0*(dist(gen) - .5);
                    s = var1*var1 + var2*var2;
                }
                
                tmp = sqrtf(-2*logf(s)/s);
                input(2*i) = var1*tmp;
                input(2*i + 1) = var2*tmp;
            }
            
            if (odd){
                s = 2;
                while (s > 1) {
                    var1 = 2.0*(dist(gen) - .5);
                    var2 = 2.0*(dist(gen) - .5);
                    s = var1*var1 + var2*var2;
                }
                
                tmp = sqrtf(-2*logf(s)/s);
                input(2*dim) = var1*tmp;
            }
        };
        
    private:
        boost::random::lagged_fibonacci607 gen;
        boost::random::uniform_01<float> dist;
        int dim;
        bool odd;
    };
    
    class InverseGaussian{
    public:
        InverseGaussian()
        : nd(0.0, 1.0f){
            gen.seed(static_cast<int>(std::time(0)));
        };
        void simulate(const float inputMean, const float inputPre, float& output){
            output = nd(gen);
            output = output*output;
            
            output = inputMean + (inputMean*inputMean*output)/(2*inputPre) - inputMean/(2*inputPre)*std::sqrt(4*inputMean*inputPre*output + inputMean*inputMean*output*output);
            float test = dist(gen);
            if (test > inputMean/(inputMean + output)){
                output = inputMean*inputMean/output;
            }
        };
        
    private:
        boost::random::lagged_fibonacci607 gen;
        boost::random::uniform_01<float> dist;
        boost::random::normal_distribution<float> nd;
    };
    
    class MultivarGaussian {
    public:
        MultivarGaussian(int inputK, int seed = static_cast<int>(time(0)))
        : bm(inputK, seed),
          L(inputK, inputK){
            K = inputK;
            info = 0;
            eig.resize(K,1);
            // Needed for quick vector operations.
            ones.resize(K,1);
            ones.fill(1.0);
            gaussVar.resize(K,1);
        }
        ~MultivarGaussian(){};
        template<class T, class C>
        void simulatePer(RMatrixXf& inputPer, T mean, C output){
            
            L = inputPer.llt().matrixL();
            
            bm.simulate<RMatrixXf&>(gaussVar);
            
            output = (L.inverse()*gaussVar).transpose() + mean;
        };  // The mean and output vectors are row vectors.
        template<class T, class C>
        void simulatePer(Eigen::Block<RMatrixXf> inputPer, T mean, C output){
            
            L = inputPer.llt().matrixL();
            
            bm.simulate<RMatrixXf&>(gaussVar);
            
            output = (L.inverse()*gaussVar).transpose() + mean;
        };  // The mean and output vectors are row vectors.
        template<class T, class C>
        void simulateCov(RMatrixXf& inputCov, T mean, C output){
            bm.simulate<RMatrixXf&>(gaussVar);
            
            L = inputCov.llt().matrixL();
            
            output = (L*gaussVar).transpose() + mean;
        }; // The mean and output vectors are row vectors.
        template<class T, class C>
        void simulateCov(Eigen::Block<RMatrixXf> inputCov, T mean, C output){
            bm.simulate<RMatrixXf&>(gaussVar);
            
            L = inputCov.llt().matrixL();
            
            output = (L*gaussVar).transpose() + mean;
        }; // The mean and output vectors are row vectors.
        template<class T, class C>
        void simulatePer(float inputPer, T mean, C output){
            float invSqrtPer = sqrt(1.0f/inputPer);
            bm.simulate<RMatrixXf&>(gaussVar);

            gaussVar *= invSqrtPer;
            output = gaussVar.transpose() + mean;
        };  // The mean and output are row vectors.
        template<class T, class C>
        void simulateCov(float inputCov, T mean, C output){
            bm.simulate<RMatrixXf&>(gaussVar);
            gaussVar *= sqrt(inputCov);
            output = gaussVar.transpose() + mean;
        };  // The mean and output are row vectors.
        
    private:
        BoxMueller bm;
        RMatrixXf eig;
        RMatrixXf gaussVar;
        RMatrixXf ones;
        RMatrixXf L;
        char vec;
        char UL;
        int info;
        int K;
        int evenK;
    };
    
    class Multinomial {
    public:
        Multinomial(int inputDim, int seed = static_cast<int>(time(0))) : dim(inputDim){gen.seed(seed);};
        ~Multinomial(){}
        template<class T>
        int simulate(T inputProb){
            float cumSum = 0;
            float ranNum = rnd(gen);
            float tmp = 0;
            
            for (int i = 0; i < dim - 1; ++i) {
                cumSum += inputProb[i];
                if ( (ranNum < cumSum) & (ranNum > tmp)) {
                    return i;
                }
            }
            
            return dim - 1;
        };
        
    private:
        boost::random::lagged_fibonacci607 gen;
        boost::random::uniform_01<float> rnd;
        int dim;
    };
    
    class Binomial {
    public:
        Binomial(int seed = static_cast<int>(time(0))){gen.seed(seed);};
        ~Binomial(){}
        int simulate(float input) {
            if (rnd(gen) > input) {
                return 1;
            }
            return 0;
        }
        
    private:
        boost::random::lagged_fibonacci607 gen;
        boost::random::uniform_01<float> rnd;
    };
    
    class Beta {
    public:
        Beta(int seed = static_cast<int>(time(0))){gen.seed(seed);};
        ~Beta(){}
        void simulate(const float a, const float b, float& output){
            float X;
            float Y;
            
            boost::random::gamma_distribution<float>::param_type param1(a,1);
            boost::random::gamma_distribution<float>::param_type param2(b,1);
            gam1.param(param1);
            gam2.param(param2);
            
            X = gam1(gen);
            Y = gam2(gen);
            
            output = (X)/(X + Y);
        };
        
    private:
        boost::random::lagged_fibonacci607 gen;
        boost::random::gamma_distribution<float> gam1;
        boost::random::gamma_distribution<float> gam2;
    };
    
    class Dirichlet {
    public:
        Dirichlet(int inputDim, int seed = static_cast<int>(time(0))) :
        dim(inputDim){gen.seed(seed);};
        ~Dirichlet(){}
        template<class T>
        void simulate(RMatrixXf::RowXpr alpha, T output){
            // Simulates all the gamma random variables
            for (int j = 0; j < dim; ++j) {
                boost::random::gamma_distribution<float>::param_type param1(alpha(j), 1);
                gam.param(param1);
                output(j) = gam(gen);
            }
            
            // Normailze the random variable.
            output.array() /= output.sum();
        };
        template<class T>
        void simulate(const RVectorXf& alpha, T output){
            // Simulates all the gamma random variables
            for (int j = 0; j < dim; ++j) {
                boost::random::gamma_distribution<float>::param_type param1(alpha(j), 1);
                gam.param(param1);
                output(j) = gam(gen);
            }
            
            // Normailze the random variable.
            output.array() /= output.sum();
        };
        
        
    private:
        boost::random::lagged_fibonacci607 gen;
        boost::random::gamma_distribution<float> gam;
        int dim;
    };
    
    class VonMisesFisher3D {
    public:
        VonMisesFisher3D(){}
        ~VonMisesFisher3D(){}
        void simulate(int inputNum, float prec, float* output){
            float w;
            float v1;
            float v2;
            float theta;
            float rndtmp;
            float pi = 3.14159;
            
            float precinv = 1.0/prec;
            float expprec = expf(-1.0*prec);
            float preccnst = (expf(prec) - expprec);
            
            for (int i = 0; i < inputNum; ++i){
                rndtmp = rnd(gen);
                w = precinv*logf(expprec + preccnst*rndtmp);
                theta = 2*pi*rnd(gen);
                v1 = cosf(theta);
                v2 = sinf(theta);
                rndtmp = sqrtf(1.0 - w*w);
                
                output[3*i] = rndtmp*v1;
                output[3*i + 1] = rndtmp*v2;
                output[3*i + 2] = w;
            }
        };
        
    private:
        boost::random::lagged_fibonacci607 gen;
        boost::random::uniform_01<float> rnd;
    };
    
    // This class was written based on Hoff at University of Washington.
    class Bingham3D{
    public:
        Bingham3D() :
        tmpY(3),
        N(100),
        numIter(10){
            gen.seed(static_cast<int>(time(0)));
            h = 1.0/static_cast<float>(N);
            grid.resize(N);
            sqrtTheta.resize(N);
            expTheta.resize(N);
            q.resize(3);
            for (int i = 0; i < N; ++i) {
                grid[i] = static_cast<float>(i + 1)*h;
            }
            
            sqrtTheta = grid.array().sqrt();
            sqrtTheta = 1.0/sqrtTheta.array(); // Takes inverse square root of the vector.
        };
        ~Bingham3D();
        void simulate(float* initial, float* Matrix, float* output);
        
    private:
        boost::random::lagged_fibonacci607 gen;
        boost::random::uniform_01<> rnd;
        Eigen::VectorXf tmpY;
        Eigen::VectorXf grid;
        Eigen::VectorXf sqrtTheta;
        Eigen::VectorXf q;
        Eigen::VectorXf expTheta;
        float h;
        int N;
        int numIter;
    };
    
    class NormalWishart {
    public:
        NormalWishart(int inputDim)
        : dim(inputDim),
          bmb(static_cast<int>(ceil(static_cast<float>(inputDim*inputDim)))),
          bmg(static_cast<int>(ceil(static_cast<float>(inputDim)))),
          L(inputDim, inputDim),
          T(inputDim, inputDim),
          b(inputDim, inputDim),
          LT(inputDim, inputDim),
          gaussRnd(inputDim){};
        ~NormalWishart(){};
        void simulateInvScale(RMatrixXf::RowXpr inputMean,
                              RMatrixXf& invScale,
                              float beta,
                              int degOfFreedom,
                              RMatrixXf::RowXpr outputMean,
                              Eigen::Block<RMatrixXf> outputPer,
                              float& logPerDet){
            assert(degOfFreedom > dim - 1);            
            L = invScale.llt().matrixL();
            L = L.inverse();
            calculate(inputMean, beta, degOfFreedom, outputMean, outputPer, logPerDet);
        }
        void simulate(RMatrixXf::RowXpr inputMean,
                      RMatrixXf& scale,
                      float beta,
                      int degOfFreedom,
                      RMatrixXf::RowXpr outputMean,
                      Eigen::Block<RMatrixXf> outputPer,
                      float& logPerDet){
            assert(degOfFreedom > dim - 1);
            
            L = scale.llt().matrixL();
            
            calculate(inputMean, beta, degOfFreedom, outputMean, outputPer, logPerDet);
        };
        int getDim(){return dim;};
        
    private:
        void calculate(RMatrixXf::RowXpr inputMean,
                       float beta,
                       int degOfFreedom,
                       RMatrixXf::RowXpr outputMean,
                       Eigen::Block<RMatrixXf> outputPer,
                       float& logPerDet){
            bmb.simulate<RMatrixXf&>(b);
            T.fill(0);
            
            logPerDet = log((L.diagonal().array()*L.diagonal().array()).prod());
            float tmpChi;
            int cnt = 0;
            for (int i = 0; i < dim; ++i) {
                boost::random::chi_squared_distribution<float>::param_type chiParam(static_cast<float>(degOfFreedom - i));
                chiSq.param(chiParam);
                for (int j = 0; j < i; ++j){
                    T(i, j) = b(cnt);
                    ++cnt;
                }
                tmpChi = chiSq(gen);
                logPerDet += log(tmpChi);
                T(i,i) = sqrt(tmpChi);
            }
            
            LT = L*T;
            
            outputPer = LT*LT.transpose();
            
            bmg.simulate<RVectorXf&>(gaussRnd);
            gaussRnd.array() = gaussRnd.array()/sqrt(beta);
            outputMean = (gaussRnd*LT.inverse() + inputMean);
        }
        boost::random::lagged_fibonacci607 gen;
        boost::random::chi_squared_distribution<float> chiSq;
        BoxMueller bmb;
        BoxMueller bmg;
        RMatrixXf T;
        RMatrixXf b;
        RVectorXf gaussRnd;
        RMatrixXf L;
        RMatrixXf LT;
        int dim;
    };
    
    class CRT{
    public:
        CRT(int seed = static_cast<int>(std::time(0))){
            gen.seed(seed);
        };
        void simulate(const int m, const float gamma, int& output){
            output = 0;
            for (int n : boost::irange(0, m)){
                float p = dist(gen);
                float q = gamma/(static_cast<float>(n) + gamma);
                if (p < q){
                    ++output;
                }
            }
        };
        
    private:
        boost::random::lagged_fibonacci607 gen;
        boost::random::uniform_01<float> dist;
    };
    
}

#endif
