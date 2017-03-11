

#ifndef MachineLearning_Setup_h
#define MachineLearning_Setup_h

#include <boost/range/irange.hpp>
#include <vector>
#include "Dense"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatrixXf;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatrixXcf;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatrixXi;
typedef Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RArrayXf;
typedef Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RArrayXi;
typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> RVectorXf;
typedef Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor> RVectorXd;
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor> RVectorXcf;
typedef Eigen::Matrix<int, 1, Eigen::Dynamic, Eigen::RowMajor> RVectorXi;
typedef Eigen::SelfAdjointEigenSolver<RMatrixXf> EigSolver;
typedef std::vector<RMatrixXf> Tensor;

std::string imagePath = "/Users/flennifer/Data/Images/";
std::string inputPath = "/Users/flennifer/Projects/Audio/BirdDataSpectrogram/";
std::string outputPath = "/Users/flennifer/Projects/Audio/BirdDataSpectrogram/Results/Run2_";
std::string machineLearningPath = "/Users/flennifer/MachineLearning/audio1T_BigRankRun1_";

const int numProcessors{28};

template<class T>
class Range;

template<class T>
class RangeIter{
public:
    RangeIter (Range<T>* inputRange, int inputLoc)
    : loc(inputLoc),
      range(inputRange){};
    ~RangeIter (){};
    bool operator != (const RangeIter& other) const{
        return loc != other.loc;
    };
    T& operator* ();
    
    const RangeIter& operator++ (){
        ++loc;
        return *this;
    };
    
private:
    int loc;
    Range<T>* range;
};

template<class T>
class Range{
public:
    Range(T begin, T end, T stepSize)
    : data(NULL){
        assert(stepSize > 0);
        assert(end > begin);
        
        N = static_cast<int>(floor((end - begin)/stepSize));
        data = new T[N];
        
        int cnt{0};
        for (T a = begin; a < end; a += stepSize, ++cnt) data[cnt] = a;
    };
    ~Range(){
        if (data != NULL){
            
        }
    };
    T& get (int loc){
        return data[loc];
    };
    RangeIter<T> begin (){
        return RangeIter<T>(this, 0);
    }
    RangeIter<T> end (){
        return RangeIter<T>(this, N);
    };
    void set (int loc, T val){
        data[loc] = val;
    };
    
private:
    int N;
    T* data;
};

template<class T>
T& RangeIter<T>::operator* () {
    return range->get(loc);
}

#endif
