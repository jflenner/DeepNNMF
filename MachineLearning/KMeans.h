

#ifndef MachineLearning_KMeans_h
#define MachineLearning_KMeans_h

#include <cfloat>

namespace TopicModels {
    
    
    /*****************************************  Ryan K-Means ********************************************************/
    const int KMEANS_INIT_RAND  = 0;
    
    void kmeans(RMatrixXf &data, int k, RVectorXi &clusterLabels, int maxIterations, int initFlag);
    void initializeClusterCentroids(RMatrixXf& clusterCentroids, const RMatrixXf& data, int initFlag);
    int closestCentroid(RMatrixXf::RowXpr dataRow, RMatrixXf& clusterCentroids);
    RMatrixXf euclideanDistance(RMatrixXf::RowXpr X, RMatrixXf &Y);
    void updateClusterCentroids(RMatrixXf& clusterCentroids, RVectorXi& clusterLabels, const RMatrixXf& data);
    
    void kmeans(RMatrixXf &data, int k, RVectorXi &clusterLabels, int maxIterations, int initFlag){
        std::cout << "RYAN KMEANS...";
        
        RMatrixXf   clusterCentroids(k, data.cols());
        initializeClusterCentroids(clusterCentroids, data, initFlag);
        
        //        std::cout << clusterCentroids << std::endl;
        
        for(int r = 0; r < data.rows(); r++)
        {
            clusterLabels(0,r)    = closestCentroid(data.row(r), clusterCentroids);
        }
        
        bool    changed;
        int     iterations  = 0;
        do {
            updateClusterCentroids(clusterCentroids, clusterLabels, data);
            
            changed = false;
            std::cout << "km iter = " << iterations << std::endl;            
            for(int r = 0; r < data.rows(); ++r)
            {
                int updatedClusterLabel = closestCentroid(data.row(r), clusterCentroids);
                if(updatedClusterLabel != clusterLabels(0, r))
                {
                    clusterLabels(0, r)    = updatedClusterLabel;
                    changed                = true;
                }
            }
            iterations++;
        } while(changed && iterations < maxIterations);
        
        std::cout << " rkm complete." << std::endl;
    }
    
    void initializeClusterCentroids(RMatrixXf& clusterCentroids, const RMatrixXf& data, int initFlag){
        switch(initFlag)
        {
            case KMEANS_INIT_RAND:
            default:
                for(int c = 0; c < clusterCentroids.rows(); c++)
                {
                    int chosenRow = rand()%data.rows();
                    clusterCentroids.row(c) = data.row(chosenRow);
                }
                break;
        }
    }
    
    int closestCentroid(RMatrixXf::RowXpr dataRow, RMatrixXf& clusterCentroids){
        float   minDist         = FLT_MAX;
        int     closestCentroid = -1;
        
        RMatrixXf   currentCentroid;
        for(int c = 0; c < clusterCentroids.rows(); c++)
        {
            currentCentroid = clusterCentroids.row(c);
            float   dist    = euclideanDistance(dataRow, currentCentroid).sum();
            //            std::cout << dist << std::endl;
            if(dist < minDist)
            {
                closestCentroid = c;
                minDist = dist;
            }
        }
        
        //        std::cout << minDist << std::endl;
        //        std::cout << closestCentroid << std::endl;
        return closestCentroid;
    }
    
    RMatrixXf euclideanDistance(RMatrixXf::RowXpr X, RMatrixXf &Y){   //Calculate distance using L2 Norm
        // ||X - Y||2 = X*X + Y*Y - 2X*Y
        
        const int N = static_cast<int>(X.rows());
        const int M = static_cast<int>(Y.rows());
        
        // Allocate parts of the expression
        RMatrixXf XX, YY, XY, D;
        XX.resize(N,1);
        YY.resize(1,M);
        XY.resize(N,M);
        D.resize(N,M);
        
        // Compute norms
        XX = X.array().square().rowwise().sum();
        YY = Y.array().square().rowwise().sum().transpose();
        XY = (2*X)*Y.transpose();
        
        // Compute final expression
        D = XX + YY - XY;
        
        return D;
    }
    
    void updateClusterCentroids(RMatrixXf& clusterCentroids, RVectorXi& clusterLabels, const RMatrixXf& data){
        for(int c = 0; c < clusterCentroids.rows(); c++)
        {
            RMatrixXf   newCentroid(1, clusterCentroids.cols());
            int         numElementsInCluster    = 0;
            for(int l = 0; l < clusterLabels.rows(); l++)
            {
                if(c == clusterLabels(l))
                {
                    newCentroid += data.row(l);
                    ++numElementsInCluster;
                }
            }
            if (numElementsInCluster > 0){
                clusterCentroids.row(c) = newCentroid / static_cast<float>(numElementsInCluster);
            }
        }
    }

}

#endif
