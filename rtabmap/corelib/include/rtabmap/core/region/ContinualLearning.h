#pragma once

#ifndef CONTINUAL_LEARNING_H
#define CONTINUAL_LEARNING_H

#include "rtabmap/core/rtabmap_core_export.h" // DLL export/import defines
#include "rtabmap/core/DBDriver.h"

#include "rtabmap/core/Parameters.h"
#include "rtabmap/core/region/models/Model.h"
#include "rtabmap/core/region/utils.h"
#include "rtabmap/core/region/storage_policy/ClassBalancedBuffer.h"
#include "rtabmap/core/region/losses/CustomLoss.h"
#include "rtabmap/core/region/TrainThread.h"

#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>

namespace rtabmap
{

    class DBDriver;
    class TrainThread;

    class ContinualLearning
    {
    public:
        ContinualLearning(DBDriver *dbDriver,
        int regionCounter,
                    const ParametersMap &parameters);

        void train(const std::unordered_map<int, std::pair<int, int>> &signatures_moved, bool new_thread = true) const;

        inline int experienceSize() const { return this->_experienceSize; }
        inline const std::unordered_map<int, std::pair<cv::Mat, int>> &currentExperience() const { return this->_currentExperience; }
        inline size_t currentExperienceSize() const { return this->_currentExperience.size(); }
        void addInExperience(int id, const cv::Mat &image, int regionId);
        void addInExperience(int id, int regionId);
        void updateInExperience(int id, int regionId);
        
        inline void clearCurrentExperience() { this->_currentExperience.clear(); }
        inline int roiX() const { return this->_roiX; }
        inline int roiY() const { return this->_roiY; }
        inline int roiWidth() const { return this->_roiWidth; }
        inline int roiHeight() const { return this->_roiHeight; }
        void setCurrentImage(Signature *s);
        inline const cv::Mat &currentImage() const { return this->_currentImage; }
        void predict();
        void checkModelUpdate();
        bool isTraining() const;

        inline const std::set<int> &topKRegions() const { return this->_topKRegions; } 

       
    private:
        void parseParameters(const ParametersMap &parameters);

        DBDriver *_dbDriver;

        // continual
        int _experienceSize;

        int _roiX;
        int _roiY;
        int _roiWidth;
        int _roiHeight;
        int _targetWidth;
        int _targetHeight;
        cv::Mat _currentImage; // current image for experience
        std::unordered_map<int, std::pair<cv::Mat, int>> _currentExperience;

        std::string _modelPath;
        std::string _checkpointPath;

        int _deviceType;
        torch::DeviceType _device;

        // for training
        std::unique_ptr<TrainThread> _trainThread;
        Model _model;

        // for inference
        int _topK;
        std::set<int> _topKRegions;
        at::Tensor _regionProbabilities;
        float _alpha;
    };

}

#endif