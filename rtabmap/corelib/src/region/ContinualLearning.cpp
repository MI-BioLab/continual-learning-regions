#include "rtabmap/core/region/ContinualLearning.h"
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UTimer.h>

namespace rtabmap
{

	ContinualLearning::ContinualLearning(DBDriver *dbDriver,
										 int regionCounter,
										 const ParametersMap &parameters) : _dbDriver(dbDriver),
																			_topK(Parameters::defaultContinualTopK()),
																			_modelPath(Parameters::defaultContinualModelPath()),
																			_checkpointPath(Parameters::defaultContinualCheckpointPath()),
																			_deviceType(Parameters::defaultContinualDevice()),
																			_experienceSize(Parameters::defaultContinualExperienceSize()),
																			_alpha(Parameters::defaultContinualAlpha()),
																			_roiX(Parameters::defaultContinualRoiX()),
																			_roiY(Parameters::defaultContinualRoiY()),
																			_roiWidth(Parameters::defaultContinualRoiWidth()),
																			_roiHeight(Parameters::defaultContinualRoiHeight()),
																			_targetWidth(Parameters::defaultContinualTargetWidth()),
																			_targetHeight(Parameters::defaultContinualTargetHeight())
	{
		this->parseParameters(parameters);
		this->_device = this->_deviceType && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
		FeatureExtractor featureExtractor;
		IncrementalLinear classifier(512, regionCounter);
		//_regionCounter corresponds to the number of classes
		if (regionCounter == 0) // no checkpoint yet, load initial model
		{
			this->_model = Model(featureExtractor, classifier, _modelPath);
		}
		else if (regionCounter >= 1)
		{
			this->_model = Model(featureExtractor, classifier, _checkpointPath);
		}
		else
		{
			ULOGGER_ERROR("region counter < 0 should never happen!");
		}

		this->_model->eval();
		this->_model->to(this->_device);
		ULOGGER_DEBUG("Model creation. Training is %s", this->_model->is_training() ? "enabled" : "disabled");
		Model new_model = this->_model->clone();
		ULOGGER_DEBUG("Model clone creation. Training is %s", this->_model->is_training() ? "enabled" : "disabled");
		this->_trainThread = std::make_unique<TrainThread>(_dbDriver,
														   new_model,
														   parameters,
														   static_cast<int64_t>(_targetWidth),
														   static_cast<int64_t>(_targetHeight),
														   _checkpointPath,
														   _device);
	}

	void ContinualLearning::parseParameters(const ParametersMap &parameters)
	{
		Parameters::parse(parameters, Parameters::kContinualTopK(), _topK);
		Parameters::parse(parameters, Parameters::kContinualModelPath(), _modelPath);
		Parameters::parse(parameters, Parameters::kContinualCheckpointPath(), _checkpointPath);
		Parameters::parse(parameters, Parameters::kContinualDevice(), _deviceType);
		Parameters::parse(parameters, Parameters::kContinualExperienceSize(), _experienceSize);
		Parameters::parse(parameters, Parameters::kContinualAlpha(), _alpha);
		Parameters::parse(parameters, Parameters::kContinualRoiX(), _roiX);
		Parameters::parse(parameters, Parameters::kContinualRoiY(), _roiY);
		Parameters::parse(parameters, Parameters::kContinualRoiWidth(), _roiWidth);
		Parameters::parse(parameters, Parameters::kContinualRoiHeight(), _roiHeight);
		Parameters::parse(parameters, Parameters::kContinualTargetWidth(), _targetWidth);
		Parameters::parse(parameters, Parameters::kContinualTargetHeight(), _targetHeight);
	}

	void ContinualLearning::addInExperience(int id, const cv::Mat &image, int regionId)
	{
		if (this->_currentExperience.count(id))
		{
			ULOGGER_DEBUG("Trying to add in experience id %d already present. Use updateInExperience instead.", id);
		}
		if (image.empty())
		{
			ULOGGER_DEBUG("Trying to add in experience an empty image.");
		}
		else
		{
			this->_currentExperience.insert({id, {image, regionId}});
		}
	}

	void ContinualLearning::addInExperience(int id, int regionId)
	{
		if (!this->_currentImage.empty())
		{
			this->addInExperience(id, this->_currentImage, regionId);
		}
	}

	void ContinualLearning::updateInExperience(int id, int regionId)
	{
		if (this->_currentExperience.count(id))
		{
			this->_currentExperience[id].second = regionId;
		}
		else
		{
			ULOGGER_DEBUG("Trying to update in experience id %d not present. Use addIdInExperience instead.", id);
		}
	}

	void ContinualLearning::train(const std::unordered_map<int, std::pair<int, int>> &signatures_moved, bool new_thread) const
	{
		if (this->_currentExperience.size() > 0)
		{
			this->_trainThread->train(this->_currentExperience, signatures_moved, new_thread);
		}
	}

	void ContinualLearning::checkModelUpdate()
	{
		if (this->_trainThread.get() != 0)
		{
			if (!this->_trainThread->is_training() && this->_trainThread->last_training_end())
			{
				this->_model = this->_trainThread->model();
				this->_model->eval();
				this->_model->to(this->_device);
				ULOGGER_DEBUG("Updating model in inference. Training is %s", this->_model->is_training() ? "enabled" : "disabled");
			}
		}
	}

	bool ContinualLearning::isTraining() const { return this->_trainThread->is_training(); }

	void ContinualLearning::setCurrentImage(Signature *s)
	{
		// cv::Mat image = s->sensorData().imageRaw();
		// ULOGGER_DEBUG("Current image is %s empty", image.empty() ? "" : "not");
		// if (image.empty())
		// {
		// 	return;
		// }
		// this->_currentImage = image.clone();
		this->_currentImage = s->sensorData().imageRaw();
		if (this->_currentImage.empty())
		{
			return;
		}
		if (this->_roiWidth != 0 && this->_roiHeight != 0)
		{
			cv::Rect roi(this->_roiX, this->_roiY, this->_roiWidth, this->_roiHeight);
			this->_currentImage = this->_currentImage(roi);
		}
		ULOGGER_DEBUG("Cropped image size: %d %d", this->_currentImage.cols, this->_currentImage.rows);
		cv::resize(this->_currentImage, this->_currentImage, cv::Size(this->_targetWidth, this->_targetHeight));
		ULOGGER_DEBUG("Resized image size: %d %d", this->_currentImage.cols, this->_currentImage.rows);
	}

	void ContinualLearning::predict()
	{
		ULOGGER_DEBUG("Model trained: %s", this->_model->is_trained() ? "true" : "false");
		if (this->_model->is_trained()) // always called when output size is at least 1
		{
			torch::NoGradGuard no_grad;
			at::Tensor input = image_to_tensor(this->_currentImage, this->_targetWidth, this->_targetHeight);
			input = input.to(this->_device);
			at::Tensor output = this->_model->forward(input);
			output = torch::squeeze(output, 0).detach().cpu();
			for (size_t i = 0; i < output.size(0); i++)
			{
				ULOGGER_DEBUG("Probability for region %d before EMA=%f", (int)i, output[i].item<double>());
			}
			if (this->_regionProbabilities.size(0) == 0)
			{
			}
			else if (output.size(0) == this->_regionProbabilities.size(0))
			{
				output = this->_alpha * output + (1.0 - this->_alpha) * this->_regionProbabilities;
			}
			else // new neurons, output.size(0) > _regionProbabilities.size(0)
			{
				output.slice(0, 0, this->_regionProbabilities.size(0)) = this->_alpha * output.slice(0, 0, this->_regionProbabilities.size(0)) + (1.0 - this->_alpha) * this->_regionProbabilities;
			}

			this->_regionProbabilities = output;
			for (size_t i = 0; i < this->_regionProbabilities.size(0); i++)
			{
				ULOGGER_DEBUG("Probability for region %d after EMA=%f", (int)i, this->_regionProbabilities[i].item<double>());
			}

			std::tuple<at::Tensor, at::Tensor> sortedProbabilites = at::sort(this->_regionProbabilities, c10::optional<bool>(false), -1, true);
			at::Tensor sortedRegionsIndices = std::get<1>(sortedProbabilites).slice(0, 0, this->_topK);
			std::vector<int> sortedRegions(sortedRegionsIndices.data_ptr<int64_t>(), sortedRegionsIndices.data_ptr<int64_t>() + sortedRegionsIndices.numel());
			for (size_t i = 0; i < sortedRegions.size(); i++)
			{
				ULOGGER_DEBUG("Top-%d prediction=%d", i + 1, sortedRegions[i]);
			}
			this->_topKRegions = std::set<int>(sortedRegions.begin(), sortedRegions.end());
		}
	}

}