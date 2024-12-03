#include "rtabmap/core/region/TrainThread.h"
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UTimer.h>
#include "rtabmap/core/region/losses/FocalLoss.h"
#include "rtabmap/core/region/losses/CrossEntropyLoss.h"
#include "rtabmap/core/region/datasets/LatentDataset.h"
#include "rtabmap/core/region/datasets/ExperienceDataset.h"
#include "rtabmap/core/region/storage_policy/ClassBalancedBuffer.h"
#include "rtabmap/core/region/storage_policy/ReservoirSamplingBuffer.h"
#include "rtabmap/core/region/samplers/ReplaySampler.h"
#include "rtabmap/utilite/UProcessInfo.h"

namespace rtabmap
{

    TrainThread::TrainThread(DBDriver *dbDriver,
                             const Model &model,
                             const ParametersMap &parameters,
                             int64_t target_width,
                             int64_t target_height,
                             const std::string &checkpointPath,
                             torch::DeviceType device) : _dbDriver(dbDriver),
                                                         _model(model),
                                                         _training_end(false),
                                                         _is_training(false),
                                                         _target_width(target_width),
                                                         _target_height(target_height),
                                                         _checkpointPath(checkpointPath),
                                                         _device(device),
                                                         _learning_rate(Parameters::defaultContinualLearningRate()),
                                                         _epochs(Parameters::defaultContinualEpochs()),
                                                         _optimizer_type(Parameters::defaultContinualOptimizer()),
                                                         _replay_memory_type(Parameters::defaultContinualReplayMemory()),
                                                         _replay_memory_size(Parameters::defaultContinualReplayMemorySize()),
                                                         _weighting_method(Parameters::defaultContinualWeightingMethod()),
                                                         _beta(Parameters::defaultContinualBeta()),
                                                         _loss_type(Parameters::defaultContinualLossFunction()),
                                                         _gamma(Parameters::defaultContinualGamma()),
                                                         _feature_batch_size(Parameters::defaultContinualFeatureBatchSize()),
                                                         _batch_size(Parameters::defaultContinualBatchSize()),
                                                         _replay_memory_batch_size(Parameters::defaultContinualReplayMemoryBatchSize())
    {
        this->parseParameters(parameters);
        this->make_loss_fn();
        this->make_replay_memory();
        ULOGGER_DEBUG("Model in training thread. Training is %s", this->_model->is_training() ? "enabled" : "disabled");
        // this->_replay_memory = std::make_shared<ClassBalancedBuffer>(this->_replay_memory_size);
    }

    Model TrainThread::model()
    {
        this->_mutex.lock();
        Model model = this->_model->clone();
        ULOGGER_DEBUG("Model cloning in training thread. Training is %s", this->_model->is_training() ? "enabled" : "disabled");
        this->_training_end = false;
        this->_mutex.unlock();
        return model;
    }

    bool TrainThread::is_training()
    {
        this->_mutex.lock();
        bool is_training = this->_is_training;
        this->_mutex.unlock();
        return is_training;
    }

    bool TrainThread::last_training_end()
    {
        this->_mutex.lock();
        bool training_end = this->_training_end;
        this->_mutex.unlock();
        return training_end;
    }

    void TrainThread::train(std::unordered_map<int, std::pair<cv::Mat, int>> experience, std::unordered_map<int, std::pair<int, int>> signatures_moved, bool new_thread)
    {
        ULOGGER_DEBUG("Training on experience with size=%d", experience.size());
        this->_mutex.lock();
        if (this->_is_training)
        {
            this->_mutex.unlock();
            ULOGGER_FATAL("Training cannot start until last is finished!");
            return;
        }
        else
        {
            this->_is_training = true;
            this->_mutex.unlock();
        }
        if(new_thread)
        {
            boost::thread t(boost::bind(&TrainThread::run, this, experience, signatures_moved));
            t.detach();
        }
        else {
            this->run(experience, signatures_moved);
        }
        
    }

    void TrainThread::run(const std::unordered_map<int, std::pair<cv::Mat, int>> &experience, const std::unordered_map<int, std::pair<int, int>> &signatures_moved)
    {
        UTimer timer;
        timer.start();

        // STEP 1: UPDATE REPLAY MEMORY IF THERE IS
        if (this->_replay_memory->buffer()->size() > 0 && !signatures_moved.empty())
        {
            //TODO
        }

        // STEP 1: EXPERIENCE TO VECTORS OF IDS, IMAGES AND LABELS
        std::vector<size_t> ids; // TODO also ids as tensors?
        std::vector<cv::Mat> images;
        std::vector<size_t> labels;

        for (const auto &e : experience)
        {
            ids.emplace_back(static_cast<size_t>(e.first));
            images.emplace_back(e.second.first);
            labels.emplace_back(e.second.second);
        }

        // STEP 2: CONVERT VECTORS TO TENSORS
        ULOGGER_DEBUG("RAM usage before converting to tensor=%ld", UProcessInfo::getMemoryUsage());
        std::vector<torch::Tensor> images_vec(images.size());
        std::vector<torch::Tensor> labels_vec(labels.size());

        for (int i = 0; i < images.size(); i++)
        {
            images_vec[i] = image_to_tensor(images[i], this->_target_width, this->_target_height);
            labels_vec[i] = torch::tensor(static_cast<float>(labels[i]));
        }

        torch::Tensor images_tensor = torch::cat(images_vec);
        torch::Tensor labels_tensor = torch::stack(labels_vec).to(torch::kLong);
        ULOGGER_DEBUG("Images_tensor=%ld", (int64_t)images_tensor.storage().nbytes());
        ULOGGER_DEBUG("Labels_tensor=%ld", (int64_t)labels_tensor.storage().nbytes());
        ULOGGER_DEBUG("RAM usage after converting to tensor=%ld", UProcessInfo::getMemoryUsage());

        // STEP 2: CREATE CURRENT EXPERIENCE DATASET
        // With dataset and dataloader is slower
        // ExperienceDataset experience_dataset(ids, images, labels, this->_target_width, this->_target_height);
        // auto map_dataset = experience_dataset.map(torch::data::transforms::Stack<>());

        // STEP 3: MODEL ADAPTATION ON NEW CLASSES
        ULOGGER_DEBUG("RAM usage before model adaptation=%ld", UProcessInfo::getMemoryUsage());
        this->_model->adapt(labels_tensor);
        ULOGGER_DEBUG("RAM usage after model adaptation=%ld", UProcessInfo::getMemoryUsage());

        // STEP 4: UPDATE OPTIMIZER
        ULOGGER_DEBUG("RAM usage before making optimizer=%ld", UProcessInfo::getMemoryUsage());
        this->make_optimizer();
        ULOGGER_DEBUG("RAM usage after making optimizer=%ld", UProcessInfo::getMemoryUsage());

        size_t num_classes = this->_model->classifier->linear->options.out_features(); // should be equal to values returned by _unique2 on total dataset (current experience + replay)

        // // STEP 5: UPDATE LOSS WEIGHTS
        // if (this->_weighting_method > 0)
        // {
        //     ULOGGER_DEBUG("RAM usage before weighting function=%ld", UProcessInfo::getMemoryUsage());
        //     // compute total labels from current experience and replay memory
        //     // if use dataset, concat datasets and get labels, use tensor concatenation otherwise
        //     torch::Tensor total_labels = torch::cat({labels_tensor, this->_replay_memory->buffer()->labels()});
        //     std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_values = at::_unique2(total_labels, false, false, true);
        //     at::Tensor samples_per_class = std::get<2>(unique_values);
        //     for(size_t i = 0; i < samples_per_class.size(0); i++)
        //     {
        //         ULOGGER_DEBUG("Samples per class=%d", (int)samples_per_class[i].item<int64_t>());
        //     }

        //     torch::Tensor weights = this->compute_weights(samples_per_class);
        //     this->_loss_fn->options.weight(weights);
        //     ULOGGER_DEBUG("RAM usage after weighting function=%ld", UProcessInfo::getMemoryUsage());
        // }

        // STEP 6: EXTRACT FREEZED FEATURES

        std::vector<torch::Tensor> freezed_features_vec; // TODO save in db
        // With dataloader is slower
        // auto feature_dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(map_dataset), torch::data::DataLoaderOptions().batch_size(this->_feature_batch_size).workers(4));
        //                                                                                                                                                                                                                                                                                                                                                             torch::data::DataLoaderOptions(this->_feature_batch_size).enforce_ordering(false).drop_last(false).workers(4));
        // ULOGGER_DEBUG("Extracting freezed features");
        // {
        //     torch::NoGradGuard no_grad;
        //     for (const auto &batch : *feature_dataloader)
        //     {
        //         torch::Tensor x = batch.data.to(this->_device, true);
        //         torch::Tensor freezed_features = this->_model->feature_extractor->extract_freezed_features(x);
        //         freezed_features_vec.emplace_back(freezed_features)
        //     }
        // }
        size_t i = 0;
        size_t j = this->_feature_batch_size;
        ULOGGER_DEBUG("RAM usage before freezed features extraction=%ld", UProcessInfo::getMemoryUsage());
        while (i < labels_tensor.size(0))
        {
            if (i + this->_feature_batch_size >= labels_tensor.size(0))
            {
                j = labels_tensor.size(0) - i;
            }
            torch::Tensor features = this->_model->feature_extractor->extract_freezed_features(images_tensor.slice(0, i, i + j).to(this->_device, true));
            freezed_features_vec.emplace_back(features);
            i += j;
        }
        torch::Tensor freezed_features = torch::cat(freezed_features_vec, 0);
        ULOGGER_DEBUG("RAM usage after freezed features extraction=%ld", UProcessInfo::getMemoryUsage());

        // STEP 7: TRAIN
        ULOGGER_DEBUG("RAM usage before dataset creation=%ld", UProcessInfo::getMemoryUsage());
        std::shared_ptr<LatentDataset> experience_dataset = std::make_shared<LatentDataset>(ids, freezed_features, labels_tensor);

        std::shared_ptr<LatentDataset> total_dataset;
        experience_dataset->concat(this->_replay_memory->buffer(), total_dataset);

        torch::Tensor samples_per_class = torch::zeros(num_classes).to(torch::kLong);
        samples_per_class.index_put_({total_dataset->classes_in_dataset().to(torch::kLong)}, total_dataset->samples_per_class().to(torch::kLong));

        for(size_t i = 0; i < samples_per_class.size(0); i++)
        {
            ULOGGER_DEBUG("Class in total dataset=%d", (int)i);
            ULOGGER_DEBUG("Samples per class=%d", (int)samples_per_class[i].item<int64_t>());
        }

        // STEP 5: UPDATE LOSS WEIGHTS
        if (this->_weighting_method > 0)
        {
            ULOGGER_DEBUG("RAM usage before weighting function=%ld", UProcessInfo::getMemoryUsage());
            // compute total labels from current experience and replay memory
            // if use dataset, concat datasets and get labels, use tensor concatenation otherwise

            torch::Tensor weights = this->compute_weights(samples_per_class);
            for(size_t i = 0; i < weights.size(0); i++)
            {
                ULOGGER_DEBUG("Weights for class %d=%f", (int)i, weights[i].item<double>());
            }
            this->_loss_fn->options.weight(weights);
            ULOGGER_DEBUG("RAM usage after weighting function=%ld", UProcessInfo::getMemoryUsage());
        }

        

        auto map_dataset = (*total_dataset).map(torch::data::transforms::Stack<>());
        ULOGGER_DEBUG("RAM usage after dataset creation=%ld", UProcessInfo::getMemoryUsage());

        ULOGGER_DEBUG("RAM usage before sampler creation=%ld", UProcessInfo::getMemoryUsage());
        ReplaySampler replay_sampler(ids.size(), this->_replay_memory->buffer()->size().value(), this->_batch_size, this->_replay_memory_batch_size);
        ULOGGER_DEBUG("RAM usage after sampler creation=%ld", UProcessInfo::getMemoryUsage());
        ULOGGER_DEBUG("RAM usage before dataloader creation=%ld", UProcessInfo::getMemoryUsage());
        auto experience_dataloader = torch::data::make_data_loader(map_dataset,
                                                                   replay_sampler,
                                                                   torch::data::DataLoaderOptions(this->_batch_size).drop_last(false).workers(2));
        ULOGGER_DEBUG("RAM usage after dataloader creation=%ld", UProcessInfo::getMemoryUsage());

        float experience_loss = 0.0;
        for (int i = 0; i < this->_epochs; i++)
        {
            ULOGGER_DEBUG("Epoch=%d", i + 1);
            ULOGGER_DEBUG("RAM usage before loop=%ld", UProcessInfo::getMemoryUsage());
            float epoch_loss = this->loop(experience_dataloader, num_classes, true);
            ULOGGER_DEBUG("RAM usage after loop=%ld", UProcessInfo::getMemoryUsage());
            ULOGGER_DEBUG("Epoch loss=%f", epoch_loss);
            experience_loss += epoch_loss;
        }
        experience_loss /= this->_epochs;
        ULOGGER_DEBUG("Experience loss=%f", experience_loss);

        // STEP 8: UPDATE REPLAY MEMORY
        ULOGGER_DEBUG("RAM usage before replay memory update=%ld", UProcessInfo::getMemoryUsage());
        this->_replay_memory->update(experience_dataset);
        ULOGGER_DEBUG("RAM usage after replay memory update=%ld", UProcessInfo::getMemoryUsage());
        ULOGGER_DEBUG("Replay memory updated");

        this->_mutex.lock();
        this->_is_training = false;
        this->_training_end = true;
        this->_mutex.unlock();
        ULOGGER_DEBUG("Time for training=%fs", timer.getElapsedTime());

        UTimer dict_timer;
        dict_timer.start();
        this->_model->save_state_dict(this->_checkpointPath);
        ULOGGER_DEBUG("Total Time for saving checkpoint=%fs", dict_timer.ticks());
        std::unordered_set<int> ids_in_memory;
        this->_replay_memory->get_ids_in_memory(ids_in_memory);
        this->saveReplayMemory(ids, freezed_features, ids_in_memory);
        ULOGGER_DEBUG("Total Time for training thread=%fs", timer.ticks());
        std::cout << "\n\nTRAINING END\n\n";
    }

    void TrainThread::parseParameters(const ParametersMap &parameters)
    {
        Parameters::parse(parameters, Parameters::kContinualLearningRate(), _learning_rate);
        Parameters::parse(parameters, Parameters::kContinualEpochs(), _epochs);
        Parameters::parse(parameters, Parameters::kContinualOptimizer(), _optimizer_type);
        Parameters::parse(parameters, Parameters::kContinualReplayMemorySize(), _replay_memory_size);
        Parameters::parse(parameters, Parameters::kContinualReplayMemory(), _replay_memory_type);
        Parameters::parse(parameters, Parameters::kContinualWeightingMethod(), _weighting_method);
        Parameters::parse(parameters, Parameters::kContinualBeta(), _beta);
        Parameters::parse(parameters, Parameters::kContinualLossFunction(), _loss_type);
        Parameters::parse(parameters, Parameters::kContinualGamma(), _gamma);
        Parameters::parse(parameters, Parameters::kContinualFeatureBatchSize(), _feature_batch_size);
        Parameters::parse(parameters, Parameters::kContinualBatchSize(), _batch_size);
        Parameters::parse(parameters, Parameters::kContinualReplayMemoryBatchSize(), _replay_memory_batch_size);
    }

    void TrainThread::make_optimizer()
    {
        if(this->_optimizer.get())
        {
            this->_optimizer->add_parameters(this->_model->parameters());
            return;
        }

        if (this->_optimizer_type == 0)
        {
            this->_optimizer = std::make_shared<torch::optim::Adam>(this->_model->parameters(), torch::optim::AdamOptions(this->_learning_rate));
        }
        else if (this->_optimizer_type == 1)
        {
            this->_optimizer = std::make_shared<torch::optim::SGD>(this->_model->parameters(), torch::optim::SGDOptions(this->_learning_rate));
        }
        else
        {
            ULOGGER_FATAL("Optimizer type error: received %d", this->_optimizer_type);
        }
    }

    void TrainThread::make_loss_fn()
    {
        if (this->_loss_type == 0)
        {
            this->_loss_fn = std::make_shared<CrossEntropyLossImpl>(torch::nn::CrossEntropyLossOptions().reduction(torch::kMean));
        }
        else if (this->_loss_type == 1)
        {
            this->_loss_fn = std::make_shared<FocalLossImpl>(this->_gamma);
        }
        else
        {
            ULOGGER_FATAL("Loss function type error: received %d", this->_loss_type);
        }
    }

    void TrainThread::make_replay_memory()
    {
        if (this->_replay_memory_type == 0)
        {
            this->_replay_memory = std::make_shared<ReservoirSamplingBuffer>(this->_replay_memory_size);
        }
        else if (this->_replay_memory_type == 1)
        {
            this->_replay_memory = std::make_shared<ClassBalancedBuffer>(this->_replay_memory_size);
        }
        else
        {
            ULOGGER_FATAL("Replay memory type error: received %d", this->_replay_memory_type);
        }
        std::vector<size_t> ids;
        torch::Tensor data;
        torch::Tensor labels;
        this->loadReplayMemory(ids, data, labels);
        if (ids.size() > 0)
        {
            std::shared_ptr<LatentDataset> loaded_dataset = std::make_shared<LatentDataset>(ids, data, labels);
            this->_replay_memory->update(loaded_dataset);
        }
    }

    torch::Tensor TrainThread::compute_weights(const torch::Tensor &samples_per_class)
    {
        if (this->_weighting_method == 1)
        {
            return compute_simple_weights(samples_per_class);
        }
        else if (this->_weighting_method == 2)
        {
            return compute_effective_weights(samples_per_class, this->_beta);
        }
        else
        {
            ULOGGER_FATAL("Weighting method error: received %d", this->_weighting_method);
        }
    }

    float TrainThread::loop(const auto &dataloader,
                            // TODo accuracy,
                            size_t num_classes,
                            bool train)
    {

        float epoch_loss = 0.0;
        size_t n_batch = 0;
        {
            torch::GradMode::set_enabled(train);
            for (const auto &batch : *dataloader)
            {
                ULOGGER_DEBUG("Batch=%d", n_batch + 1);

                torch::Tensor x = batch.data.to(this->_device, true);
                torch::Tensor y = batch.target.to(this->_device, true);

                for(size_t i = 0; i < y.size(0); i++)
                {
                    ULOGGER_DEBUG("Label in this batch=%d", (int)y[i].item<int64_t>());
                }

                ULOGGER_DEBUG("Batch size %d", x.size(0));

                ULOGGER_DEBUG("RAM usage before trainable features extraction=%ld", UProcessInfo::getMemoryUsage());
                torch::Tensor features = this->_model->feature_extractor->extract_trainable_features(x);
                ULOGGER_DEBUG("Extracted trainable features of shape [%d, %d]", features.size(0), features.size(1));
                ULOGGER_DEBUG("RAM usage before forward=%ld", UProcessInfo::getMemoryUsage());
                torch::Tensor predictions = this->_model->classifier->forward(features);
                
                
                ULOGGER_DEBUG("Output predictions of shape [%d, %d]", predictions.size(0), predictions.size(1));
                ULOGGER_DEBUG("RAM usage before onehot=%ld", UProcessInfo::getMemoryUsage());
                torch::Tensor one_hot_labels = torch::one_hot(y, num_classes).to(this->_device);
                ULOGGER_DEBUG("One hot labels of shape [%d, %d]", one_hot_labels.size(0), one_hot_labels.size(1));
                for(size_t i = 0; i < one_hot_labels.size(0); i++)
                {
                    std::stringstream ss;
                    ss << "Predictions ";
                    ss << predictions[i];
                    ss << " and one hot labels ";
                    ss << one_hot_labels[i];
                    ULOGGER_DEBUG("%s", ss.str().c_str());
                }
                ULOGGER_DEBUG("RAM usage before loss=%ld", UProcessInfo::getMemoryUsage());
                torch::Tensor loss = this->_loss_fn->compute(predictions.to(torch::kFloat), one_hot_labels.to(torch::kFloat));
                ULOGGER_DEBUG("RAM usage before backward=%ld", UProcessInfo::getMemoryUsage());
                if (train)
                {
                    this->_optimizer->zero_grad();
                    loss.backward();
                    this->_optimizer->step();
                }

                epoch_loss += loss.item().toFloat();
                n_batch++;
            }
        }
        return epoch_loss /= n_batch;
    }

    void TrainThread::saveReplayMemory(const std::vector<size_t> &ids, const torch::Tensor &data, const std::unordered_set<int> &ids_in_memory) const
    {
        if (this->_dbDriver)
        {
            std::vector<std::vector<char>> serialized_data(ids.size());
            for (size_t i = 0; i < ids.size(); i++)
            {
                std::vector<char> serialized_tensor = torch::pickle_save(data[i].clone());
                serialized_data[i] = serialized_tensor;
            }
            this->_dbDriver->saveReplayMemory(ids, serialized_data, ids_in_memory);
        }
    }

    void TrainThread::loadReplayMemory(std::vector<size_t> &ids, torch::Tensor &data, torch::Tensor &labels) const
    {
        if (this->_dbDriver)
        {
            std::vector<std::vector<char>> serialized_data;
            std::vector<int> loaded_labels;
            std::vector<torch::Tensor> loaded_tensors;

            this->_dbDriver->loadReplayMemory(ids, serialized_data, loaded_labels);

            if(ids.size() > 0)
            {
                for (size_t i = 0; i < ids.size(); i++)
                {
                    loaded_tensors.emplace_back(torch::pickle_load(serialized_data[i]).toTensor());
                }

                data = torch::stack(loaded_tensors);
                labels = torch::from_blob(loaded_labels.data(), {static_cast<long>(loaded_labels.size())}, torch::kInt32).to(torch::kLong);
            }

            
        }
    }

}