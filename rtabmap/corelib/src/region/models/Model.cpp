#include "rtabmap/core/region/models/Model.h"
#include "rtabmap/utilite/ULogger.h"
#include "rtabmap/core/region/utils.h"

#include <exception>
#include <typeinfo>
#include <stdexcept>

namespace rtabmap
{

    ModelImpl::ModelImpl() : feature_extractor(FeatureExtractor()), classifier(IncrementalLinear()) {}

    ModelImpl::ModelImpl(const FeatureExtractor &feature_extractor,
                         const IncrementalLinear &classifier,
                         const std::string &model_path) : feature_extractor(feature_extractor),
                                                          classifier(classifier)
    {

        register_module("feature_extractor", this->feature_extractor);
        register_module("classifier", this->classifier);

        // load the state dict
        if (!model_path.empty())
        {
            this->load_state_dict(model_path);
        }

        // set freezed part
        this->set_freezed_part();
    }

    torch::Tensor ModelImpl::forward(const torch::Tensor &input)
    {
        // To use only in inference
        torch::Tensor x = this->feature_extractor->extract_freezed_features(input);
        x = this->feature_extractor->extract_trainable_features(x);
        x = this->classifier->forward(x);
        return x;
    }

    void ModelImpl::adapt(const torch::Tensor &classes_in_this_experience)
    {
        this->classifier->adapt(classes_in_this_experience);
    }

    void ModelImpl::set_freezed_part()
    {
        for (const auto &p : this->feature_extractor->freezed_part->named_parameters())
        {
            p.value().requires_grad_(false);
        }
        this->feature_extractor->freezed_part->eval();
    }

    // create a deep clone of the model
    std::shared_ptr<ModelImpl> ModelImpl::clone()
    {
        auto clone = std::make_shared<ModelImpl>(FeatureExtractor(),
                                                 IncrementalLinear(this->classifier->linear->options.in_features(),
                                                                   this->classifier->linear->options.out_features()));

        std::string data;
        {
            std::ostringstream oss;
            torch::serialize::OutputArchive archive;
            this->save(archive);
            archive.save_to(oss);
            data = oss.str();
        }

        {
            std::istringstream iss(data);
            torch::serialize::InputArchive archive;
            archive.load_from(iss);
            clone->load(archive);    
        }

        return clone;
    }

    // load the state dict
    void ModelImpl::load_state_dict(const std::string &model_path)
    {
        try
        {
            std::ifstream file(model_path, std::ios::binary);
            std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            file.close();
            torch::IValue ivalue = torch::pickle_load(data);
            auto dict = ivalue.toGenericDict();

            // for (const auto &e : dict)
            // {
            //     std::string key = e.key().toString().get()->string();
            //     if (this->named_buffers().contains(key))
            //     {
            //         this->named_buffers().update(key, e.value().toTensor());
            //     }
            //     else if (this->named_parameters().contains(key))
            //     {
            //         this->named_parameters().update(key, e.value().toTensor());
            //     }
            // }

            for (const auto &p : this->named_buffers())
            {
                if (dict.contains(p.key()))
                {
                    p.value().set_data(dict.at(p.key()).toTensor());
                }
            }

            for (const auto &p : this->named_parameters())
            {
                if (dict.contains(p.key()))
                {
                    p.value().set_data(dict.at(p.key()).toTensor());
                }
            }
        }

        catch (const c10::Error &e)
        {
            std::cerr << e.what() << "\n";
            std::cerr << "Error loading the state_dict\n";
        }
    }

    // save the state dict
    void ModelImpl::save_state_dict(const std::string &model_path)
    {
        try
        {
            torch::Dict<std::string, torch::Tensor> dict;

            for (const auto &p : this->named_buffers())
            {
                dict.insert(p.key(), p.value());
            }

            for (const auto &p : this->named_parameters())
            {
                dict.insert(p.key(), p.value());
            }

            // save_tensor_serialized(model_path, dict);

            std::vector<char> state_dict = torch::pickle_save(dict);
            std::ofstream file(model_path, std::fstream::out | std::ios::binary);
            file.write(state_dict.data(), state_dict.size());
            file.close();
        }
        catch (const c10::Error &e)
        {
            std::cerr << e.what() << "\n";
            std::cerr << "Error saving the state_dict\n";
        }
    }

}