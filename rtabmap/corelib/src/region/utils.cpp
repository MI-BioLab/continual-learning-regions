#include "rtabmap/core/region/utils.h"
#include "rtabmap/core/region/losses/FocalLoss.h"
#include <opencv4/opencv2/opencv.hpp>
#include <rtabmap/utilite/ULogger.h>

namespace rtabmap
{

    torch::Tensor image_to_tensor(const cv::Mat &image, const int64_t &image_width, const int64_t &image_height)
    {
        cv::Mat img;
        cv::resize(image, img, cv::Size(image_width, image_height));
        img.convertTo(img, CV_32FC3, 1 / 255.0);
        torch::Tensor tensor = torch::from_blob(img.data, {1, img.rows, img.cols, img.channels()}, c10::kFloat);
        tensor = tensor.permute({0, 3, 1, 2});
        return tensor.clone();
    }

    torch::Tensor compute_simple_weights(const torch::Tensor &samples_per_class)
    {
        torch::Tensor samples_copy = samples_per_class.clone();
        torch::Tensor mask = samples_per_class == 0;
        samples_copy.index_put_({mask}, torch::full({}, static_cast<int64_t>(1e10)));
        return 1.0 / samples_copy.to(torch::kFloat);
    }

    torch::Tensor compute_effective_weights(const torch::Tensor &samples_per_class, float beta)
    {
        torch::Tensor effective_sum = 1.0 - torch::pow(beta, samples_per_class.to(torch::kFloat));
        torch::Tensor mask = (samples_per_class == 0).to(torch::kLong);
        effective_sum.to(torch::kLong).index_put_({mask}, torch::full({}, static_cast<int64_t>(1e10)));
        torch::Tensor weights = (1.0 - beta) / effective_sum.to(torch::kFloat);
        return weights / torch::sum(weights) * samples_per_class.size(0);
    }

    void save_tensor_serialized(const std::string &file_path, const c10::IValue &ivalue)
    {
        std::vector<char> state_dict = torch::pickle_save(ivalue);
        std::ofstream file(file_path, std::fstream::out | std::ios::binary);
        file.write(state_dict.data(), state_dict.size());
        file.close();
    }

    c10::IValue load_tensor_deserialized(const std::string &file_path)
    {
        std::ifstream file(file_path, std::ios::binary);
        std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        return torch::pickle_load(data);
    }
}