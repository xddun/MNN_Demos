#include <iostream>
#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>

#define IMAGE_VERIFY_SIZE 224
#define CLASSES_SIZE 1000
#define INPUT_NAME "input"
#define OUTPUT_NAME "output"

// mnn model input=[1, 3, 224, 224], output=[1, 1000]
int main(int argc, char* argv[]){
    if(argc < 2){
        printf("Usage:\n\t%s mnn_model_path image_path\n", argv[0]);
        return -1;
    }

    // create net and session
    const char *mnn_model_path = argv[1];
    const char *image_path = argv[2];

    auto mnnNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_model_path));
    MNN::ScheduleConfig netConfig;
    netConfig.type = MNN_FORWARD_CPU;
    netConfig.numThread = 4;
    auto session = mnnNet->createSession(netConfig);

    auto input = mnnNet->getSessionInput(session, INPUT_NAME);
    if (input->elementSize() <= 4) {
        mnnNet->resizeTensor(input, {1, 3, IMAGE_VERIFY_SIZE, IMAGE_VERIFY_SIZE});
        mnnNet->resizeSession(session);
    }
    std::cout << "input shape: " << input->shape()[0] << " " << input->shape()[1] << " " << input->shape()[2] << " " << input->shape()[3] << std::endl;

    // preprocess image
    MNN::Tensor givenTensor(input, MNN::Tensor::CAFFE);
    // const int inputSize = givenTensor.elementSize();
    // std::cout << inputSize << std::endl;
    auto inputData = givenTensor.host<float>();
    cv::Mat bgr_image = cv::imread(image_path);
    cv::Mat norm_image;
    cv::resize(bgr_image, norm_image, cv::Size(IMAGE_VERIFY_SIZE, IMAGE_VERIFY_SIZE));
    for(int k = 0; k < 3; k++){
        for(int i = 0; i < norm_image.rows; i++){
            for(int j = 0; j < norm_image.cols; j++){
                const auto src = norm_image.at<cv::Vec3b>(i, j)[k];
                auto dst = 0.0;
                if(k == 0) dst = (float(src) / 255.0f - 0.485) / 0.229;
                if(k == 1) dst = (float(src) / 255.0f - 0.456) / 0.224;
                if(k == 2) dst = (float(src) / 255.0f - 0.406) / 0.225;
                inputData[k * IMAGE_VERIFY_SIZE * IMAGE_VERIFY_SIZE + i * IMAGE_VERIFY_SIZE + j] = dst;
            }
        }
    }
    input->copyFromHostTensor(&givenTensor);

    // run session
    mnnNet->runSession(session);

    // get output data
    auto output = mnnNet->getSessionOutput(session, OUTPUT_NAME);
    // std::cout << "output shape: " << output->shape()[0] << " " << output->shape()[1] << std::endl;
    auto output_host = std::make_shared<MNN::Tensor>(output, MNN::Tensor::CAFFE);
    output->copyToHostTensor(output_host.get());
    auto values = output_host->host<float>();
    
    // post process
    std::vector<float> output_values;
    auto exp_sum = 0.0;
    auto max_index = 0;
    for(int i = 0; i < CLASSES_SIZE; i++){
        if(values[i] > values[max_index]) max_index = i;
        output_values.push_back(values[i]);
        exp_sum += std::exp(values[i]);
    }
    std::cout << "cls id: " << max_index << std::endl;
    std::cout << "cls prob: " << std::exp(output_values[max_index]) / exp_sum << std::endl;

    return 0;
}