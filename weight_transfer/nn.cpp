#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

/********************************************************** Helpers **********************************************************/

void print_vector(const std::vector<double>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i != vec.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    std::cout << "Length: " << vec.size() << std::endl;
}

/********************************************************** Read weights **********************************************************/

std::vector<double> load_csv_vector(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    std::string line;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        while (getline(ss, val, ',')) {
            data.push_back(std::stod(val));
        }
    }
    return data;
}


void save_vector_to_csv(const std::vector<double>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        file << data[i];
        if (i != data.size() - 1)
            file << ",";
    }

    file << "\n";
    file.close();
}

/********************************************************** Layer **********************************************************/

class Layer {
public:
    virtual ~Layer() {}
    virtual std::vector<double> forward(const std::vector<double>& input) = 0;
};


class Tanh : public Layer {
public:
    std::vector<double> forward(const std::vector<double>& input) override {
        std::vector<double> output;
        output.reserve(input.size());
        for (double val : input) {
            output.push_back(std::tanh(val));
        }
        return output;
    }
};


class ReLU : public Layer {
public:
    std::vector<double> forward(const std::vector<double>& input) override {
        std::vector<double> output;
        output.reserve(input.size());
        for (double val : input) {
            output.push_back(val > 0.0f ? val : 0.0f);
        }
        return output;
    }
};


class Conv2D : public Layer {
public:
    int in_channels, out_channels, kernel_size, input_height, input_width, padding;
    std::vector<double> weights;  // Shape: [out_channels * in_channels * kernel_size * kernel_size]
    std::vector<double> bias;     // Shape: [out_channels]

    Conv2D(int in_ch, int out_ch, int k_size, int height, int width, int pad = 0,
           const std::string& weight_file = "", const std::string& bias_file = "")
        : in_channels(in_ch), out_channels(out_ch), kernel_size(k_size), 
          input_height(height), input_width(width), padding(pad)
    {
        if (!weight_file.empty()) {
            weights = load_csv_vector(weight_file);
            if (weights.size() != out_channels * in_channels * kernel_size * kernel_size) {
                std::cerr << "Error: Weights size mismatch!" << std::endl;
            }
        }
        if (!bias_file.empty()) {
            bias = load_csv_vector(bias_file);
            if (bias.size() != out_channels) {
                std::cerr << "Error: Bias size mismatch!" << std::endl;
            }
        }
    }

    std::vector<double> pad_input(const std::vector<double>& input) {
        int padded_height = input_height + 2 * padding;
        int padded_width = input_width + 2 * padding;
        std::vector<double> padded(in_channels * padded_height * padded_width, 0.f);

        for (int c = 0; c < in_channels; ++c) {
            for (int h = 0; h < input_height; ++h) {
                for (int w = 0; w < input_width; ++w) {
                    padded[c * padded_height * padded_width + (h + padding) * padded_width + (w + padding)] =
                        input[c * input_height * input_width + h * input_width + w];
                }
            }
        }
        return padded;
    }

    std::vector<double> forward(const std::vector<double>& input) override {
        int padded_height = input_height + 2 * padding;
        int padded_width = input_width + 2 * padding;
        auto padded_input = pad_input(input);

        int output_height = input_height;  // padding so output size = input size
        int output_width = input_width;

        std::vector<double> output(out_channels * output_height * output_width, 0.f);

        for (int oc = 0; oc < out_channels; ++oc) {
            for (int h = 0; h < output_height; ++h) {
                for (int w = 0; w < output_width; ++w) {
                    double sum = (bias.size() == out_channels) ? bias[oc] : 0.f;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int ih = h + kh;
                                int iw = w + kw;
                                int input_idx = ic * padded_height * padded_width + ih * padded_width + iw;
                                int weight_idx = oc * in_channels * kernel_size * kernel_size
                                                 + ic * kernel_size * kernel_size
                                                 + kh * kernel_size + kw;
                                sum += weights[weight_idx] * padded_input[input_idx];
                            }
                        }
                    }
                    output[oc * output_height * output_width + h * output_width + w] = sum;
                }
            }
        }
        return output;
    }
};


class Linear : public Layer{
public:
    int input_size;
    int output_size;
    std::vector<double> weights; // Shape: [output_size * input_size]
    std::vector<double> bias;    // Shape: [output_size]

    Linear(int in_size, int out_size, const std::string& weight_file = "", const std::string& bias_file = "")
        : input_size(in_size), output_size(out_size)
    {
        if (!weight_file.empty()) {
            weights = load_csv_vector(weight_file);
            if (weights.size() != output_size * input_size) {
                std::cerr << "Error: Weights size mismatch!" << std::endl;
            }
        }
        if (!bias_file.empty()) {
            bias = load_csv_vector(bias_file);
            if (bias.size() != output_size) {
                std::cerr << "Error: Bias size mismatch!" << std::endl;
            }
        }
    }

    std::vector<double> forward(const std::vector<double>& input) override {
        if (input.size() != input_size) {
            std::cerr << "Error: Input size mismatch!" << std::endl;
            return {};
        }
        std::vector<double> output(output_size, 0.f);

        for (int o = 0; o < output_size; ++o) {
            double sum = (bias.size() == output_size) ? bias[o] : 0.f;
            for (int i = 0; i < input_size; ++i) {
                sum += weights[o * input_size + i] * input[i];
            }
            output[o] = sum;
        }
        return output;
    }
};


class BatchNorm2D : public Layer {
public:
    int num_channels;
    std::vector<double> gamma;         
    std::vector<double> beta;         
    std::vector<double> running_mean; 
    std::vector<double> running_var;  
    double eps = 1e-5;            

    BatchNorm2D(int channels,
                const std::string& weight_file = "",
                const std::string& bias_file = "",
                const std::string& mean_file = "",
                const std::string& var_file = "")
        : num_channels(channels)
    {
        gamma        = load_csv_vector(weight_file);
        beta         = load_csv_vector(bias_file);
        running_mean = load_csv_vector(mean_file);
        running_var  = load_csv_vector(var_file);
    }

    std::vector<double> forward(const std::vector<double>& input) override {
        std::vector<double> output(input.size());
        int spatial_size = input.size() / num_channels;

        #pragma omp parallel for
        for (int c = 0; c < num_channels; ++c) {
            double g = gamma[c];
            double b = beta[c];
            double m = running_mean[c];
            double v = running_var[c];

            double denom = 1.0 / std::sqrt(v + eps);

            for (int i = 0; i < spatial_size; ++i) {
                int idx = c * spatial_size + i;
                output[idx] = g * ((input[idx] - m) * denom) + b;
            }
        }

        return output;
    }
};


class ResBlock : public Layer {
public:
    Conv2D* conv1;
    BatchNorm2D* bn1;
    ReLU* relu1;
    Conv2D* conv2;
    BatchNorm2D* bn2;
    ReLU* relu_out;

    ResBlock(int index, int num_hidden, int height, int width) {
    std::string base = "weight_transfer/weights_csv/backBone." + std::to_string(index) + ".";

        conv1 = new Conv2D(num_hidden, num_hidden, 3, height, width, 1, base + "conv1.weight.csv", base + "conv1.bias.csv");
        bn1 = new BatchNorm2D(num_hidden, base + "bn1.weight.csv", base + "bn1.bias.csv", base + "bn1.running_mean.csv", base + "bn1.running_var.csv");
        relu1 = new ReLU();
        conv2 = new Conv2D(num_hidden, num_hidden, 3, height, width, 1, base + "conv2.weight.csv", base + "conv2.bias.csv");
        bn2 = new BatchNorm2D(num_hidden, base + "bn2.weight.csv", base + "bn2.bias.csv", base + "bn2.running_mean.csv", base + "bn2.running_var.csv");
        relu_out = new ReLU();
    }

    ~ResBlock() {
        delete conv1;
        delete bn1;
        delete relu1;
        delete conv2;
        delete bn2;
        delete relu_out;
    }

    std::vector<double> forward(const std::vector<double>& input) override {
        std::vector<double> residual = input;

        std::vector<double> x = conv1->forward(input);
        x = bn1->forward(x);
        x = relu1->forward(x);

        x = conv2->forward(x);
        x = bn2->forward(x);

        // Skip-connection
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += residual[i];
        }

        x = relu_out->forward(x);
        return x;
    }
};

/********************************************************** Net **********************************************************/

class Net {
public:
    std::vector<Layer*> layers;

    ~Net() {
        for (auto* layer : layers) delete layer;
    }

    void add(Layer* layer) {
        layers.push_back(layer);
    }

    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> output = input;
        for (auto* layer : layers) {
            // std::cout << "Input size: " << output.size();
            output = layer->forward(output);
            // std::cout << " | Output size: " << output.size() << std::endl;
        }
        return output;
    }
};


Net create_base_model(int num_hidden = 64, int game_width = 15, int game_height = 15, int num_blocks = 4, int num_players = 2) {
    Net model;

    // Start block
    model.add(new Conv2D(num_players+2, num_hidden, 3, game_width, game_height, 1, "weight_transfer/weights_csv/startBlock.0.weight.csv", "weight_transfer/weights_csv/startBlock.0.bias.csv"));
    model.add(new BatchNorm2D(num_hidden, "weight_transfer/weights_csv/startBlock.1.weight.csv", "weight_transfer/weights_csv/startBlock.1.bias.csv",
        "weight_transfer/weights_csv/startBlock.1.running_mean.csv", "weight_transfer/weights_csv/startBlock.1.running_var.csv"));
    model.add(new ReLU());

    // Res blocks
    for (int i = 0; i < 4; i++) {
        model.add(new ResBlock(i, num_hidden, game_width, game_height));
    }  

    return model;
}


Net create_policy_head(int num_hidden, int game_width, int game_height, int game_action_size, int num_players) {
    Net policy_head;

    policy_head.add(new Conv2D(num_hidden, 32, 3, game_width, game_height, 1, "weight_transfer/weights_csv/policyHead.0.weight.csv", "weight_transfer/weights_csv/policyHead.0.bias.csv"));
    policy_head.add(new BatchNorm2D(32, "weight_transfer/weights_csv/policyHead.1.weight.csv", "weight_transfer/weights_csv/policyHead.1.bias.csv",
        "weight_transfer/weights_csv/policyHead.1.running_mean.csv", "weight_transfer/weights_csv/policyHead.1.running_var.csv"));
    policy_head.add(new ReLU());
    policy_head.add(new Linear(32 * game_width * game_height, game_action_size * num_players, "weight_transfer/weights_csv/policyHead.4.weight.csv", "weight_transfer/weights_csv/policyHead.4.bias.csv"));

    return policy_head;
}


Net create_value_head(int num_hidden, int game_width, int game_height, int num_players) {
    Net value_head;

    value_head.add(new Conv2D(num_hidden, 3, 3, game_width, game_height, 1, "weight_transfer/weights_csv/valueHead.0.weight.csv", "weight_transfer/weights_csv/valueHead.0.bias.csv"));
    value_head.add(new BatchNorm2D(3, "weight_transfer/weights_csv/valueHead.1.weight.csv", "weight_transfer/weights_csv/valueHead.1.bias.csv",
        "weight_transfer/weights_csv/valueHead.1.running_mean.csv", "weight_transfer/weights_csv/valueHead.1.running_var.csv"));
    value_head.add(new ReLU());
    value_head.add(new Linear(3 * game_width * game_height, num_players, "weight_transfer/weights_csv/valueHead.4.weight.csv", "weight_transfer/weights_csv/valueHead.4.bias.csv"));
    value_head.add(new Tanh());

    return value_head;
}

/********************************************************** Main **********************************************************/


int main() {
    std::vector<double> board = {
        0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

        1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

        0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

        0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,
        0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
    };
    
    auto start = std::chrono::high_resolution_clock::now();

    Net base_model = create_base_model(64, 15, 15, 4, 2);
    Net policy_head = create_policy_head(64, 15, 15, 15*15, 2);
    Net value_head = create_value_head(64, 15, 15, 2);

    auto load = std::chrono::high_resolution_clock::now();
    auto duration_load = std::chrono::duration_cast<std::chrono::milliseconds>(load - start).count();

    std::cout << "Duration load: " << duration_load << " ms" << std::endl;

    std::vector<double> base_output = base_model.forward(board);
    std::vector<double> policy = policy_head.forward(base_output);
    std::vector<double> value = value_head.forward(base_output);

    auto forward = std::chrono::high_resolution_clock::now();
    auto duration_forward = std::chrono::duration_cast<std::chrono::milliseconds>(forward - load).count();

    std::cout << "Duration forward: " << duration_forward << " ms" << std::endl;

    // print_vector(policy);
    print_vector(value);

    save_vector_to_csv(policy, "weight_transfer/policy.csv");
}