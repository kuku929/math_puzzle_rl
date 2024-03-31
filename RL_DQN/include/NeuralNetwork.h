#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath> //pow
using namespace std;
extern ofstream dout;
extern float completion_reward;

struct Network{
	vector<vector<float>> weights{vector<vector<float>>(2)};
	vector<vector<float>> bias{vector<vector<float>>(2)};
	vector<vector<float>> gradient_sum{vector<vector<float>>(2)};
	vector<vector<float>> bias_grad_sum{vector<vector<float>>(2)};
	int input_size, hidden_size, output_size;

	Network(){};
	Network(int input_size, int hidden_size, int output_size);
	Network(int input_size, int hidden_size, int output_size, vector<vector<float>> &weights,vector<vector<float>> &bias); 
	
	vector<float> predict(const vector<float> &input);

	void fit(const vector<float> &input, const float true_output, const float learning_rate, const int action, int verbose);

	//Network &operator=(Network& some_net);
};

float sigmoid(float weighted_sum);
float ReLU(float weighted_sum);
float Leaky(float weighted_sum);
float Fdash(float output, string activation_function); 

