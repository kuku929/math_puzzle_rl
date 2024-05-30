#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath> //pow
using namespace std;
extern ofstream dout;
extern ofstream plt;
extern float completion_reward;

struct Network{
	vector<vector<float>> weights{vector<vector<float>>(2)};
	vector<vector<float>> bias{vector<vector<float>>(2)};
	vector<vector<float>> gradient_sum{vector<vector<float>>(2)};
	vector<vector<float>> bias_grad_sum{vector<vector<float>>(2)};
	int input_size, hidden_size, output_size;

	Network(){};
	Network(int in, int h, int o);
	Network(int in, int h, int o, vector<vector<float>> &w,vector<vector<float>> &b); 
	Network& operator=(const Network&) = default;
	
	vector<float> predict(const vector<float> &input);

	void fit(const vector<float> &input, const float true_output, const float learning_rate, const float grad_decay, const int action, int verbose, int optimize);

};

float sigmoid(float weighted_sum);
float ReLU(float weighted_sum);
float Leaky(float weighted_sum);
float Fdash(float output, string activation_function); 
void beautiful_print(const vector<float> &input, vector<float> &hidden_weighted_sum, vector<float> &hidden_output, vector<float> &output_weighted_sum, vector<float> &output,vector<vector<float>> weights, vector<vector<float>> bias);
