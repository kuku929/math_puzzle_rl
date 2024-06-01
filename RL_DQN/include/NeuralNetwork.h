#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath> //pow
#include <map>
using namespace std;
extern ofstream dout;
extern ofstream plt;
extern float completion_reward;
typedef float (*activation_funcptr)(float);

struct Network{
	vector<vector<float>> weights;
	vector<vector<float>> bias;
	vector<vector<float>> gradient_sum;
	vector<vector<float>> bias_grad_sum;
	vector<size_t> layer_sizes;
	map<string, activation_funcptr> activation_func_map;
	vector<string> activation_func;

	Network(){};
	Network(vector<size_t> &l, vector<string> &activ_func);
	Network(vector<size_t> &l, vector<string> &activ_func, vector<vector<float>> &w,vector<vector<float>> &b); 
	Network& operator=(const Network&) = default;
	
	vector<float> predict(const vector<float> &input); //does one forward pass

	void fit(const vector<float> &input, const vector<float> &true_output, const float learning_rate, const float grad_decay, int verbose, int optimize); //does one backward pass 

};

//activation function implementations
float sigmoid(float weighted_sum);
float ReLU(float weighted_sum);
float Leaky(float weighted_sum);
float Linear(float weighted_sum);

float Fdash(float output, string activation_function); //finds derivative of activation function 
