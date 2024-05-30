#include "NeuralNetwork.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath> //pow
#include <chrono>
#include <iomanip>
using namespace std;
extern ofstream dout;

Network::Network(vector<int> &l, vector<string> &activ_func) : layer_sizes(l), activation_func(activ_func){
	float range = 2.0;//4*pow(6.0/(float)(input_size+output_size), 0.5); 
	//srand(chrono::high_resolution_clock::now().time_since_epoch().count());
	
	this->weights = vector<vector<float>>(static_cast<int>(layer_sizes.size()) - 1);
	this->bias = vector<vector<float>>(static_cast<int>(layer_sizes.size()) - 1);
	this->gradient_sum = vector<vector<float>>(static_cast<int>(layer_sizes.size()) - 1);
	this->bias_grad_sum = vector<vector<float>>(static_cast<int>(layer_sizes.size()) - 1);

	//iterate each layer starting from 0th layer to second to last layer
	for(size_t layer_index=0; layer_index < layer_sizes.size()-1; ++layer_index){ 
		const int current_layer_size = layer_sizes[layer_index];
		const int next_layer_size = layer_sizes[layer_index+1];
		
		//set the bias and weights
		for(int i=0;i<next_layer_size;++i){
			this->bias[layer_index].push_back(0.0f); //bias for the next layer
			this->bias_grad_sum[layer_index].push_back(0.0f); 
			for(int j=0;j<current_layer_size;++j){
				float normalized_value = static_cast<float>(rand())/static_cast<float>(RAND_MAX)-0.5f; //between (-0.5,0.5)	
				this->weights[layer_index].push_back(normalized_value*range); 
				this->gradient_sum[layer_index].push_back(0.0f); 
			}
		}
	}


	activation_func_map.emplace("sigmoid", &sigmoid);
	activation_func_map.emplace("ReLU", &ReLU);
	activation_func_map.emplace("Leaky", &Leaky);
	activation_func_map.emplace("Linear", &Linear);

}

Network::Network(vector<int> &l, vector<string> &activ_func, vector<vector<float>> &w, vector<vector<float>> &b):layer_sizes(l), activation_func(activ_func){
	this->weights=w;
	this->bias=b;
	this->gradient_sum = vector<vector<float>>(static_cast<int>(layer_sizes.size()) - 1);
	this->bias_grad_sum = vector<vector<float>>(static_cast<int>(layer_sizes.size()) - 1);

	//explicitly setting the values to 0, maybe the default are not 0
	for(size_t layer_index=0; layer_index < layer_sizes.size()-1; ++layer_index){
		const int current_layer_size = layer_sizes[layer_index];
		const int next_layer_size = layer_sizes[layer_index+1];
		
		//set the gradient sums 
		for(int i=0;i<next_layer_size;++i){
			this->bias_grad_sum[layer_index].push_back(0.0f); 
			for(int j=0;j<current_layer_size;++j){
				this->gradient_sum[layer_index].push_back(0.0f); 
			}
		}
	}

	activation_func_map.emplace("sigmoid", &sigmoid);
	activation_func_map.emplace("ReLU", &ReLU);
	activation_func_map.emplace("Leaky", &Leaky);
	activation_func_map.emplace("Linear", &Linear);
}

vector<float> Network::predict(const vector<float> &input){
	//stores all the outputs of previous layers, better than keeping two vectors input and output,which requires copying after each iteration
	vector<vector<float>> layer_outputs(layer_sizes.size()); 
	layer_outputs[0] = input;
	float weighted_sum;

	//start with the second layer and go until the output layer
	for(size_t layer_index=1; layer_index < layer_sizes.size(); ++layer_index){
		const int current_layer_size = layer_sizes[layer_index]; 
		const int prev_layer_size = layer_sizes[layer_index-1];

		for(int i=0;i<current_layer_size;++i){
			weighted_sum = this->bias[layer_index-1][i];
			for(int j=0;j<prev_layer_size;++j){
				weighted_sum += weights[layer_index-1][i*prev_layer_size + j]*layer_outputs[layer_index-1][j];
			}
			//ATTENTION : all are sigmoid, change it!
			layer_outputs[layer_index].push_back(activation_func_map[activation_func[layer_index-1]](weighted_sum));
		}
	}

	return layer_outputs.back();
}

void Network::fit(const vector<float> &input, const vector<float> &true_output, const float learning_rate,const float grad_decay, int verbose, int optimize){
	/*
	 *sanity check
	 */
	if(input.size()!=layer_sizes.front()){
		dout << "Mismatch of input dimensions!\n";
		return;
	}
	

	/*
	 *predicting
	 */
	vector<vector<float>> layer_outputs(layer_sizes.size()); 
	layer_outputs[0] = input;
	float weighted_sum;

	//start with the second layer and go until the output layer
	for(size_t layer_index=1; layer_index < layer_sizes.size(); ++layer_index){
		const int current_layer_size = layer_sizes[layer_index]; 
		const int prev_layer_size = layer_sizes[layer_index-1];

		for(int i=0;i<current_layer_size;++i){
			weighted_sum = this->bias[layer_index-1][i];
			for(int j=0;j<prev_layer_size;++j){
				weighted_sum += this->weights[layer_index-1][i*prev_layer_size + j]*layer_outputs[layer_index-1][j];
			}
			//ATTENTION : all are sigmoid, change it!
			layer_outputs[layer_index].push_back(activation_func_map[activation_func[layer_index-1]](weighted_sum));
		}
	}
	

	/*
	 *backpropagation
	 */
	
	//this will store the deltas that will be propagated backwards
	vector<vector<float>> layer_deltas(layer_sizes.size() - 1);
	
	//storing the initial deltas
	for(int i=0;i < layer_sizes.back();++i){
		layer_deltas[layer_sizes.size() - 2].push_back((true_output[i]-layer_outputs.back()[i])*Fdash(layer_outputs.back()[i], activation_func.back()));
	}

	//backpropagating the error
	float layer_delta_sum;
	for(int layer_index=layer_sizes.size() - 2;layer_index > 0 ;--layer_index){
		const int current_layer_size = layer_sizes[layer_index]; 
		const int prev_layer_size = layer_sizes[layer_index+1]; //since iterating backwards, +1 is previous
		for(int i=0;i < current_layer_size; ++i){ //iterate through each node in the layer
			layer_delta_sum = 0.0f;
			for(int j=0; j<prev_layer_size; ++j){ //find the back-propagated deltas
				layer_delta_sum += this->weights[layer_index][j*current_layer_size + i]*layer_deltas[layer_index][j]; 
				weights[layer_index][j*current_layer_size + i] += learning_rate*layer_deltas[layer_index][j]*layer_outputs[layer_index][i];
			}
			layer_deltas[layer_index-1].push_back(layer_delta_sum*Fdash(layer_outputs[layer_index][i], activation_func[layer_index-1]));
		}
	}

	//update first layer weights
	for(int i=0; i< layer_sizes[1]; ++i){
		for(int j=0; j<layer_sizes[0]; ++j){
			weights[0][i*layer_sizes[0] + j] += learning_rate*layer_deltas[0][i]*layer_outputs[0][j];
		}
	}
	
	return;
}

float sigmoid(float weighted_sum){
	float e_x=exp(weighted_sum);
	return e_x/(1.0f+e_x);
}

float ReLU(float weighted_sum){
	if(weighted_sum>0.0f)return weighted_sum;
	return 0.0f;
}

float Leaky(float weighted_sum){
	if(weighted_sum<0.0f)return weighted_sum*0.01f;
	return weighted_sum;
}

float Linear(float weighted_sum){
	return weighted_sum;
}

float Fdash(float output, string activation_function){ 
	if(activation_function=="sigmoid")
		return output-output*output;
	else if(activation_function=="ReLU"){
		if(output==0.0f)return 0.0f;
		else return 1.0;
	}		
	else if(activation_function=="Linear")
		return 1.0;
	else if(activation_function=="Leaky"){
		if(output<0)return 0.01f;
		else return 1.0f;
	}
	
	dout << "unidentified activation function!\n";
	return 0.0;
}

////void beautiful_print(const vector<float> &input, vector<float> &hidden_weighted_sum, vector<float> &hidden_output, vector<float> &output_weighted_sum, vector<float> &output,vector<vector<float>> weights, vector<vector<float>> bias){
	////cout << setprecision(1)<< input[0] << " ---"<<weights[0][0]<<'|'<<weights[0][1]<<"--> "<<bias[0][0]<<'/'<<hidden_output[0];
	////cout << "\n                     "<<weights[1][0];
	////cout << "\n                       |--->"<<bias[1][0]<<'/'<<output[0]<<'\n';
	////cout << "                     "<<weights[1][1]<<'\n';
	////cout << input[1] << " ---"<<weights[0][2]<<'|'<<weights[0][3]<<"--> "<<bias[0][1]<<'/'<<hidden_output[1]<<'\n';
	////cout << setprecision(6);
////}
