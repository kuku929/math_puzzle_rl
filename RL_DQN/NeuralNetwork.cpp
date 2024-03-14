#include "NeuralNetwork.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath> //pow
#include <chrono>
using namespace std;

Network::Network(int input_size, int hidden_size, int output_size):input_size(input_size), hidden_size(hidden_size), output_size(output_size){
	float range = 2;//4*pow(6.0/(float)(input_size+output_size), 0.5); 
	//srand(chrono::high_resolution_clock::now().time_since_epoch().count());
	for(int i=0;i<input_size*hidden_size;i++){
		float normalized_value = rand()/(float)RAND_MAX-0.5; //between (-0.5,0.5)	
		//cout << normalized_value*range << ' ';
		this->weights[0].push_back(normalized_value*range); //between (-range/2, range/2)
	}

	//cout << "\n\n";
	
	for(int i=0;i<output_size*hidden_size;i++){
		float normalized_value = rand()/(float)RAND_MAX-0.5; //between (-0.5,0.5)	
		//cout << normalized_value*range << ' ';
		this->weights[1].push_back(normalized_value*range); //between (-range/2, range/2)
	}
	//cout << "\ninitial values end\n\n";
	for(int i=0;i<hidden_size;i++)bias[0].push_back(0.0);
	for(int i=0;i<output_size;i++)bias[1].push_back(0.0);

}
	
vector<float> Network::predict(const vector<float> &input){
	vector<float> hidden_output(hidden_size, 0.0);
	vector<float> output(output_size, 0.0);
	float weighted_sum;
	if(input.size()!=input_size){
		cout << "Mismatch of input dimensions!\n";
		return output;
	}
	for(int i=0;i<hidden_size;i++){
		weighted_sum = bias[0][i];
		for(int j=0;j<input_size;j++){
			weighted_sum += weights[0][i*input_size+j]*input[j];
		}
		//cout << weighted_sum;
		//if(fabs(weighted_sum)>1000)for(auto in: input)cout << in<<' ';
		//cout <<'\n';
		hidden_output[i]=ReLU(weighted_sum);	
	}
	for(int i=0;i<output_size;i++){
		weighted_sum = bias[1][i];
		for(int j=0;j<hidden_size;j++)
			weighted_sum += weights[1][i*hidden_size+j]*hidden_output[j];
		
		output[i]=weighted_sum;	
	}

	return output;

		
}

void Network::fit(const vector<float> &input, const float true_output, const float learning_rate, const int action, int verbose){
	//sanity check
	if(input.size()!=input_size){
		cout << "Mismatch of input dimensions!\n";
		return;
	}
	vector<float> hidden_output(hidden_size, 0);
	vector<float> output(output_size, 0);
	float weighted_sum; 

	//predicting first
	for(int i=0;i<hidden_size;i++){
		weighted_sum = bias[0][i]; //sk
		for(int j=0;j<input_size;j++)
			weighted_sum += weights[0][i*input_size+j]*input[j];
		hidden_output[i]=ReLU(weighted_sum); //yk	
	}

	for(int i=0;i<output_size;i++){
		weighted_sum = bias[1][i]; //s0 
		for(int j=0;j<hidden_size;j++)
			weighted_sum += weights[1][i*hidden_size+j]*hidden_output[j];
		output[i]=weighted_sum; //y0	
	}
	if(verbose){
		cout << "\nprediction: ";
		for(auto t: output)cout << t<<' ';
		cout << "\n";
	}

	//backpropagation
	vector<float> delta_weights;
	float delta_weight, delta, hidden_delta, delta_theta;
	
	delta=(true_output-output[action])*Fdash(output[action], "Linear"); 
	if(verbose){
		cout << "predicted value: "<<output[action]<<' ';
		cout << "Error: "<<delta<<'\n';
	}
	delta_theta=delta*learning_rate; //delta bias of output layer
	this->bias[1][action]+=delta_theta; //updating the bias
	//if(verbose){
		//cout << "bias: ";
		//for(auto b: this->bias[1])cout << b<<' ';
		//cout << '\n';
	//}
	for(int j=0;j<hidden_size;j++){
		delta_weight = delta_theta*hidden_output[j]; 
		delta_weights.push_back(delta_weight);
	}

	for(int i=0;i<hidden_size;i++){
		float hidden_delta_sum=delta*weights[1][action*hidden_size+i];	
		hidden_delta=Fdash(hidden_output[i], "ReLU")*hidden_delta_sum;
		delta_theta=hidden_delta*learning_rate; //delta bias of hidden layer
		this->bias[0][i]+=delta_theta; //update bias
		for(int k=0;k<input_size;k++){
			delta_weight=delta_theta*input[k];
			this->weights[0][i*input_size+k]+=delta_weight; //updating weights
		}
	}

	for(int j=0;j<hidden_size;j++)
		this->weights[1][action*hidden_size+j]+=delta_weights[j]; //updating weights
		
	return;
}

float sigmoid(float weighted_sum){
	float e_x=exp(weighted_sum);
	return e_x/(1.0+e_x);
}

float ReLU(float weighted_sum){
	if(weighted_sum>0.0)return weighted_sum;
	return 0.0;
}

float Leaky(float weighted_sum){
	if(weighted_sum<0.0)return weighted_sum*0.01;
	return weighted_sum;
}

float Fdash(float output, string activation_function){ 
	if(activation_function=="sigmoid")
		return output-output*output;
	else if(activation_function=="ReLU"){
		if(output==0.0)return 0.0;
		else return 1.0;
	}		
	else if(activation_function=="Linear")
		return 1.0;
	else if(activation_function=="Leaky"){
		if(output<0)return 0.01;
		else return 1.0;
	}
	
	cout << "unidentified activation function!\n";
	return 0.0;
}


