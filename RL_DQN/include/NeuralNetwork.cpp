#include "NeuralNetwork.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath> //pow
#include <chrono>
using namespace std;

Network::Network(int input_size, int hidden_size, int output_size):input_size(input_size), hidden_size(hidden_size), output_size(output_size){
	float range = 2.0;//4*pow(6.0/(float)(input_size+output_size), 0.5); 
	//srand(chrono::high_resolution_clock::now().time_since_epoch().count());
	for(int i=0;i<input_size*hidden_size;i++){
		float normalized_value = rand()/(float)RAND_MAX-0.5; //between (-0.5,0.5)	
		this->weights[0].push_back(normalized_value*range); //between (-range/2, range/2)
		this->gradient_sum[0].push_back(0.0); //between (-range/2, range/2)
	}

	//dout << "\n\n";
	
	for(int i=0;i<output_size*hidden_size;i++){
		float normalized_value = rand()/(float)RAND_MAX-0.5; //between (-0.5,0.5)	
		this->weights[1].push_back(normalized_value*range); //between (-range/2, range/2)
		this->gradient_sum[1].push_back(0.0); //between (-range/2, range/2)
	}
	//dout << "\ninitial values end\n\n";
	for(int i=0;i<hidden_size;i++){
		this->bias[0].push_back(0.0);
		this->bias_grad_sum[0].push_back(0.0);	
	}
	for(int i=0;i<output_size;i++){
		this->bias[1].push_back(0.0);
		this->bias_grad_sum[1].push_back(0.0);	
	}

}

Network::Network(int input_size, int hidden_size, int output_size, vector<vector<float>> &weights, vector<vector<float>> &bias):input_size(input_size), hidden_size(hidden_size), output_size(output_size){
	this->weights=weights;
	this->bias=bias;
	for(int i=0;i<input_size*hidden_size;i++){
		this->gradient_sum[0].push_back(0.0); //between (-range/2, range/2)
	}
	for(int i=0;i<output_size*hidden_size;i++){
		this->gradient_sum[1].push_back(0.0); //between (-range/2, range/2)
	}
	for(int i=0;i<hidden_size;i++)this->bias_grad_sum[0].push_back(0.0);
	for(int i=0;i<output_size;i++)this->bias_grad_sum[1].push_back(0.0);
}
vector<float> Network::predict(const vector<float> &input){
	vector<float> hidden_output(hidden_size, 0.0);
	vector<float> output(output_size, 0.0);
	float weighted_sum;
	if(input.size()!=input_size){
		dout << "Mismatch of input dimensions!\n";
		return output;
	}
	for(int i=0;i<hidden_size;i++){
		weighted_sum = bias[0][i];
		for(int j=0;j<input_size;j++){
			weighted_sum += weights[0][i*input_size+j]*input[j];
		}
		//dout << weighted_sum;
		//if(fabs(weighted_sum)>1000)for(auto in: input)dout << in<<' ';
		//dout <<'\n';
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
	/*
	 *sanity check
	 */
	if(input.size()!=input_size){
		dout << "Mismatch of input dimensions!\n";
		return;
	}
	vector<float> hidden_output(hidden_size, 0);
	vector<float> output(output_size, 0);
	float weighted_sum; 

	/*
	 *predicting first
	 */
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
		dout << "\nprediction: ";
		for(auto t: output)dout << t<<' ';
		dout << "\n";
	}

	/*
	 *backpropagation
	 */
	vector<float> delta_weights;
	float delta_weight, delta, hidden_delta;
	
	delta=(true_output-output[action])*Fdash(output[action], "Linear"); 

	if(verbose){
		dout << "predicted value: "<<output[action]<<' ';
		dout << "Error: "<<delta<<'\n';
	}
	
	this->bias_grad_sum[1][action]=this->bias_grad_sum[1][action]*0.9+0.1*delta*delta;
	this->bias[1][action]+=learning_rate*delta/pow(this->bias_grad_sum[1][action]+1e-10,0.5); //updating the bias
	if(verbose)dout << this->bias_grad_sum[1][action]<<'\n';
	//if(verbose){
		//dout << "bias: ";
		//for(auto b: this->bias[1])dout << b<<' ';
		//dout << '\n';
	//}
	for(int j=0;j<hidden_size;j++){
		delta_weight = delta*hidden_output[j]; 
		delta_weights.push_back(delta_weight);
	}

	for(int i=0;i<hidden_size;i++){
		float hidden_delta_sum=delta*this->weights[1][action*hidden_size+i];	
		hidden_delta=Fdash(hidden_output[i], "ReLU")*hidden_delta_sum;
		this->bias_grad_sum[0][i]=this->bias_grad_sum[0][i]*0.9+0.1*hidden_delta*hidden_delta;
		//if(i==hidden_size-2)cout << "hiddenDelta: "<<hidden_delta<<" sum: "<<this->bias_grad_sum[0][i]<<" value: "<<this->bias[0][i]<<'\n';
		this->bias[0][i]+=learning_rate*hidden_delta/pow(this->bias_grad_sum[0][i]+1e-10,0.5); //updating the bias
		
		for(int k=0;k<input_size;k++){
			delta_weight=hidden_delta*input[k];
			this->gradient_sum[0][i*input_size+k]=this->gradient_sum[0][i*input_size+k]*0.9+ 0.1*delta_weight*delta_weight;
			this->weights[0][i*input_size+k]+=learning_rate*delta_weight/pow(this->gradient_sum[0][i*input_size+k]+1e-10,0.5); //updating weights
		}
	}

	for(int j=0;j<hidden_size;j++){
		this->gradient_sum[1][action*hidden_size+j]=this->gradient_sum[1][action*hidden_size+j]*0.9 + 0.1*delta_weights[j]*delta_weights[j];
		this->weights[1][action*hidden_size+j]+=learning_rate*delta_weights[j]/pow(this->gradient_sum[1][action*hidden_size+j]+1e-10, 0.5); //updating weights
	}
	if(verbose){
		vector<float> out = this->predict(input);
		dout << "final_q_values: ";
		for(const float t: out)dout << t<< ' ';
		dout << '\n';
	}
		
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
	
	dout << "unidentified activation function!\n";
	return 0.0;
}

