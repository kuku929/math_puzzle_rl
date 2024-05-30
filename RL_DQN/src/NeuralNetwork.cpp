#include "NeuralNetwork.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath> //pow
#include <chrono>
#include <iomanip>
using namespace std;
extern ofstream dout;

Network::Network(int in, int h, int o):input_size(in), hidden_size(h), output_size(o){
	float range = 2.0;//4*pow(6.0/(float)(input_size+output_size), 0.5); 
	//srand(chrono::high_resolution_clock::now().time_since_epoch().count());
	for(int i=0;i<input_size*hidden_size;i++){
		float normalized_value = static_cast<float>(rand())/static_cast<float>(RAND_MAX)-0.5f; //between (-0.5,0.5)	
		this->weights[0].push_back(normalized_value*range); //between (-range/2, range/2)
		this->gradient_sum[0].push_back(0.0f); //between (-range/2, range/2) 

	}
	
	for(int i=0;i<output_size*hidden_size;i++){
		float normalized_value = static_cast<float>(rand())/static_cast<float>(RAND_MAX)-0.5f; //between (-0.5,0.5)	
		this->weights[1].push_back(normalized_value*range); //between (-range/2, range/2)
		this->gradient_sum[1].push_back(0.0f); //between (-range/2, range/2)
	}

	for(int i=0;i<hidden_size;i++){
		this->bias[0].push_back(0.0f);
		this->bias_grad_sum[0].push_back(0.0f);	
	}

	for(int i=0;i<output_size;i++){
		this->bias[1].push_back(0.0f);
		this->bias_grad_sum[1].push_back(0.0f);	
	}

}

Network::Network(int in, int h, int o, vector<vector<float>> &w, vector<vector<float>> &b):input_size(in), hidden_size(h), output_size(o){
	this->weights=w;
	this->bias=b;
	for(int i=0;i<input_size*hidden_size;i++){
		this->gradient_sum[0].push_back(0.0f); //between (-range/2, range/2)
	}

	for(int i=0;i<output_size*hidden_size;i++){
		this->gradient_sum[1].push_back(0.0f); //between (-range/2, range/2)
	}

	for(int i=0;i<hidden_size;i++)this->bias_grad_sum[0].push_back(0.0f);
	for(int i=0;i<output_size;i++)this->bias_grad_sum[1].push_back(0.0f);
}

vector<float> Network::predict(const vector<float> &input){
	vector<float> hidden_output(hidden_size, 0.0f);
	vector<float> output(output_size, 0.0f);
	float weighted_sum;
	//if(input.size()!=input_size){
		//dout << "Mismatch of input dimensions!\n";
		//return output;
	//}
	for(int i=0;i<hidden_size;i++){
		weighted_sum = bias[0][i];
		for(int j=0;j<input_size;j++){
			weighted_sum += weights[0][i*input_size+j]*input[j];
		}
		
		//CHANGING ReLU to sigmoid - 29 may
		hidden_output[i]=sigmoid(weighted_sum);	
		//cout << "hidden output " << i << ": " << hidden_output[i]<<endl; 
	}
	for(int i=0;i<output_size;i++){
		weighted_sum = bias[1][i];
		for(int j=0;j<hidden_size;j++)
			weighted_sum += weights[1][i*hidden_size+j]*hidden_output[j];
		
		output[i]=weighted_sum;
	}

	return output;
}

void Network::fit(const vector<float> &input, const float true_output, const float learning_rate,const float grad_decay, const int action, int verbose, int optimize){
	/*
	 *sanity check
	 */
	//if(input.size()!=input_size){
		//dout << "Mismatch of input dimensions!\n";
		//return;
	//}
	vector<float> hidden_output(hidden_size, 0);
	vector<float> hidden_weighted_sum(hidden_size,0);
	vector<float> output(output_size, 0);
	vector<float> output_weighted_sum(output_size,0);
	float weighted_sum; 

	/*
	 *predicting first
	 */
	for(int i=0;i<hidden_size;i++){
		weighted_sum = bias[0][i]; //sk
		for(int j=0;j<input_size;j++)
			weighted_sum += weights[0][i*input_size+j]*input[j];
		//CHANGING ReLU to sigmoid - 29 may
		hidden_output[i]=sigmoid(weighted_sum); //yk	
	}

	for(int i=0;i<output_size;i++){
		weighted_sum = bias[1][i]; //s0 
		for(int j=0;j<hidden_size;j++)
			weighted_sum += weights[1][i*hidden_size+j]*hidden_output[j];
		output[i]=weighted_sum;
	}
	if(verbose){
		dout << "PREDICTION\n	initial : ";
		for(auto t: output)dout << t<<' ';
		dout << "\n";
	}
	

	/*
	 *backpropagation
	 */
	vector<float> delta_weights;
	float delta_weight, delta, hidden_delta;
	
	delta=(true_output-output[action]); //for gradient descent on output bias and weights
									    //bias += delta*learning-rate*Fdash
	if(verbose){
		dout <<"	error: "<<delta<<'\n';
	}

	if(optimize){
		float grad_sum_term=delta*Fdash(output[action], "Linear"); //Fdash takes in F(s) i.e. the output	
		this->bias_grad_sum[1][action]=this->bias_grad_sum[1][action]*grad_decay+(1-grad_decay)*grad_sum_term*grad_sum_term; //adding to delta sum
		this->bias[1][action]+=learning_rate*grad_sum_term/(static_cast<float>(pow(this->bias_grad_sum[1][action],0.5))+1e-10f); //updating the bias
	}
	else{
		this->bias[1][action]+=learning_rate*delta*Fdash(output[action], "Linear");
	}

	if(verbose){
		dout << "BIAS GRAD SUM\n";
		for(const float t: bias_grad_sum[1])
			dout << t << ' ';
		dout << '\n';
	}

	for(int i=0;i<hidden_size;i++){
		//output layer
		delta_weight = delta*hidden_output[i]*Fdash(output[action], "Linear"); 
		delta_weights.push_back(delta_weight); //amount to change the output weights
						       
		//hidden layer
		float hidden_delta_sum=delta*this->weights[1][action*hidden_size+i]; //propagation of delta backwards	
		
		//changing ReLU to sigmoid - 29 may
		hidden_delta=Fdash(hidden_output[i], "sigmoid")*hidden_delta_sum; //for gradient descent on hidden bias and weights

		if(verbose)
			dout << "HIDDEN DELTA : " << hidden_delta << '\n';

		if(optimize){
			this->bias_grad_sum[0][i]=this->bias_grad_sum[0][i]*grad_decay+(1-grad_decay)*hidden_delta*hidden_delta; //adding to bias delta sum
			this->bias[0][i]+=learning_rate*hidden_delta/(static_cast<float>(pow(this->bias_grad_sum[0][i],0.5))+1e-10f); //updating the bias
		}
		else{
			this->bias[0][i]+=learning_rate*hidden_delta; //updating the bias
		}
		
		for(int k=0;k<input_size;k++){
			//changing the hidden weights first
			delta_weight=hidden_delta*input[k]; //amount to change the hidden weights
			if(optimize){
				this->gradient_sum[0][i*input_size+k]=this->gradient_sum[0][i*input_size+k]*grad_decay+ (1-grad_decay)*delta_weight*delta_weight; //adding to weight delta sum
				this->weights[0][i*input_size+k]+=learning_rate*delta_weight/(static_cast<float>(pow(this->gradient_sum[0][i*input_size+k],0.5))+1e-10f); //updating weights
			}
			else{
				this->weights[0][i*input_size+k]+=learning_rate*delta_weight; //updating weights
			}
		}
	}
	//cout << '\n';
	//if(verbose)dout<<'\n';

	for(int j=0;j<hidden_size;j++){
		if(optimize){
			this->gradient_sum[1][action*hidden_size+j]=this->gradient_sum[1][action*hidden_size+j]*grad_decay + (1-grad_decay)*delta_weights[j]*delta_weights[j];
			this->weights[1][action*hidden_size+j]+=learning_rate*delta_weights[j]/(static_cast<float>(pow(this->gradient_sum[1][action*hidden_size+j], 0.5))+1e-10f); //updating weights
		}
		else{
			this->weights[1][action*hidden_size+j]+=learning_rate*delta_weights[j]; //updating weights
		}
	}

	
	if(verbose){
		vector<float> out = this->predict(input);
		dout << "	final : ";
		for(const float t: out)dout << t<< ' ';
		dout << '\n';
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

//void beautiful_print(const vector<float> &input, vector<float> &hidden_weighted_sum, vector<float> &hidden_output, vector<float> &output_weighted_sum, vector<float> &output,vector<vector<float>> weights, vector<vector<float>> bias){
	//cout << setprecision(1)<< input[0] << " ---"<<weights[0][0]<<'|'<<weights[0][1]<<"--> "<<bias[0][0]<<'/'<<hidden_output[0];
	//cout << "\n                     "<<weights[1][0];
	//cout << "\n                       |--->"<<bias[1][0]<<'/'<<output[0]<<'\n';
	//cout << "                     "<<weights[1][1]<<'\n';
	//cout << input[1] << " ---"<<weights[0][2]<<'|'<<weights[0][3]<<"--> "<<bias[0][1]<<'/'<<hidden_output[1]<<'\n';
	//cout << setprecision(6);
//}
