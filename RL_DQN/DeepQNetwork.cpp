#include "NeuralNetwork.h"
#include "DeepQNetwork.h"
#include "rl_utils.h"
#include <iostream>
#include <vector>
#include <utility>
#include <stdlib.h> //rand, srand
#include <time.h> //for time
#include <algorithm> //max_element
using namespace std;


state actor::act(const state &some_state, int verbose){
	//vector<int> action_space=possible_moves(some_state); //possible actions	
	vector<float> input=normalize(some_state); //normalizing input between 0 to 1
	vector<float> q_values=this->actor_net.predict(input); //finding Q(s,a)
	if(verbose){
		cout << "q_values: ";
		for(auto t: q_values)cout << t<<' ';
		cout << "\n\n";
	}
	
	int action = EpsilonGreedy(q_values, this->epsilon, verbose); //choosing an action
	this->experience_replay.push_back(state_action(some_state, action)); //saving to experience log
	if(this->experience_replay.size()>this->replay_size)this->experience_replay.erase(this->experience_replay.begin()); //maintaining the size 
	return move(some_state, action);
}

void actor::learn(int verbose){
	if(this->batch_size>this->replay_size)cout << "batch size more than replay memory!\n";
	for(int i=0;i<this->batch_size;i++){ //selecting a batch of size batch_size 
		int rand_index = rand()%(this->experience_replay.size());//experience_replay.size();
		state_action random_state = experience_replay[rand_index];

		if(verbose)print_state(random_state.first);

		vector<int> action_space=possible_moves(random_state.first);
		float target_value;
		vector<float> input;
		state next_state=random_state.first;
		float reward;
		if(find(action_space.begin(), action_space.end(), random_state.second)==action_space.end()){
			reward=out_of_bounds;
			if(verbose)cout << "action "<<random_state.second<<"not allowed!\n";
		}

		else{
			next_state=move(random_state.first, random_state.second);
			reward=R(random_state);
		}
		input=normalize(next_state);
		vector<float> target_q_values =this->target_net.predict(input); 
		if(verbose){
			cout << "target values for action "<<random_state.second<<": ";
			for(auto t: target_q_values)cout << t<<' ';
			cout << '\n';
		}

		target_value=reward + *max_element(target_q_values.begin(),target_q_values.end())*gamma;	
		if(verbose)cout << "target: "<<target_value<<'\n';

		input=normalize(random_state.first); //normalizing input
		this->actor_net.fit(input, target_value, this->learning_rate, random_state.second, verbose); //learning using back propagation 
	}
	if(verbose)cout << "------learned--------\n\n";
}

void actor::update_epsilon(int t){
	this->epsilon=pow(1.0/(float)(1+t),0.5);
	return;
}

void actor::update_target(){
	this->target_net=this->actor_net;
	return;
}

void actor::print_weights(){
	cout << "first layer:\n";
	int count=0;
	for(const auto t: this->actor_net.weights[0]){
		if(count%16<15)cout << t<<' ';
		if(count%16==15){
			cout << "bias: " << this->actor_net.bias[0][count/16]; 
			cout<< t<<'\n';
		}
		count++;
	}
	cout << "\nsecond layer:\n";
	count=0;
	for(const auto t: this->actor_net.weights[1]){
		if(count%8<7)cout << t<<' ';
		if(count%8==7){
			cout << "bias: "<<this->actor_net.bias[1][count/32]; 
			cout<< t<<'\n';
		}
		count++;
	}
	cout << "\n\n-------weights-end-------\n\n";
}

int EpsilonGreedy(const vector<float> &q_values, float epsilon, int verbose){ //const vector<int> &action_space){
	//srand(2); //initializing seed	
	if(rand()/(float)RAND_MAX <= epsilon) //random action
		return rand()%4;
	if(verbose)cout << "\nmaking greedy action!\n";
	
	return max_element(q_values.begin(), q_values.end())-q_values.begin();
	//vector<float> allowed_q_values(action_space.size());
	//for (int i = 0; i < action_space.size(); i++) {
		//allowed_q_values[i] = q_values[action_space[i]];
	//}
	
	//return action_space[max_element(allowed_q_values.begin(), allowed_q_values.end()) - allowed_q_values.begin()]; //greedy action
}

vector<float> normalize(const state &some_state){
	vector<float> normalized_output;
	for(int i=0;i<16;i++){
		int cell_value = some_state.compressed_state[i]-'A';
		normalized_output.push_back((cell_value-7.5)/(7.5));	
	}
	return normalized_output;
}

float R(const state_action &state_pair){
	if(isFinal(state_pair.first))return -1;
	if(isFinal(move(state_pair.first)))return 1;
	return 0.0;
	//vector<int> action_space = possible_moves(state_pair.first);
	//int curr_blank_position=state_pair.first.blank_position;
	//int next_blank_position;

	//if(find(action_space.begin(), action_space.end(), state_pair.second)==action_space.end())return out_of_bounds;
	//if(isFinal(state_pair.first))return 100;
	//switch(state_pair.second){
		//case 0:
			//next_blank_position=curr_blank_position-4;
			//break;
		//case 1:
			//next_blank_position=curr_blank_position+1;
			//break;
		//case 2:
			//next_blank_position=curr_blank_position+4;
			//break;
		//case 3:
			//next_blank_position=curr_blank_position-1;
			//break;
	//}
	//int cell_value = state_pair.first.compressed_state[next_blank_position]-'A';
	//float reward = abs(curr_blank_position/4-cell_value/4)+abs(curr_blank_position%4-cell_value%4);
	//reward -= abs(next_blank_position/4-cell_value/4)+abs(next_blank_position%4-cell_value%4);

	//return 10*reward;
	
}

