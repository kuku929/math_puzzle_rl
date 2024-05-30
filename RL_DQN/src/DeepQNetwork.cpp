#include "NeuralNetwork.h"
#include "DeepQNetwork.h"
#include "rl_utils.h"
#include "globals.h"
#include <iostream>
#include <vector>
#include <deque>
#include <utility>
#include <cstdio> //size_t
#include <stdlib.h> //rand, srand
#include <time.h> //for time
#include <algorithm> //max_element

using namespace std;


void actor::act(state &some_state, int verbose){
	vector<float> input=normalize(some_state); //normalizing input between 0 to 1
	vector<float> q_values=this->actor_net.predict(input); //finding Q(s,a)
	if(verbose){
		dout << "q_values: ";
		for(auto t: q_values)dout << t<<' ';
		dout << "\n\n";
	}
	int action = EpsilonGreedy(q_values, EPSILON, 0); //choosing an action
	/*
	 *finding the reward	
	 */
	vector<int> action_space=possible_moves(some_state);
	float reward = 0.0f;
	if(find(action_space.begin(), action_space.end(), action)==action_space.end())
		reward=OUT_OF_BOUNDS;

	else{
		move(some_state, action); //changing states
		reward=R(some_state);
	}

	/*
	 *if final state, store many times in experience replay, idk if this will work
	 */

	//if(reward == COMPLETION_REWARD){
		//for(int i=0;i<100;i++){
			//this->experience_replay.push_back(state_action(some_state, action, reward)); //saving to experience log
			//if(this->experience_replay.size()>REPLAY_SIZE)
				//this->experience_replay.pop_front(); //maintaining the size 
		//}
	//}

	this->experience_replay.push_back(state_action(some_state, action ,reward)); //storing the resulting state after the action is taken
	if(this->experience_replay.size()>REPLAY_SIZE)
		this->experience_replay.pop_front(); //maintaining the size 
	return;
}

void actor::learn(int verbose){
	//vector<int> interesting_array({0, 1, 3, 2});
	
	//state INTERESTING_STATE({1,2,3,4,5,6,7,0,8});

	for(size_t i=0;i<BATCH_SIZE;i++){ 
		size_t rand_index = rand()%(this->experience_replay.size());//experience_replay.size();
		state_action random_state = experience_replay[rand_index];
		

		vector<int> action_space=possible_moves(random_state.first);
		float target_value;
		vector<float> input;
		vector<float> target_q_values;
		state next_state=random_state.first; //initializing next state, state stored in log is after the action 
		float reward=random_state.reward;

		/*
		 *finding target value based on the reward recieved
		 */
		if(reward==COMPLETION_REWARD){
			target_value=reward; //if final state stop 
			dout << "FINAL STATE\n";
			verbose = 1; //if final state show in log
		}
		else{  //find next state q values
			input=normalize(next_state);
			vector<float> target_q_action=this->target_net.predict(input); //using target net to find 'optimal' action
			target_q_values=this->actor_net.predict(input); //using the values of actor net in the target value i.e. the 'optimal' q value
			target_value=reward + target_q_values[max_element(target_q_action.begin(),target_q_action.end())-target_q_action.begin()]*GAMMA;	
		}
		/*
		 *debugging
		 */
		if(reward != OUT_OF_BOUNDS)
			move(next_state, (random_state.second + 2)%4); //reversing the move to get previous state
		//if(next_state.compressed_state == INTERESTING_STATE.compressed_state){ //checking if state is interesting
			//verbose = 1;
		//}
		if(verbose){ 
			dout << "---learning state---\n";
			dout << "STATE\n";
			print_state(next_state);
			dout << "ACTION : "<<random_state.second;
			if(reward == OUT_OF_BOUNDS)
				dout << " ( not allowed )";
			dout << "\nTARGET : "<<target_value<<'\n';
			dout << "	target values after action "<<random_state.second<<": "; for(auto t: target_q_values)dout << t<<' ';
			dout << '\n';
			dout << "	reward received: "<< reward<<'\n';
		}


		/*
		 *fitting
		 */
		input=normalize(next_state); //normalizing input
		this->actor_net.fit(input, target_value, INITIAL_LEARNING_RATE, GRAD_DECAY, random_state.second, verbose, OPTIMIZER); //learning using back propagation 

		if(verbose)
			dout << "---learnt state---\n\n";
		//this line is for testing the interesting states
		//verbose = 0;
		
		//this line is for testing the final state learning 
		if(reward==COMPLETION_REWARD){
			verbose = 0; 
		}
	}
	if(verbose)dout<< "------learnt all states--------\n\n";
}

void actor::update_target(){
	this->target_net=this->actor_net;
	return;
}

void actor::print_weights(){
	dout << "first layer:\n";
	int count=0;
	for(const auto t: this->actor_net.weights[0]){
		if(count%(INPUT_SIZE)<INPUT_SIZE-1)dout << t<<' ';
		if(count%INPUT_SIZE==INPUT_SIZE-1){
			dout<< t<<' ';
			dout << "bias: " << this->actor_net.bias[0][count/INPUT_SIZE]<<'\n'; 
		}
		count++;
	}
	dout << "\nsecond layer:\n";
	count=0;
	for(const auto t: this->actor_net.weights[1]){
		if(count%HIDDEN_SIZE<HIDDEN_SIZE-1)dout << t<<' ';
		if(count%HIDDEN_SIZE==HIDDEN_SIZE-1){
			dout<< t<<' ';
			dout << "bias: "<<this->actor_net.bias[1][count/HIDDEN_SIZE]<<'\n'; 
		}
		count++;
	}
	dout << "\n\n-------weights-end-------\n\n";
}

void actor::save_weights(string weights_filename){
	ofstream fout(weights_filename);
	int count=0;
	for(const auto t: this->actor_net.weights[0]){
		if(count%(INPUT_SIZE)<INPUT_SIZE-1)fout << t<<' ';
		if(count%INPUT_SIZE==INPUT_SIZE-1){
			fout<< t<<' ';
			fout << "bias: " << this->actor_net.bias[0][count/INPUT_SIZE]<<'\n'; 
		}
		count++;
	}
	fout << "second\n";
	count=0;
	for(const auto t: this->actor_net.weights[1]){
		if(count%HIDDEN_SIZE<HIDDEN_SIZE-1)fout << t<<' ';
		if(count%HIDDEN_SIZE==HIDDEN_SIZE-1){
			fout<< t<<' ';
			fout << "bias: "<<this->actor_net.bias[1][count/HIDDEN_SIZE]<<'\n'; 
		}
		count++;
	}
	fout.close();
}
void actor::print_target_weights(){
	dout << "first layer:\n";
	int count=0;
	for(const auto t: this->target_net.weights[0]){
		if(count%(INPUT_SIZE)<INPUT_SIZE-1)dout << t<<' ';
		if(count%INPUT_SIZE==INPUT_SIZE-1){
			dout << "bias: " << this->target_net.bias[0][count/INPUT_SIZE]; 
			dout<< t<<'\n';
		}
		count++;
	}
	dout << "\nsecond layer:\n";
	count=0;
	for(const auto t: this->target_net.weights[1]){
		if(count%HIDDEN_SIZE<HIDDEN_SIZE-1)dout << t<<' ';
		if(count%HIDDEN_SIZE==HIDDEN_SIZE-1){
			dout << "bias: "<<this->target_net.bias[1][count/HIDDEN_SIZE]; 
			dout<< t<<'\n';
		}
		count++;
	}
	dout << "\n\n-------weights-end-------\n\n";
}


int EpsilonGreedy(const vector<float> &q_values, float epsilon, int verbose){ //const vector<int> &action_space){
	//srand(2); //initializing seed	
	if(static_cast<float>(rand())/static_cast<float>(RAND_MAX) <= epsilon) //random action
		return rand()%4;
	if(verbose)dout << "\nmaking greedy action!\n";
	
	return static_cast<int>(max_element(q_values.begin(), q_values.end())-q_values.begin());
}

vector<float> normalize(const state &some_state){
	vector<float> normalized_output;
	for(int i=0;i<INPUT_SIZE;i++){
		int cell_value = some_state.compressed_state[i]-'A';
		normalized_output.push_back((float(cell_value)-static_cast<float>(INPUT_SIZE-1)/2.0f)/(static_cast<float>(INPUT_SIZE-1)/2.0f));	
	}
	return normalized_output;
}

float actor::R(state &next_state){
	if(isFinal(next_state))return COMPLETION_REWARD;
	return -5e-6f;
}

