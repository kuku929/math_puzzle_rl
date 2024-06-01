/*
 *@Author: Krutarth Patel                                           
 *@Date: 1st June 2024
 *@Description : implementation of Deep Q-Learning.
 *		definition of functions inside the actor struct
 */

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
	/*
	 * does one move using epsilon greedy policy
	 * pushes it to experience replay, which is a deque.
	 *
	 * (state_after_action, action_taken, resulting_reward ) --> experience replay
	 * if the state_after_action is the final state, it is stored many times in an effort to reduce sparsity of reward
	 * the size of the experience replay is also maintained by popping the front when required
	 */
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
	if(reward == COMPLETION_REWARD){
		for(int i=0;i<100;i++){
			this->experience_replay.push_back(state_action(some_state, action, reward)); //saving to experience log
			if(this->experience_replay.size()>REPLAY_SIZE)
				this->experience_replay.pop_front(); //maintaining the size 
		}
	}

	this->experience_replay.push_back(state_action(some_state, action ,reward)); //storing the resulting state after the action is taken
	if(this->experience_replay.size()>REPLAY_SIZE)
		this->experience_replay.pop_front(); //maintaining the size 
	return;
}

void actor::learn(int verbose){
	/*
	 * calls fit function provided in the network
	 * if verbose is set, prints stuff to the debug log
	 *
	 */
	for(size_t i=0;i<BATCH_SIZE;i++){ 
		size_t rand_index = rand()%(this->experience_replay.size());
		state_action random_state = experience_replay[rand_index];
		

		vector<int> action_space=possible_moves(random_state.first);
		vector<float> input;
		vector<float> target_q_values;
		state next_state=random_state.first; //initializing next state, state stored in log is after the action 
		float reward=random_state.reward;

		/*
		 *finding target value based on the reward recieved
		 */
		float target_value;
		if(reward==COMPLETION_REWARD){
			target_value=reward; //if final state stop 
		}
		
		//the ideal value now is -0.5
		else if(reward==OUT_OF_BOUNDS){
			target_value=reward;
		}
		else{  //find next state q values
			input=normalize(next_state);
			vector<float> target_q_action=this->target_net.predict(input); //using target net to find 'optimal' action
			target_q_values=this->actor_net.predict(input); //using the values of actor net in the target value i.e. the 'optimal' q value
			target_value=reward + target_q_values[max_element(target_q_action.begin(),target_q_action.end())-target_q_action.begin()]*GAMMA;	
		}

		if(reward != OUT_OF_BOUNDS)
			move(next_state, (random_state.second + 2)%4); //reversing the move to get previous state
								       
		/*
		 *debugging
		 */
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
		vector<float> target(LAYER_SIZES.back(), 0.0f);
		target[random_state.second] = target_value;
		this->actor_net.fit(input, target, INITIAL_LEARNING_RATE, GRAD_DECAY, verbose, OPTIMIZER); //learning using back propagation 

		if(verbose){
			vector<float> final_output = this->actor_net.predict(input);
			dout << "after : ";
			for(const auto output : final_output)
				dout << output << ' ';
			dout << '\n';
			dout << "---learnt state---\n\n";
		}
	}
	if(verbose)dout<< "------learnt all states--------\n\n";
}

void actor::update_target(){
	/*
	 *update the target weights to the current actor weights
	 */
	this->target_net=this->actor_net;
	return;
}

void actor::print_weights(){
	size_t layer_size;
	for(size_t current_layer = 0;current_layer < LAYER_SIZES.size() - 1;++current_layer){
		layer_size = LAYER_SIZES[current_layer];
		dout << "layer " << current_layer << '\n';
		for(size_t i=0; i < this->actor_net.weights[current_layer].size();++i){
			if(i%layer_size == layer_size - 1){
				dout << this->actor_net.weights[current_layer][i] << ' ';
				dout << "bias: " << this->actor_net.bias[current_layer][i/layer_size] << '\n';
			}

			else{
				dout << this->actor_net.weights[current_layer][i] << ' ';
			}
		}
		dout << '\n';
	}
	dout << "\n\n-------weights-end-------\n\n";
}

void actor::save_weights(string weights_filename){
	ofstream fout(weights_filename);
	size_t layer_size;
	for(size_t current_layer = 0;current_layer < LAYER_SIZES.size() - 1;++current_layer){
		layer_size = LAYER_SIZES[current_layer];
		for(size_t i=0; i < this->actor_net.weights[current_layer].size();++i){
			if(i%layer_size == layer_size - 1){
				fout << this->actor_net.weights[current_layer][i] << ' ';
				fout << "bias: " << this->actor_net.bias[current_layer][i/layer_size] << '\n';
			}

			else{
				fout << this->actor_net.weights[current_layer][i] << ' ';
			}
		}
		fout << "layer\n";
	}
	fout.close();
}
void actor::print_target_weights(){
	size_t layer_size;
	for(size_t current_layer = 0;current_layer < LAYER_SIZES.size() - 1;++current_layer){
		layer_size = LAYER_SIZES[current_layer];
		dout << "layer " << current_layer << '\n';
		for(size_t i=0; i < this->target_net.weights[current_layer].size();++i){
			if(i%layer_size == layer_size - 1){
				dout << this->target_net.weights[current_layer][i] << ' ';
				dout << "bias: " << this->target_net.bias[current_layer][i/layer_size] << '\n';
			}

			else{
				dout << this->target_net.weights[current_layer][i] << ' ';
			}
		}
		dout << '\n';
	}
	dout << "\n\n-------weights-end-------\n\n";
}


int EpsilonGreedy(const vector<float> &q_values, float epsilon, int verbose){ //const vector<int> &action_space){
	if(static_cast<float>(rand())/static_cast<float>(RAND_MAX) <= epsilon) //random action
		return rand()%4;
	//debug
	if(verbose)dout << "\nmaking greedy action!\n";
	
	return static_cast<int>(max_element(q_values.begin(), q_values.end())-q_values.begin());
}

vector<float> normalize(const state &some_state){
	vector<float> normalized_output;

	for(const auto cell: some_state.compressed_state){
		normalized_output.push_back((static_cast<float>(cell-'A') - 7.5f)/7.5f);
	}
	return normalized_output;
}

float actor::R(state &next_state){
	if(isFinal(next_state))return COMPLETION_REWARD;
	return 0.0f;
}

