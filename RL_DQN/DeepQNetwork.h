#pragma once
#include "NeuralNetwork.h"
#include "rl_utils.h"
#include <iostream>
#include <vector>
#include <utility>
#include <stdlib.h> //rand, srand
#include <time.h> //for time
#include <algorithm> //max_element
using namespace std;
extern float out_of_bounds;
extern float in_bound;
extern float completion_reward;

struct state_action{
	state first;
	int second;
	state_action():first(), second(){};
	state_action(state first, int second):first(first), second(second){};
};

struct actor{
	int input_size=16;
	int hidden_size=8;
	int output_size=4;
	Network actor_net;	
	Network target_net;
	float epsilon;
	int batch_size;
	float learning_rate;
	int replay_size;
	float gamma;
	vector<state_action> experience_replay;	

	actor():epsilon(1), learning_rate(0.9), batch_size(4), replay_size(10), gamma(0.9), experience_replay(), actor_net(Network(input_size, hidden_size, output_size)){
		this->target_net=actor_net;
	};
	actor(float e, float l, int b, int r, float g):epsilon(e), learning_rate(l), batch_size(b), replay_size(r), gamma(g), experience_replay(), actor_net(Network(input_size, hidden_size, output_size)){
		this->target_net=actor_net;
	};
	actor(float e, float l, int b, int r, float g, vector<vector<float>> weights, vector<vector<float>> bias):epsilon(e), learning_rate(l), batch_size(b), replay_size(r), gamma(g), experience_replay(), actor_net(Network(input_size, hidden_size, output_size,weights, bias)){
		this->target_net=actor_net;
	};
	state act(const state &some_state, int verbose); //does one move starting from some_state and logs it into experience
	void learn(int verbose); //samples from experience log and updates the weights
	void end_learn(int verbose); //learning at the end of the episode
	void update_epsilon(const int t);
	void update_target(); //updates target network with current actor_net weights
	void print_weights();
	void save_weights(string weights_filename);
	void print_target_weights();
	float R(const state_action &state_pair);
};

int EpsilonGreedy(const vector<float> &q_values, float epsilon, int verbose);//const vector<int> &action_space);
vector<float> normalize(const state &some_state);
float R(const state_action &state_pair);



