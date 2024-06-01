#pragma once
#include "NeuralNetwork.h"
#include "rl_utils.h"
#include "globals.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <utility>
#include <stdlib.h> //rand, srand
#include <time.h> //for time
#include <algorithm> //max_element
using namespace std;
extern float out_of_bounds;
extern float in_bound;
extern float completion_reward;
extern ofstream dout;

struct state_action{
	state first;
	int second;
	float reward;
	state_action():first(), second(), reward(){};
	state_action(state f, int s, float r):first(f), second(s), reward(r){};
	state_action& operator=(const state_action&) = default;
};

struct actor{
	deque<state_action> experience_replay;	
	Network actor_net, target_net;

	actor():experience_replay(), actor_net(Network(LAYER_SIZES, ACTIVATION_FUNCTIONS)){
		this->target_net=actor_net;
	};

	actor(vector<vector<float>> &weights, vector<vector<float>> &bias):experience_replay(), actor_net(Network(LAYER_SIZES, ACTIVATION_FUNCTIONS,weights, bias)){
		this->target_net=actor_net;
	};

	actor& operator=(const actor&) = default;
	void act(state &some_state, int verbose); //does one move starting from some_state and logs it into experience
	void learn(int verbose); //samples from experience log and updates the weights
	void update_target(); //updates target network with current actor_net weights
	void print_weights(); //prints actor weights in the debug log
	void save_weights(string weights_filename); //saves the weights to a file
	void print_target_weights(); //prints target weights in the debug log
	float R(state &next_state); //reward function
};

int EpsilonGreedy(const vector<float> &q_values, float epsilon, int verbose);//const vector<int> &action_space);
vector<float> normalize(const state &some_state); //normalizes input to be fed into the neural network for a forward pass
