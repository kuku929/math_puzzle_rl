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

struct state_action{
	state first;
	int second;
	state_action():first(), second(){};
	state_action(state first, int second):first(first), second(second){};
};

struct actor{
	Network actor_net{Network(16,8,4)};	
	Network target_net=actor_net;
	float epsilon;
	int batch_size;
	float learning_rate;
	int replay_size;
	float gamma;
	vector<state_action> experience_replay;	

	actor():epsilon(1), learning_rate(0.9), batch_size(4), replay_size(10), gamma(0.9), experience_replay(){};
	actor(float e, float l, int b, int r, float g):epsilon(e), learning_rate(l), batch_size(b), replay_size(r), gamma(g), experience_replay(){};
	state act(const state &some_state, int verbose); //does one move starting from some_state and logs it into experience
	void learn(int verbose); //samples from experience log and updates the weights
	void update_epsilon(const int t);
	void update_target(); //updates target network with current actor_net weights
	void print_weights();
};

int EpsilonGreedy(const vector<float> &q_values, float epsilon, int verbose);//const vector<int> &action_space);
vector<float> normalize(const state &some_state);
float R(const state_action &state_pair);



