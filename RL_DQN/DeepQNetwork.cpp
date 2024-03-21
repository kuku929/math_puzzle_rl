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
	//if(isFinal(some_state)){
		//cout << "\n\nREACHED!\n\n";
		//return some_state;
	//}
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
	for(int i=0;i<this->batch_size;i++){ 
		int rand_index = rand()%(this->experience_replay.size());//experience_replay.size();
		state_action random_state = experience_replay[rand_index];

		if(verbose)print_state(random_state.first);

		vector<int> action_space=possible_moves(random_state.first);
		float target_value;
		vector<float> input;
		vector<float> target_q_values;
		state next_state=random_state.first; //initializing next state as current state
		float reward=0;
		/*
		 *finding the reward
		 */
		if(find(action_space.begin(), action_space.end(), random_state.second)==action_space.end()){
			reward=out_of_bounds;
			if(verbose)cout << "action "<<random_state.second<<" not allowed!\n";
		}

		else{
			//if(action_space.size()<4)reward+=in_bound;
			next_state=move(random_state.first, random_state.second);
			reward+=R(random_state);
		}

		/*
		 *finding target value based on the reward recieved
		 */
		if(reward==completion_reward)target_value=reward; //if final state stop 
		else{                                            //find next state q values
			input=normalize(next_state);
			target_q_values =this->target_net.predict(input); 
			target_value=reward + *max_element(target_q_values.begin(),target_q_values.end())*gamma;	
		}
		/*
		 *debugging
		 */
		if(verbose){ cout << "target values for action "<<random_state.second<<": "; for(auto t: target_q_values)cout << t<<' ';
			cout << '\n';
			cout << "reward received: "<< reward<<'\n';
			cout << "target: "<<target_value<<'\n';
		}

		/*
		 *fitting
		 */
		input=normalize(random_state.first); //normalizing input
		this->actor_net.fit(input, target_value, this->learning_rate, random_state.second, verbose); //learning using back propagation 
	}
	if(verbose)cout<< "------learned--------\n\n";
}

void actor::end_learn(int verbose){
	if(verbose)cout << "learning the last states\n";
	int initial_size = this->experience_replay.size(); 
	if(initial_size==0){
		cout << "no memory!\n";
		return;
	}
	for(int i=0;i<initial_size;i++){
		this->learn(verbose);
		this->experience_replay.erase(this->experience_replay.begin());	
	}
	this->experience_replay.clear();
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
		if(count%(input_size)<input_size-1)cout << t<<' ';
		if(count%input_size==input_size-1){
			cout<< t<<' ';
			cout << "bias: " << this->actor_net.bias[0][count/input_size]<<'\n'; 
		}
		count++;
	}
	cout << "\nsecond layer:\n";
	count=0;
	for(const auto t: this->actor_net.weights[1]){
		if(count%hidden_size<hidden_size-1)cout << t<<' ';
		if(count%hidden_size==hidden_size-1){
			cout<< t<<' ';
			cout << "bias: "<<this->actor_net.bias[1][count/hidden_size]<<'\n'; 
		}
		count++;
	}
	cout << "\n\n-------weights-end-------\n\n";
}

void actor::save_weights(string weights_filename){
	ofstream fout(weights_filename);
	int count=0;
	for(const auto t: this->actor_net.weights[0]){
		if(count%(input_size)<input_size-1)fout << t<<' ';
		if(count%input_size==input_size-1){
			fout<< t<<' ';
			fout << "bias: " << this->actor_net.bias[0][count/input_size]<<'\n'; 
		}
		count++;
	}
	fout << "second\n";
	count=0;
	for(const auto t: this->actor_net.weights[1]){
		if(count%hidden_size<hidden_size-1)fout << t<<' ';
		if(count%hidden_size==hidden_size-1){
			fout<< t<<' ';
			fout << "bias: "<<this->actor_net.bias[1][count/hidden_size]<<'\n'; 
		}
		count++;
	}
	fout.close();
}
void actor::print_target_weights(){
	cout << "first layer:\n";
	int count=0;
	for(const auto t: this->target_net.weights[0]){
		if(count%(input_size)<input_size-1)cout << t<<' ';
		if(count%input_size==input_size-1){
			cout << "bias: " << this->target_net.bias[0][count/input_size]; 
			cout<< t<<'\n';
		}
		count++;
	}
	cout << "\nsecond layer:\n";
	count=0;
	for(const auto t: this->target_net.weights[1]){
		if(count%hidden_size<hidden_size-1)cout << t<<' ';
		if(count%hidden_size==hidden_size-1){
			cout << "bias: "<<this->target_net.bias[1][count/hidden_size]; 
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

float actor::R(const state_action &state_pair){
	if(isFinal(move(state_pair.first, state_pair.second)))return completion_reward;
	//if(replay_index>0){
		//int prev_move=this->experience_replay[replay_index-1].second;
		//if(abs(prev_move-state_pair.second)!=2)return 0.1; //avoid repitition of same moves
		//else return -0.1;
	//}
	return 0.0;
}

