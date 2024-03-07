#include "DeepQNetwork.h"
#include "rl_utils.h"
#include "NeuralNetwork.h"
#include <iostream>
#include <fstream>
#include <cmath>
float out_of_bounds;

int main(){
	//defining params
	float epsilon=1.0;
	float initial_learning_rate=0.09;
	int batch_size=4;
	int replay_size=20;
	float gamma=0.9;
	int count;
	int update_count=7;
	int steps_per_episode=1000;
	out_of_bounds=-0.5;
	int epoch=0;
	state curr_state;

	//instantiating the actor
	actor a(epsilon, initial_learning_rate, batch_size, replay_size, gamma);

	//loading the training set
	ifstream fin("../train.txt");
	vector<string> training_set;
	string s;
	while(getline(fin,s)){
		training_set.push_back(s);
	}
	
	//training on data
	for(auto data: training_set){
		curr_state.compressed_state=data;
		curr_state.blank_position=curr_state.find_number(0);
		
		//print_state(curr_state);
		count=0;
		int t=0; //time
		int verbose=0;
		out_of_bounds-=0.05;	
		//cout << "initial moves done\n";

		a.epsilon=pow(1.0/(float)(1+epoch),0.5);	
		a.learning_rate=pow(0.99, epoch)*initial_learning_rate;
		if(a.learning_rate<0.005)a.learning_rate=0.005;
		a.print_weights();

		for(;t<batch_size;t++){
			//print_state(curr_state);
			curr_state=a.act(curr_state, 0);
		}

		for(;t<steps_per_episode+batch_size;t++){
			count++;
			curr_state = a.act(curr_state, 0);
			a.learn(verbose);
			if(verbose==0 && t>steps_per_episode-5)verbose=1;	
			if(count==update_count){
				//cout << "weights updated!\n";
				count=0;
				a.update_target();		
			}
		}
		cout << "\n\n-------epoch "<< epoch<< " over------\n\n";
		//for(const auto &row: a.actor_net.weights) {
			//for (const auto cell : row) {
				//std::cout << cell << ' ';
			//}
			//std::cout << "\n\n";
		//}
		epoch++;
		//a.print_weights();
		//if(epoch==5)break;
	}
	cout << "\n\n-----playing-----\n\n";
	curr_state.compressed_state="GOKHCBFEINPJLAMD";
	curr_state.blank_position=curr_state.find_number(0);
	for(int i=0;i<1000;i++){
		print_state(curr_state);
		curr_state=a.act(curr_state, 1);
		if(isFinal(curr_state))break;
	}
}
