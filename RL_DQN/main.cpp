#include "DeepQNetwork.h"
#include "rl_utils.h"
#include "NeuralNetwork.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <unistd.h>
float out_of_bounds;
float in_bound;
float completion_reward;
int main(int argc, char **argv){
	/*
	 *flags for weights
	 */
	int c, flag=0;
	while((c=getopt(argc,argv,"w")) != -1){
		switch(c){
		case 'w':
			flag=1;
			break;
		}
	}
	/*
	 *defining params
	 */
	float epsilon=1.0;
	float initial_learning_rate=0.09;
	float min_factor=0.0001;
	float exp_factor=0.96;
	int batch_size=4;
	int replay_size=20;
	float gamma=0.9;
	int update_count=7;
	int steps_per_episode=10000;
	out_of_bounds=-0.5;
	//in_bound=-out_of_bounds/4.0;
	completion_reward=1.0;	
	int epoch=0;
	state curr_state;

	/*
	 *instantiating the actor
	 */
	vector<vector<float>> weights, bias;
	vector<float> sub_weights, sub_bias;
	actor a;
	if(flag){
		int curr_index=0;
		ifstream weights_in("weights.txt");
		string s;
		while(weights_in>>s){
			if(s=="second"){
				weights.push_back(sub_weights);
				bias.push_back(sub_bias);
				sub_weights.clear();
				sub_bias.clear();
				continue;
			}
			if(s=="bias:"){
				weights_in>>s;
				sub_bias.push_back(stof(s));
				continue;
			}
			sub_weights.push_back(stof(s));
		}
	weights.push_back(sub_weights);
	bias.push_back(sub_bias);
	weights_in.close();
	actor a_copy(epsilon, initial_learning_rate, batch_size, replay_size, gamma, weights, bias);
	a=a_copy; //kinda dumb but whatever
	}

	else{ 
		actor a_copy(epsilon, initial_learning_rate, batch_size, replay_size, gamma);
		a=a_copy;
	}

	/*
	 *loading the training set
	 */
	ifstream fin("../train.txt");	
	vector<string> training_set;
	string s;
	while(getline(fin,s)){
		training_set.push_back(s);
	}
	fin.close();
	
	int progress_count=0;	
	int update_after=training_set.size()/100;
	for(int l=0;l<3;l++){
		cout << "round "<<l<<'\n';
		a.learning_rate=initial_learning_rate;
		out_of_bounds=-0.5;
		completion_reward=1.0;
		for(auto data: training_set){
			//if(progress_count%update_after==0)cout << '\r'<<progress_count/update_after<<"%"<<flush;
			curr_state.compressed_state=data;
			curr_state.blank_position=curr_state.find_number(0);
				
			int count=0;
			int t=0; //time
			int verbose=0;

			out_of_bounds-=0.01;	
			//if(in_bound>0.001/4)in_bound-=0.001/4;
			//else in_bound=0;
			completion_reward+=0.2;

			cout << "out_of_bounds: "<<out_of_bounds<<'\n';
			cout << "in_bound: "<<in_bound<<'\n';
			cout << "completion_reward: "<<completion_reward<<'\n';
			
			a.epsilon=pow(1.0/(float)(1+epoch/20),0.5);	
			a.learning_rate=exp_factor*a.learning_rate;
			//if(a.learning_rate<initial_learning_rate*min_factor)
				//exp_factor=1/exp_factor;
			
			//if(a.learning_rate>=initial_learning_rate)
				//exp_factor=1/exp_factor;
				//out_of_bounds=-1;

			cout << "epsilon: "<< a.epsilon<<'\n';
			cout << "learning rate: "<<a.learning_rate<<'\n';
			cout << "start state:\n";
			print_state(curr_state);
			//if(a.learning_rate<0.005)a.learning_rate=0.005;
			a.print_weights();

			for(;t<batch_size;t++){
				//print_state(curr_state);
				if(isFinal(curr_state)){
					cout << "FINAL\n";
					break;
				}
				curr_state=a.act(curr_state, 0);
			}

			for(;t<steps_per_episode+batch_size;t++){
				count++;
				if(isFinal(curr_state)){
					cout << "FINAL\n";
					break;
					print_state(curr_state);
				}
				curr_state = a.act(curr_state, 0);
				a.learn(verbose);
				if(t>steps_per_episode-1 && verbose==0)verbose=1;	
				if(count==update_count){
					//cout << "weights updated!\n";
					count=0;
					a.update_target();		
					//a.print_target_weights();
				}
			}
			a.end_learn(1);
			cout << "\n\n-------epoch "<< epoch<< " over------\n\n";
			//cout << "size of exp: "<<a.experience_replay.size()<<'\n';
			//for(const auto &row: a.actor_net.weights) {
				//for (const auto cell : row) { std::cout << cell << ' ';
				//}
				//std::cout << "\n\n";
			//}
			epoch++;
			//a.print_weights();
			//if(epoch==1)break;
		}
	}
	a.save_weights("weights.txt");

	cout << "\n\n-----playing-----\n\n";
	curr_state.compressed_state="BCDEFGAHJKLINOPM";
	curr_state.blank_position=curr_state.find_number(0);
	a.epsilon=0.0;
	for(int i=0;i<1000;i++){
		print_state(curr_state);
		curr_state=a.act(curr_state, 1);
		if(isFinal(curr_state))break;
	}
}
