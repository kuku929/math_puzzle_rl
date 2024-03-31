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
std::ofstream dout("debug.txt");
int main(int argc, char **argv){
	/*
	 *flags for weights
	 */
	int c, flag=0, train_model=0;
	while((c=getopt(argc,argv,"wt")) != -1){
		switch(c){
		case 'w':
			flag=1;
			break;
		
		case 't':
			train_model=1;
			break;
		}
	}

	/*
	 *defining params
	 */
	float epsilon=1.0;
	float initial_learning_rate=0.0009;
	int batch_size=30;
	int replay_size=1000000;
	float gamma=0.9;
	int update_count=50;
	int steps_per_episode=100000;
	out_of_bounds=-0.5;
	completion_reward=2.0;	
	int epoch=0;
	state curr_state;
	//float exp_factor=0.98;
	//in_bound=-out_of_bounds/4.0;

	/*
	 *instantiating the actor
	 */
	vector<vector<float>> weights, bias;
	vector<float> sub_weights, sub_bias; //weights/bias of the 2 layers
	actor a;
	if(flag){
		int curr_index=0;
		ifstream weights_in("weights.txt");
		string s; while(weights_in>>s){
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
	 *training
	 */
	if(train_model){
		ifstream fin("../data/train.txt");	
		vector<string> training_set;
		string s;
		while(getline(fin,s)){
			training_set.push_back(s);
		}
		fin.close();
		
		//a.print_weights();
		int progress_count=0;	
		int total_epochs=training_set.size();
		int count=0;

		for(auto data: training_set){
			int percentage=progress_count*100/training_set.size();
			std::cout<<percentage<<"%\r";
			std::cout.flush();
			progress_count++;

			curr_state.compressed_state=data;
			curr_state.blank_position=curr_state.find_number(0);
				
			int t=0; //time
			int verbose=0;

			//out_of_bounds-=0.01;	
			//if(in_bound>0.001/4)in_bound-=0.001/4;
			//else in_bound=0;
			//completion_reward+=0.2;

			//dout << "out_of_bounds: "<<out_of_bounds<<'\n';
			//dout << "in_bound: "<<in_bound<<'\n';
			//dout << "completion_reward: "<<completion_reward<<'\n';

			//if(a.learning_rate<1e-7)a.learning_rate=1e-7;
			//else a.learning_rate=exp_factor*a.learning_rate;
			//if(a.learning_rate<initial_learning_rate*min_factor)
				//exp_factor=1/exp_factor;
			
			//if(a.learning_rate>=initial_learning_rate)
				//exp_factor=1/exp_factor;
				//out_of_bounds=-1;

			dout << "epsilon: "<< a.epsilon<<'\n';
			dout << "learning rate: "<<a.learning_rate<<'\n';
			dout << "start state:\n";
			print_state(curr_state);

			//for(;t<batch_size;t++){
				////print_state(curr_state);
				//if(isFinal(curr_state)){
					//dout << "FINAL\n";
					//break;
				//}
				//a.act(curr_state, 0);
				
			//}
			//std::cout << "check1\n"<<std::flush;
			for(;t<steps_per_episode;t++){
				if(a.epsilon>0.1)a.epsilon-=2*1e-5*(a.epsilon*a.epsilon);//pow(1/(1 + (epoch*steps_per_episode+(float)t)/10000.0),0.5);
				if(isFinal(curr_state)){
					dout << "FINAL\n";
					break;
				}
				a.act(curr_state, 0);
				if(t%5==0)a.learn(verbose);
				
				count++;
				//if(t>steps_per_episode-5 && verbose==0)verbose=1;	
				if(count==update_count){
					//dout << "weights updated!\n";
					count=0;
					a.update_target();		
					//a.print_target_weights();
				}
				//if(t>steps_per_episode-5||percentage==46)std::cout << t<<"\n"<<std::flush;

			}
			//std::cout << "reached!\n"<<std::flush;
			//a.end_learn(verbose);
			dout << "\n\n-------epoch "<< epoch<< " over------\n\n";
			epoch++;
			a.print_weights();
			a.save_weights("weights.txt");

		}
		//a.end_learn(0);
		//}
	}

	dout << "\n\n-----playing-----\n\n";
	ifstream input("../data/input.txt");
	string start_state="";
	while(input>>c){
		start_state+='A'+c;	
	}
	input.close();
	curr_state.compressed_state=start_state;//"BCDEFGAHJKLINOPM";
	curr_state.blank_position=curr_state.find_number(0);
	a.epsilon=0.0;
	for(int i=0;i<10;i++){
		print_state(curr_state);
		a.act(curr_state, 1);
		if(isFinal(curr_state)){
			print_state(curr_state);
			break;
		}
	}
	dout.close();
}
