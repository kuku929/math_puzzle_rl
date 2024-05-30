#include "DeepQNetwork.h"
#include "rl_utils.h"
#include "NeuralNetwork.h"
#include "globals.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <unistd.h>
#include <chrono>
#include <cstdio>
float out_of_bounds;
float in_bound;
float completion_reward;
using namespace std;
using Clock = std::chrono::steady_clock;
using Second = std::chrono::duration<double, std::ratio<1> >;
std::ofstream dout("debug.txt");
//std::ofstream plt("plot.txt");
int main(int argc, char **argv){
	std::chrono::time_point<Clock> m_beg = Clock::now();
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
	 *instantiating the actor
	 */
	vector<vector<float>> weights, bias;
	vector<float> sub_weights, sub_bias; //weights/bias of the 2 layers
	actor a;
	if(flag){
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
	actor a_copy(weights, bias);
	a=a_copy; //kinda dumb but whatever
	}

	state curr_state;
	/*
	 *training
	 */
	int total_loop_time = 0;
	if(train_model){
		ifstream fin("../data/train.txt");	
		vector<string> training_set;
		string s;
		while(getline(fin,s)){
			training_set.push_back(s);
		}
		fin.close();
		
		//a.print_weights();
		size_t progress_count=0;	
		int count=0;
		
		float EPSILON_DECAY_FACTOR = 1.0f/static_cast<float>(training_set.size());
		//cout <<"decay factor: "<< EPSILON_DECAY_FACTOR<<'\n';
		for(auto data: training_set){
			dout << "-------epoch "<< EPOCH<< " starting------\n";
			//printing progress
			size_t percentage=progress_count*100/training_set.size();
			std::cout<<percentage<<"%\r";
			std::cout.flush();
			progress_count++;

			curr_state.compressed_state=data;
			curr_state.blank_position=curr_state.find_number(0);
				
			int t=0; //time
			int verbose=0;

			dout << "EPSILON : "<< EPSILON<<'\n';
			dout << "START STATE\n";
			print_state(curr_state);

			for(;t<STEPS_PER_EPISODE;t++){
				if(t>STEPS_PER_EPISODE-6 && verbose==0)verbose=1;	
				if(isFinal(curr_state)){
					dout << "final state reached, aborting\n";
					break;
				}
				a.act(curr_state, 0);
				if(t%5==0)a.learn(verbose);
				
				count++;
				if(count==UPDATE_COUNT){
					count=0;
					a.update_target();		
				}

			}
			total_loop_time += t;
			if(EPSILON>0.1)EPSILON-=EPSILON_DECAY_FACTOR;//6*1e-3f*(EPSILON*EPSILON);//pow(1/(1 + (epoch*steps_per_episode+(float)t)/10000.0),0.5);
			dout << "-------epoch "<< EPOCH<< " over------\n\n";
			EPOCH++;
			a.save_weights("weights.txt");

		}
	}
	std::cout << "\ntime taken: "<<std::chrono::duration_cast<Second>(Clock::now() - m_beg).count()<<'\n'; 
	std::cout << "no of iterations: " << total_loop_time << '\n';
	dout << "\n\n-----playing-----\n\n";
	ifstream input("../data/input.txt");
	string start_state="";
	while(input>>c){
		start_state+=static_cast<char>(static_cast<int>('A')+c);	
	}
	input.close();
	curr_state.compressed_state=start_state;//"BCDEFGAHJKLINOPM";
	curr_state.blank_position=curr_state.find_number(0);
	EPSILON=0.0;
	for(int i=0;i<10;i++){
		print_state(curr_state);
		a.act(curr_state, 1);
		if(isFinal(curr_state)){
			print_state(curr_state);
			break;
		}
	}
	dout.close();
	//plt.close();
}
