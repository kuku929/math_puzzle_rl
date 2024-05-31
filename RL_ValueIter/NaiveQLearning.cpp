#include "NaiveQLearning.h"
#include "rl_utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <string>
#include <bitset>
#include <cmath>
using namespace std;
#define MAX_PRIME 524160 


int R(const state &curr_state, const state &next_state, const vector<string> &policy, int current_row){
	if(isFinal(next_state,current_row))return 100; 
	if(isFinal(curr_state,current_row))return -100;
	return 0;
}

void findMax(const state& curr_state, const vector<state>& connected_states, const vector<float>& value, const vector<string>& policy,int current_row, pair<string, float>& max_value){
	for(auto connected_state: connected_states){
		int index_connected_state=index_map[connected_state.compressed_state];
		float value_of_pair = (float)R(curr_state, connected_state,policy,current_row)+0.9*value[index_connected_state]; //finding Q(s,a)
		if(max_value.second<value_of_pair){ //using greedy to find V(s)
			max_value.second=value_of_pair;
			max_value.first=connected_state.compressed_state;
		}
	}
}

float difference(vector<float> &value_next, vector<float> &value_curr){
	float d=0;
	for(int i=0;i<(int)value_next.size();i++){
		d+=fabs(value_next[i]-value_curr[i]);
	}
	return d;
}

void train(unordered_map<string, vector<state>> &possible_states, vector<float>& value_curr, vector<float>& value_next, vector<string>& policy, float threshold, int save_file, int current_row){
	 int iter=0;
	 int index;
	 float delta=threshold+1;
	 int count=0;
	 pair<string, float> next_best_state;  

	 while(delta>threshold){
		 for(auto itr=possible_states.begin();itr!=possible_states.end();itr++){
			 index=index_map[itr->first];
			 next_best_state.first = string(16,'A');
			 next_best_state.second = -1000.0;
			 state current_state(itr->first);
			 findMax(current_state, itr->second, value_next, policy, current_row, next_best_state);

			 policy[index]=next_best_state.first;
			 value_next[index]=next_best_state.second;
		 }
		 delta = difference(value_next,value_curr);
		 cout << "\u0394: "<<delta << "\n"<<flush;
		 for(int i=0;i<(int)possible_states.size();i++)value_curr[i]=value_next[i];
	 }
	 if(save_file){
		string policy_filename;
		string value_filename;
		ofstream fout("temp_policy.txt", ios::app);
		ofstream ffout("temp_value.txt", ios::app);	
		for(int i=0;i<possible_states.size();i++){
			fout << policy[i]<<'\n';
			ffout<<value_curr[i]<<'\n';
		}
		fout << "done\n";
		fout.close();
		ffout.close();
	 }
	 return;
}

int play(int arr[16], vector<string>& policy,int current_row){
	int no_of_moves=0;
	int row_initial_state[16]={0};
	int initial_blank_position;
	if(current_row==2){
		for(int i=0;i<16;i++){
			if(arr[i]==0){
				row_initial_state[i]=1;
				initial_blank_position=i;
			}
			else if(8<arr[i]&&arr[i]<16)row_initial_state[i]=arr[i];
		}
	}
	else{
		for(int i=0;i<16;i++){
			if(arr[i]==0){
				row_initial_state[i]=15;
				initial_blank_position=i;
			}
			else if(4*current_row<arr[i]&&arr[i]<4*current_row+5)row_initial_state[i]=arr[i];
		}
	}

	state current_state(row_initial_state);
	int iter=0;
	int current_blank_position = initial_blank_position;
	if(isFinal(current_state,current_row))return 0;
	while(!isFinal(current_state,current_row)){
		if(iter++>1000){
			cout << "INFINITE LOOP!\n";
			return -1;
		}
		beautiful_print(arr);
		int current_index = index_map[current_state.compressed_state];
		int move_done = find_the_move(current_state, policy[current_index], current_row);
		move_puzzle(current_blank_position, move_done, arr);	
		current_blank_position+=move_done;
		current_state.compressed_state= policy[current_index];
		no_of_moves++;
	}
	
	return no_of_moves;
}
