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
	int next_state_index=index_map[next_state.compressed_state];
	if(policy[next_state_index]==curr_state.compressed_state)return -100;


	int reward=0;
	if(current_row == 2){
		int curr_blank_position=curr_state.find_number(1);
		int next_blank_position=next_state.find_number(1);
		int tile_swapped = curr_state.compressed_state[next_blank_position]-'A';	
		int row=(tile_swapped-1)/4;
		int col=(tile_swapped-1)%4;

		reward += abs(row-next_blank_position/4)+abs(col-next_blank_position%4);
		reward -= abs(row-curr_blank_position/4)+abs(col-curr_blank_position%4);
	}
	else{
		int curr_blank_position=curr_state.find_number(15);
		int next_blank_position=next_state.find_number(15);
		int tile_swapped = curr_state.compressed_state[next_blank_position]-'A';	
		if(tile_swapped!=0){
			int row=(tile_swapped-1)/4;
			int col=(tile_swapped-1)%4;

			reward += abs(row-next_blank_position/4)+abs(col-next_blank_position%4);
			reward -= abs(row-curr_blank_position/4)+abs(col-curr_blank_position%4);
		}
		else{
			vector<int> not_in_place;
			for(int i=4*current_row;i<4*current_row+4;i++){
				if(curr_state.compressed_state[i]!=i+1+'A')not_in_place.push_back(i+1);
			}
			for(auto t: not_in_place){
				int position=curr_state.find_number(t);
				int row=(position)/4;
				int col=position%4;

				reward -= abs(row-next_blank_position/4)+abs(col-next_blank_position%4);
				reward += abs(row-curr_blank_position/4)+abs(col-curr_blank_position%4);
			}
			//int min_curr_distance=10;
			//int min_next_distance=10;
			//for(int i=15-4*current_row;i>11-4*current_row;i--){
				//if((int)(((curr_state/POWER[i])%16))!=16-i){
					//int position=findBlank(curr_state,16-i);
					//int row=(position)/4;
					//int col=position%4;
					//int next_distance=abs(row-next_blank_position/4)+abs(col-next_blank_position%4);
					//int curr_distance=abs(row-curr_blank_position/4)+abs(col-curr_blank_position%4);

					//if(next_distance < min_next_distance)min_next_distance=next_distance;
					//if(curr_distance < min_curr_distance)min_curr_distance=curr_distance;
				
				//}
			//}
			//reward -= min_next_distance;
			//reward += min_curr_distance;

		}
	}
	return reward;
}

void findMax(const state& curr_state, const vector<state>& connected_states, const vector<float>& value, const vector<string>& policy,int current_row, pair<string, float>& max_value){
	for(auto connected_state: connected_states){
		int index_connected_state=index_map[connected_state.compressed_state];
		float value_of_pair = (float)R(curr_state, connected_state,policy,current_row)+0.9*value[index_connected_state];
		if(max_value.second<value_of_pair){
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
			 findMax(current_state, itr->second, value_curr, policy, current_row, next_best_state);

			 policy[index]=next_best_state.first;
			 value_next[index]=next_best_state.second;
		 }
		 delta = difference(value_next,value_curr);
		 cout << delta << "\n"<<flush;
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
