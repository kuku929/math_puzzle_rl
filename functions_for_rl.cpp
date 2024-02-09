#include "functions_for_rl.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <bitset>
#include <cmath>
typedef unsigned long long int usl;
using namespace std;
#define MAX_PRIME 524171

usl power(int x, int n){
	if(n == 1) return x;
	if(n==0) return 1;

	else{
		usl y = power(x, n/2);
		if(n&1){
			return ((y*y)*x);
		}else{
			return (y*y);
		}
	}
}




void findMoves(state curr_state, vector<state> &connected_states, int current_row){
	int curr_blank_position=curr_state.blank_position;
	for(int i=0;i<4;i++){
		int possible=1;
		int next_blank_position;
		switch(i){
			case 0:
				if(curr_blank_position/4==current_row)possible=0;
				next_blank_position=curr_blank_position-4;
				
				break;
			case 1:
				if(curr_blank_position%4==3)possible=0;
				else next_blank_position=curr_blank_position+1;
				
				break;
			case 2:
				if(curr_blank_position/4==3)possible=0;
				else next_blank_position=curr_blank_position+4;
				
				break;
			case 3:
				if(curr_blank_position%4==0)possible=0;
				else next_blank_position=curr_blank_position-1;

				break;
		}
		if(possible){
			state next_state(curr_state.compressed_state);
			int adjacent_tile=(curr_state.compressed_state/POWER[15-next_blank_position])%16;
			next_state.compressed_state+=(15-adjacent_tile)*POWER[15-next_blank_position];
			next_state.compressed_state-=(15-adjacent_tile)*POWER[15-curr_blank_position];
			next_state.blank_position=next_blank_position;
			connected_states.push_back(next_state);
		}
		
	}
}


void findStates(unordered_map<usl, vector<state>> &possible_states, int current_row){
	int first_element=current_row*4+1;
	int final_state_array[16]={0};
	for(int i=first_element-1;i<first_element+3;i++)final_state_array[i]=i+1;
	final_state_array[(current_row+1)*4+3]=15;
	state final_state(final_state_array);
	final_state.blank_position=(current_row+1)*4+3; //last position of the row after current_row
	queue<state> q;
	q.push(final_state);
	int index=0;
	//int iter=0;
	while(!q.empty()){
		//if(iter++>1000)return;
		state curr_state=q.front();
		q.pop();
		if(possible_states.find(curr_state.compressed_state)==possible_states.end()){
			vector<state> connected_states;
			findMoves(curr_state, connected_states,current_row);
			if(connected_states.size()>0){
				//cout << "in";
				possible_states.emplace(curr_state.compressed_state,connected_states);
				index_map.emplace(curr_state.compressed_state, index);
				index++;
				for(auto state: connected_states)q.push(state);
			}
		}
	}
	return;
}

int findBlank(usl curr_state, int blank_no){
	for(int i=0;i<16;i++){
		if((int)((curr_state/POWER[i])%16)==blank_no)return (15-i);
	}
	return -1;
}

bool isFinal(usl given_state){
	for(int i=15;i>11;i--){
		if((int)((given_state/POWER[i])%16)!=16-i)return false;
	}
	return true;
}

int R(usl curr_state, usl next_state, vector<usl> policy){
	if(isFinal(next_state))return 100; 
	int next_state_index=index_map[next_state];

	if(policy[next_state_index]==curr_state)return -100;
	int curr_blank_position=findBlank(curr_state,15);
	int next_blank_position=findBlank(next_state,15);
	int tile_swapped = (curr_state/POWER[15-next_blank_position])%16;	
	//int reward = (6-curr_blank_position%4-curr_blank_position/4)-(6-next_blank_position%4-next_blank_position/4);
	int reward=0;
	if(tile_swapped!=0){
		int row=(tile_swapped-1)/4;
		int col=(tile_swapped-1)%4;

		reward += abs(row-next_blank_position/4)+abs(col-next_blank_position%4);
		reward -= abs(row-curr_blank_position/4)+abs(col-curr_blank_position%4);
	}
	else{
		vector<int> not_in_place;
		for(int i=15;i>11;i--){
			if((int)(((curr_state/POWER[i])%16))!=16-i)not_in_place.push_back(16-i);
		}
		for(auto t: not_in_place){
			int position=findBlank(curr_state,t);
			int row=(position)/4;
			int col=position%4;

			reward -= abs(row-next_blank_position/4)+abs(col-next_blank_position%4);
			reward += abs(row-curr_blank_position/4)+abs(col-curr_blank_position%4);
		}
	}
	return reward;
}

void print_state(usl given_state){
	for(int i=3;i>-1;i--){
		for(int j=3;j>-1;j--)cout << (given_state/POWER[4*i+j])%16<<' ';
		cout << '\n';
	}
	cout << "*******************\n";
	return;
}

pair<usl, float> findMax(usl curr_state, vector<state> connected_states, vector<float> value, vector<usl> policy){
	pair<usl,float> max_value(10.0,-3);
	for(auto connected_state: connected_states){
		int index_connected_state=index_map[connected_state.compressed_state];
		//cout << index_connected_state<<' ';
		float value_of_pair = (float)R(curr_state, connected_state.compressed_state,policy)+0.9*value[index_connected_state];
		if(max_value.second<value_of_pair){
			max_value.second=value_of_pair;
			max_value.first=connected_state.compressed_state;
		}
	}
	return max_value;
}

float difference(vector<float> value_next, vector<float> value_curr){
	float d=0;
	for(int i=0;i<(int)value_next.size();i++){
		d+=fabs(value_next[i]-value_curr[i]);
	}
	return d;
}

