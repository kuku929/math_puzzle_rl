#include "rl_utils.h"
#include <iostream> 
#include <fstream> 
#include <vector>
#include <unordered_map>
#include <queue>
#include <string>
#include <bitset>
#include <cmath>
#include <unistd.h>
using namespace std;
#define MAX_PRIME 524160 

bool isFinal(const state& given_state){
	for(int i=0;i<15;i++){
		if(given_state.compressed_state[i]!=i+1+'A')return false;
	}

	return true;
}

void print_state(const state& given_state){
	for(int i=0;i<4;i++){
		for(int j=0;j<4;j++)cout << (given_state.compressed_state[4*i+j]-'A')<<' ';
		cout << '\n';
	}
	cout << "*******************\n";
	return;
}

void print_all_states(unordered_map<string,vector<state>>& possible_states){
	for(auto t=possible_states.begin();t!=possible_states.end();t++){
		cout << "current state\n";
		for(int i=0;i<4;i++){
			for(int j=0;j<4;j++)cout << (t->first[4*i+j]-'A')<<' ';
			cout << '\n';
		}
		cout<<"******************\n";
		for(auto u: t->second){
			for(int i=0;i<4;i++){
				for(int j=0;j<4;j++)cout <<u.compressed_state[4*i+j]-'A' <<' ';
				cout<<'\n';
			}
			cout<<"********************\n";
		}
	}
}

int find_the_move(state &first_state, state &next_state, int current_row){
	int blank_value = (current_row==2) ? 1:15;
	int first_blank_position = first_state.find_number(blank_value); 
	int next_blank_position = next_state.find_number(blank_value);

	return (next_blank_position-first_blank_position);
}
void move_puzzle(int blank_position, int move_no, int arr[16]){
	int blank_value = arr[blank_position];
	int next_blank_position = blank_position+move_no;

	arr[blank_position] = arr[next_blank_position];
	arr[next_blank_position] = blank_value;
	
	return;
}
void beautiful_print(int arr[16]){
	cout << "_____________\n"<<flush;
	for(int i=0;i<4;i++){
		for(int j=0;j<4;j++){
			if(arr[4*i+j]<10)cout <<"| "<< arr[4*i+j]<<flush;
			else cout<< "|"<<arr[4*i+j]<<flush;
		}
		cout << "|\n____________\n"<<flush;
	}
	cout << "\n\n********************\n\n"<<flush;
	usleep(500000);
	return;
}

vector<int> possible_moves(const state &curr_state){
	int curr_blank_position=curr_state.blank_position;
	vector<int> action_space;
	for(int i=0;i<4;i++){
		int possible=1;
		switch(i){
			case 0:
				if(curr_blank_position/4==0)possible=0;
				break;
			case 1:
				if(curr_blank_position%4==3)possible=0;
				break;
			case 2:
				if(curr_blank_position/4==3)possible=0;
				break;
			case 3:
				if(curr_blank_position%4==0)possible=0;
				break;
		}
		if(possible)
			action_space.push_back(i);
	}	
	return action_space;	
}


state move(const state &old_state, int action){
	state curr_state(old_state); 
	int curr_blank_position=curr_state.blank_position;
	int next_blank_position;
	int possible=0;
	switch(action){
		case 0:
			if(curr_blank_position>3)possible=1;
			next_blank_position=curr_blank_position-4;
			break;
		case 1:
			if(curr_blank_position%4<3)possible=1;
			next_blank_position=curr_blank_position+1;
			break;
		case 2:
			if(curr_blank_position<12)possible=1;
			next_blank_position=curr_blank_position+4;
			break;
		case 3:
			if(curr_blank_position%4>0)possible=1;
			next_blank_position=curr_blank_position-1;
			break;
	}
	if(possible){
		curr_state.compressed_state[curr_blank_position]=curr_state.compressed_state[next_blank_position];
		curr_state.compressed_state[next_blank_position]='A';
		curr_state.blank_position=next_blank_position;
		return curr_state;
	}
	return curr_state;
}
