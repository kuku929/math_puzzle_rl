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

void findMoves(state& curr_state, vector<state> &connected_states, int current_row){
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
			if(current_row == 2){
				next_state.compressed_state[curr_blank_position]=curr_state.compressed_state[next_blank_position];
				next_state.compressed_state[next_blank_position]=1+'A';

			}
			else{
				next_state.compressed_state[curr_blank_position]=curr_state.compressed_state[next_blank_position];
				next_state.compressed_state[next_blank_position]=15+'A';
			}
			next_state.blank_position=next_blank_position;
			connected_states.push_back(next_state);
		}
		
	}
}


void findStates(unordered_map<string, vector<state>> &possible_states, int current_row){
	possible_states.clear();
	state final_state;
	if(current_row == 2){
		for(int i=8;i<15;i++)final_state.compressed_state[i]=i+1+'A';
		final_state.compressed_state[15]=1+'A';
	}

	else{
		int first_element=current_row*4+1;
		for(int i=first_element-1;i<first_element+3;i++)final_state.compressed_state[i]=i+1+'A';
		final_state.compressed_state[(current_row+1)*4+3]=15+'A';
	}

	final_state.blank_position=(current_row+1)*4+3; //last position of the row after current_row
	queue<state> q;
	q.push(final_state);
	int index=0;

	ofstream ffout("temp_index_map.txt", ios::app);
	//int iter=0;
	while(!q.empty()){
		//if(iter++>1000)return;
		state curr_state=q.front();
		q.pop();
		if(possible_states.find(curr_state.compressed_state)==possible_states.end()){
			vector<state> connected_states;
			findMoves(curr_state, connected_states,current_row);
			if(connected_states.size()>0){
				possible_states.emplace(curr_state.compressed_state,connected_states);
				index_map.emplace(curr_state.compressed_state, index);
				ffout << curr_state.compressed_state << ' '<<index<<'\n';
				index++;
				for(auto state: connected_states)q.push(state);
			}
			else{
				cout << "State with no connections found!\n";
				return;
			}
		}
	}
	ffout.close();
	return;
}


bool isFinal(const state& given_state, int current_row){
	if(current_row == 2){
		for(int i=8;i<15;i++){
			if(given_state.compressed_state[i]!=i+1+'A')return false;
		}
	}
	else{
		for(int i=current_row*4;i<current_row*4+4;i++){
			if(given_state.compressed_state[i]!=i+1+'A')return false;
		}
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

int find_the_move(state& first_state, state next_state, int current_row){
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
