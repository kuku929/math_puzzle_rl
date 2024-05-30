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
extern ofstream dout;
int SIZE=3;
bool isFinal(const state& given_state){
	/*
	 *checks if the puzzle has reached final state
	 */
	for(int i=0;i<SIZE*SIZE-1;i++){
		if(given_state.compressed_state[i]!=static_cast<char>(i+1+static_cast<int>('A')))
				return false;
	}

	return true;
}

void print_state(const state& given_state){
	/*
	 *prints the state in a matrix form
	 */
	for(int i=0;i<SIZE;i++){
		for(int j=0;j<SIZE;j++)dout << static_cast<int>(given_state.compressed_state[SIZE*i+j]-'A')<<' ';
		dout << '\n';
	}
	dout << "*******************\n";
	return;
}

void print_all_states(unordered_map<string,vector<state>>& possible_states){
	/*
	 *prints all the states in an array
	 */
	for(auto t=possible_states.begin();t!=possible_states.end();t++){
		cout << "current state\n";
		for(int i=0;i<SIZE;i++){
			for(int j=0;j<SIZE;j++)cout << static_cast<int>(t->first[SIZE*i+j]-'A')<<' ';
			cout << '\n';
		}
		cout<<"******************\n";
		for(auto u: t->second){
			for(int i=0;i<SIZE;i++){
				for(int j=0;j<SIZE;j++)cout <<static_cast<int>(u.compressed_state[SIZE*i+j]-'A') <<' ';
				cout<<'\n';
			}
			cout<<"********************\n";
		}
	}
}

void beautiful_print(int arr[]){
	/*
	 *sexy print
	 */
	cout << "_____________\n"<<flush;
	for(int i=0;i<SIZE;i++){
		for(int j=0;j<SIZE;j++){
			if(arr[SIZE*i+j]<10)cout <<"| "<< arr[SIZE*i+j]<<flush;
			else cout<< "|"<<arr[SIZE*i+j]<<flush;
		}
		cout << "|\n____________\n"<<flush;
	}
	cout << "\n\n********************\n\n"<<flush;
	usleep(500000);
	return;
}

vector<int> possible_moves(const state &curr_state){
	/*
	 *finds all the possible moves in the current state
	 *returns an array of all the possible moves
	 */
	int curr_blank_position=curr_state.blank_position;
	vector<int> action_space; 
	for(int i=0;i<4;i++){
		int possible=1;
		switch(i){
			case 0:
				if(curr_blank_position/SIZE==0)possible=0;
				break;
			case 1:
				if(curr_blank_position%SIZE==SIZE-1)possible=0;
				break;
			case 2:
				if(curr_blank_position/SIZE==SIZE-1)possible=0;
				break;
			case 3:
				if(curr_blank_position%SIZE==0)possible=0;
				break;
		}
		if(possible)
			action_space.push_back(i);
	}	
	return action_space;	
}


void move(state &curr_state, int action){
	/*
	 *moves the blank in the direction of the action
	 *changes the curr_state struct
	 */
	int curr_blank_position=curr_state.blank_position;
	int next_blank_position;
	int possible=0;
	switch(action){
		case 0:
			if(curr_blank_position>SIZE-1)possible=1;
			next_blank_position=curr_blank_position-SIZE;
			break;
		case 1:
			if(curr_blank_position%SIZE<SIZE-1)possible=1;
			next_blank_position=curr_blank_position+1;
			break;
		case 2:
			if(curr_blank_position<SIZE*(SIZE-1))possible=1;
			next_blank_position=curr_blank_position+SIZE;
			break;
		case 3:
			if(curr_blank_position%SIZE>0)possible=1;
			next_blank_position=curr_blank_position-1;
			break;
	}
	if(possible){
		curr_state.compressed_state[curr_blank_position]=curr_state.compressed_state[next_blank_position];
		curr_state.compressed_state[next_blank_position]='A';
		curr_state.blank_position=next_blank_position;
	}
	return;
}
