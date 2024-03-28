#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <bitset>
#include <cmath>
#include <unistd.h>
#include <algorithm>
using namespace std;
#define MAX_PRIME 524160 
extern ofstream dout;

struct state{
	int blank_position;
	string compressed_state;

	state(){
		this->compressed_state = string(16,'A');
	};
	
	state(const state &some_state){
		this->compressed_state = some_state.compressed_state;
		this->blank_position = some_state.blank_position;
	};
	
	state(int arr[16]){
		this->compressed_state="";
		for(int i=0;i<16;i++){
			this->compressed_state+=arr[i]+'A';	
		}
	};

	int find_number(int n)const{
		auto itr = find(this->compressed_state.begin(), this->compressed_state.end(), n+'A');
		if(itr!=this->compressed_state.end())return (itr-this->compressed_state.begin());
		return -1;
	}


};

void findMoves(state &curr_state, vector<state> &connected_states, int current_row);
void findStates(unordered_map<string, vector<state>> &possible_states, int current_row);
bool isFinal(const state &given_state);
void print_state(const state &given_state);
void print_all_states(unordered_map<string,vector<state>>& possible_states);
int find_the_move(state &first_state, state &next_state, int current_row);
void move_puzzle(int blank_position, int move_no, int arr[16]);
void beautiful_print(int arr[16]);
vector<int> possible_moves(const state &curr_state);
void move(state &curr_state, int action);
