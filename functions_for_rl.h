#pragma once
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

extern unordered_map<usl, int> index_map;
extern usl POWER[16];


struct state{
	usl compressed_state;
	int blank_position=-1;

	void find_blank(){
		if(blank_position==-1){
			for(int i=0;i<16;i++){
				if((compressed_state/POWER[i])%16==0){
					this->blank_position=15-i;	
					return;
				}
			}
		}
		return;
	}

	state():compressed_state(0){};
	
	state(usl some_state):compressed_state(some_state){
		this->find_blank();
	};
	
	state(int arr[16]){
		this->compressed_state=0;
		for(int i=15;i>-1;i--)this->compressed_state+=(arr[15-i])*(POWER[i]);
		this->find_blank();
	};

	void decompress(int arr[16]){
		for(int i=15;i>-1;i--)arr[i]=((this->compressed_state)/(POWER[15-i]))%16;
		return;
	}
};


usl power(int x, int n);
void findMoves(state curr_state, vector<state> &connected_states, int current_row);
void findStates(unordered_map<usl, vector<state>> &possible_states, int current_row);
int findBlank(usl curr_state, int blank_no);
bool isFinal(usl given_state);
int R(usl curr_state, usl next_state, vector<usl> policy);
void print_state(usl given_state);
pair<usl, float> findMax(usl curr_state, vector<state> connected_states, vector<float> value, vector<usl> policy);
float difference(vector<float> value_next, vector<float> value_curr);

