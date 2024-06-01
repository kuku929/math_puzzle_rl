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
extern int SIZE;
struct state{
	int blank_position;
	string compressed_state;

	state(){
		this->compressed_state = string(SIZE*SIZE,'A');
	};
	
	state(const state &some_state){
		this->compressed_state = some_state.compressed_state;
		this->blank_position = some_state.blank_position;
	};
	
	state(int arr[]){ 
		this->compressed_state="";
		for(int i=0;i<SIZE*SIZE;i++){
			this->compressed_state+=static_cast<char>(arr[i]+static_cast<int>('A'));	
		}
	};

	state(vector<int> &v){ //left hand referance, use when passing a variable as an argument while calling the function
		this->compressed_state="";
		for(int i=0;i<SIZE*SIZE;i++){
			this->compressed_state+=static_cast<char>(v[i]+static_cast<int>('A'));	
		}
	};

	state(vector<int> &&v){ //right hand reference so that vector is not copied
		this->compressed_state="";
		for(int i=0;i<SIZE*SIZE;i++){
			this->compressed_state+=static_cast<char>(v[i]+static_cast<int>('A'));	
		}
	};
	state& operator=(const state&) = default;

	int find_number(int n)const{ //find the number n in the puzzle board
		auto itr = find(this->compressed_state.begin(), this->compressed_state.end(), char(n+int('A')));
		if(itr!=this->compressed_state.end())return (int(itr-this->compressed_state.begin())); //normal subtraction gives long int?
		return -1;
	}


};

bool isFinal(const state &given_state); //check if final state reached
void print_state(const state &given_state); //print a state
void beautiful_print(const state& some_state); //print a state, beautifully
vector<int> possible_moves(const state &curr_state); //find all possible moves in a state
void move(state &curr_state, int action); //move the puzzle according to the action
