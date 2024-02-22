#include "NaiveQLearning.h"
#include "rl_utils.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <string>
#include <bitset>
#include <cmath>
#include <unistd.h>
unordered_map<string, int> index_map;
unordered_map<string,vector<state>> possible_states;
using namespace std;
#define MAX_PRIME 524160 

int main(int argc, char **argv){

	string input_filename;
	int input_flag=0;
	int flag=0;
	int train_flag=0;
	int c;
	float threshold = 100.0;
	int arr[16];
	while((c=getopt(argc,argv,"pt:i:")) != -1){
		switch(c){
		case 'p':
			flag=1;
			break;
		
		case 't':
			train_flag=1;
			threshold=atof(optarg);
			break;
		
		case 'i':
			input_flag=1;
			input_filename.assign(optarg);
			break;
		}
	}

	if(input_flag){
		ifstream input(input_filename);	
		for(int i=0;i<16;i++){
			input>>arr[i];
		
		}
		input.close();
	}
	vector<string> policy(MAX_PRIME,string(16,'A'));
	vector<float> value_curr(MAX_PRIME,0);
	vector<float> value_next(MAX_PRIME,0);
	int no_of_moves=0;
	if(train_flag){
		ofstream fout("temp_policy.txt");
		ofstream ffout("temp_value.txt");
		ofstream fffout("temp_index_map.txt");
		fout.close();
		ffout.close();
		fffout.close();
	}

	if(flag){
		cout << "IN";
		ifstream fin("policy.txt");
		ifstream ffin("index_map.txt");
		ifstream fffin("value.txt");	
	
		int index=0;
		string state_val;
		int corresponding_index;
		string policy_val;
		int current_row=0;

		while(fin){
			fin>>policy_val;
			if(policy_val=="done"){
				if(index==0)break;
				index=0;
				if(train_flag){
					findStates(possible_states,current_row);
					train(possible_states,value_curr,value_next,policy,threshold,1, current_row); 	
				}
				
				if(input_flag)no_of_moves+=play(arr, policy, current_row);
				current_row++;
				index_map.clear();
				continue;	
			}
			policy[index++]=policy_val;
			ffin>>state_val>>corresponding_index;
			if(train_flag)fffin>>value_curr[index-1];
			index_map.emplace(state_val, corresponding_index);
		}
		fin.close();
		ffin.close();
		fffin.close();

	}
	else if(train_flag){
		for(int current_row=0;current_row<3;current_row++){
			findStates(possible_states,current_row);
			train(possible_states,value_curr,value_next,policy,threshold,1, current_row); 	
			if(input_flag)no_of_moves+=play(arr,policy,current_row); 
			index_map.clear();
		}
	}

	else{
		cout << "enter the no of generations with the -n flag\n";
	}
	if(train_flag){
		remove("policy.txt");
		remove("index_map.txt");
		remove("value.txt");
		rename("temp_policy.txt", "policy.txt");
		rename("temp_value.txt", "value.txt");
		rename("temp_index_map.txt", "index_map.txt");
	}
	if(input_flag){
		beautiful_print(arr);	
		cout << "total moves: "<<no_of_moves<<'\n';
	}

	return 0;
}




