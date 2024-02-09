/*
 *NOTE: 0 in the state is the empty blank.
 * also, make a power array to access the power, will reduce the time.
 */
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
unordered_map<usl, int> index_map;
usl POWER[16];

int main(){
	ofstream fout("policy.txt");
	ofstream ffout("index_map.txt");
	for(int i=0;i<16;i++)POWER[i]=power(16,i);
	int arr[16];
	for(int i=0;i<16;i++){
		cin>>arr[i];
	}
	int first_row_initial_state[16]={0};
	for(int i=0;i<16;i++){
		if(arr[i]==0)first_row_initial_state[i]=15;
		else if(0<arr[i]&&arr[i]<5)first_row_initial_state[i]=arr[i];
	}
	state first_row_state(first_row_initial_state);
	unordered_map<usl,vector<state>> possible_states;
	findStates(possible_states, 0);
	usl some_state = (possible_states.begin()->first);
	state next_state = (possible_states.begin()->second)[0];
	//for(int i=3;i>-1;i--){
		//for(int j=3;j>-1;j--)cout << (some_state/POWER[4*i+j])%16<<' ';
		//cout << '\n';
	//}
	//for(int i=3;i>-1;i--){
		//for(int j=3;j>-1;j--)cout << (next_state.compressed_state/POWER[4*i+j])%16<<' ';
		//cout << '\n';
	//}

	//vector<usl> policy(MAX_PRIME,0);
	//int reward = R(some_state, next_state.compressed_state, policy);
	//cout << reward;
	
	//int blank=findBlank(first_row_state.compressed_state,15);
	//cout << blank;
	/*
	 *for(int i=0;i<4;i++){
	 *        for(int j=0;j<4;j++)cout<<first_row_initial_state[4*i+j]<<' ';
	 *        cout<<'\n';
	 *}
	 */
	
	//for(auto t=possible_states.begin();t!=possible_states.end();t++){
		//cout << "current state\n";
		//for(int i=3;i>-1;i--){
			//for(int j=3;j>-1;j--)cout<<((t->first)/POWER[4*i+j])%16<<' ';
			//cout<<'\n';
		//}
		//cout<<"******************\n";
		//for(auto u: t->second){
			//int a[16];
			//u.decompress(a);
			//for(int i=0;i<4;i++){
				//for(int j=0;j<4;j++)cout << a[4*i+j]<<' ';
				//cout<<'\n';
			//}
			//cout<<"********************\n";
		//}
	//}

	vector<float> value_curr(MAX_PRIME,0);
	vector<float> value_next(MAX_PRIME,0);
	vector<usl> policy(MAX_PRIME,0);
	float threshold=1;
	float delta=10;
	//cout << possible_states.size()<<'\n';
	int index=0;
	//pair<usl, float> max_value = findMax(possible_states.begin()->first, possible_states.begin()->second, value_curr,policy);
	//cout<< max_value.first<<' '<<max_value.second;
	pair<usl, float> next_best_state;
	//int iter=0;
	for(auto itr=possible_states.begin();itr!=possible_states.end();itr++){
		index=index_map[itr->first];
		next_best_state = findMax(itr->first, itr->second, value_curr, policy);
		policy[index]=next_best_state.first;
		value_next[index]=next_best_state.second;
	}
	// while(delta>threshold){
	// 	iter++;
	// 	if(iter>2){
	// 		cout << "POSSIBLE OVERFLOW!\n";
	// 		break;
	// 	}
	// 	for(auto itr=possible_states.begin();itr!=possible_states.end();itr++){
	// 		index=index_map[itr->first];
	// 		pair<usl, float> next_best_state = findMax(itr->first, itr->second, value_curr, policy);
	// 		policy[index]=next_best_state.first;
	// 		value_next[index]=next_best_state.second;
	// 	}
	// 	delta = difference(value_next,value_curr);
	// 	for(int i=0;i<(int)possible_states.size();i++)value_curr[i]=value_next[i];
	// }
	// // for(auto t: possible_states)ffout<<t.first<<' '<<index_map[t.first]<<'\n';
	// // for(int i=0;i<possible_states.size();i++)fout<<policy[i]<<'\n';
	
	// usl current_state=first_row_state.compressed_state;	
	// iter=0;
	// print_state(current_state);
	// while(!isFinal(current_state)){
	// 	if(iter++>100){
	// 		cout << "CANT CONVERGE!";
	// 		return 0;
	// 	}
	// 	int current_index=index_map[current_state];
	// 	current_state=policy[current_index];
	// 	print_state(current_state);
	// }
}




