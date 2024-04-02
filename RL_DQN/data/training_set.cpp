#include <cmath> 
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
using namespace std;
vector<int> possible_moves(int curr_blank_position){
	vector<int> action_space;
	for(int i=0;i<4;i++){
		int possible=1;
		switch(i){
			case 0: if(curr_blank_position/4==0)possible=0;
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

void move(int curr_state[16], int action, int &curr_blank_position){
	int next_blank_position;
	switch(action){
		case 0:
			next_blank_position=curr_blank_position-4;
			break;
		case 1:
			next_blank_position=curr_blank_position+1;
			break;
		case 2:
			next_blank_position=curr_blank_position+4;
			break;
		case 3:
			next_blank_position=curr_blank_position-1;
			break;
	}
	curr_state[curr_blank_position]=curr_state[next_blank_position];
	curr_state[next_blank_position]=0;
	curr_blank_position=next_blank_position;
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
	return;
}

int main(int argc, char **argv){
	srand(chrono::high_resolution_clock::now().time_since_epoch().count());
	if(argc<4){
		cout << "enter the size, starting moves and the total moves.\n";
		return 0;
	}
	int a[16];
	for(int i=0;i<15;i++)a[i]=i+1;
	a[15]=0;
	int blank_position = 15;
	vector<int> action_space;	
	int size=atoi(argv[1]);
	int total_moves=atoi(argv[3]);
	int no_of_moves=atoi(argv[2]);
	int max_repititions=5;
	ofstream fout("train.txt");
	int prev_action=0;
	for(int j=0;j<size;j++){
		for(int i=0;i<15;i++)a[i]=i+1;
		a[15]=0;
		blank_position=15;
		for(int i=0;i<no_of_moves;i++){
			action_space=possible_moves(blank_position);
			int action = rand()%action_space.size();
			if(abs(prev_action-action_space[action])==2){
				continue;
				i--;
			}
			move(a, action_space[action], blank_position);
			prev_action=action_space[action];
		}
		no_of_moves+=(j*total_moves)/size;
		//for(int k=0;k<max_repititions;k++){
		for(int i=0;i<16;i++)fout<<(char)(a[i]+'A');
		fout<<'\n';
		//beautiful_print(a);
		//}
		//max_repititions=5-(j*5)/size;
	}
}
