#include <cmath> 
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <unistd.h>
using namespace std;
int SIZE = 2;
vector<int> possible_moves(int curr_blank_position){
	/*
	 *finds all the possible moves in the current state
	 *returns an array of all the possible moves
	 */
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



void move(int a[], int action, int &curr_blank_position){
	/*
	 *moves the blank in the direction of the action
	 *changes the curr_state struct
	 */
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
		a[curr_blank_position]=a[next_blank_position];
		a[next_blank_position]=0;
		curr_blank_position=next_blank_position;
	}
	return;
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
	//usleep(500000);
	return;
}

int main(int argc, char **argv){
	srand(chrono::high_resolution_clock::now().time_since_epoch().count());

	unsigned int size=0; //the number of starting positions
	unsigned int max_moves=0; //maximum no of moves to do 
	unsigned int min_moves=0; //minimum no of moves to do before saving
	int c, flag=0;
	while((c = getopt(argc, argv, "m:M:s:rh")) != -1){ //u can't use 'r' because arguments require const char *, so use string
		switch(c){
		case 'r':
			flag=1;
			break;
		
		
		case 'm':
			min_moves = atoi(optarg);
			break;
		
		
		case 'M':
			max_moves = atoi(optarg);
			break;
		
		
		case 's':
			size = atoi(optarg);
			break;

		case 'h':
			cout << "this program supports the following flags: "<< '\n';
			cout << " -s : the size of the training set.( REQUIRED )\n";
			cout << " -m : the minimum number of moves.( REQUIRED )\n";
			cout << " -M : the maximum number of moves.( REQUIRED )\n";
			cout << " -r : put this flag if you want to randomly order the training set. otherwise the set will be ordered based on no of moves( ascending ).\n";
		
			return 0;
		}
			
	}

	//sanity check
	if(size == 0 || max_moves == 0){
		cout << "ERROR! you have entered invalid values of either size or maximum no of moves.\n";
		return 0;
	}

	//initializing puzzle
	int a[SIZE*SIZE];
	int blank_position = SIZE*SIZE-1;
	vector<int> action_space;	

	//getting input from user
	int no_of_moves = min_moves;
	ofstream fout("train.txt");
	int prev_action=0;


	if(flag){ //random no of moves
		//initializing the puzzle to original state
		for(int i=0;i<SIZE*SIZE-1;i++)a[i]=i+1; 
		a[SIZE*SIZE-1]=0;
		blank_position=SIZE*SIZE-1;

		for(int j=0;j<size;++j){
			for(int i=0;i<SIZE*SIZE-1;i++)a[i]=i+1; 
			a[SIZE*SIZE-1]=0;
			blank_position=SIZE*SIZE-1;
			no_of_moves = min_moves + rand()%(max_moves-min_moves+1);
			for(int i=0;i<no_of_moves;i++){ //moving
				action_space=possible_moves(blank_position);
				int action = rand()%action_space.size(); 
				if(abs(prev_action-action_space[action])==2){ //if redundant move, do again
					i--;
					continue;
				}
				move(a, action_space[action], blank_position);
				prev_action=action_space[action];
			}
		for(int i=0;i<SIZE*SIZE;i++)fout<<static_cast<char>(a[i]+static_cast<int>('A'));
		fout<<'\n';
		}

	}
	else{
		for(int j=0;j<size;j++){
			//initializing the puzzle to original state
			for(int i=0;i<SIZE*SIZE-1;i++)a[i]=i+1; 
			a[SIZE*SIZE-1]=0;
			blank_position=SIZE*SIZE-1;

			for(int i=0;i<no_of_moves;i++){ //moving
				for(int i=0;i<SIZE*SIZE-1;i++)a[i]=i+1; 
				a[SIZE*SIZE-1]=0;
				blank_position=SIZE*SIZE-1;

				action_space=possible_moves(blank_position);
				int action = rand()%action_space.size(); 
				if(abs(prev_action-action_space[action])==2){ //if redundant move, do again
					i--;
					continue;
				}
				move(a, action_space[action], blank_position);
				prev_action=action_space[action];
			}
			no_of_moves=min_moves + (j*max_moves)/size; //we keep increasing the amount of shuffling 

			//saving
			for(int i=0;i<SIZE*SIZE;i++)fout<<(char)(a[i]+'A');
			fout<<'\n';
		}
	}
}
