#include <iostream>
#include <stdlib.h>
#include <vector>
using namespace std;
int random_no(){
	return rand()%20;
}
struct tp{
	int c,b;
	tp(int c,int b):c(c), b(b){};
	tp():c(1), b(2){};
	tp &operator=(tp &other_state){
		this->c=other_state.c;
		this->b=other_state.b;

		return *this;
	}
};
int main(){
	vector<vector<float>> weights(2);
	//tp t1(2,5);
	//tp t2;
	//cout << t2.c<<' '<<t2.b<<endl;
	//t2=t1;
	//cout << t2.c<<' '<<t2.b<<endl;
}
