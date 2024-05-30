#include "globals.h"
#include <vector>
#include <string>
#include <cstdio>
//what are you changing:
float INITIAL_LEARNING_RATE=1.0f;
float GRAD_DECAY=0.0f;
int OPTIMIZER=0;

std::vector<int> LAYER_SIZES({2,4,2,1});
std::vector<std::string> ACTIVATION_FUNCTIONS({"ReLU","sigmoid", "Linear"});
