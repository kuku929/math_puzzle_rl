#include "globals.h"
#include <vector>
#include <cstdio>
#include <string>

//model parameters
float EPSILON=1.0f;
float INITIAL_LEARNING_RATE=0.0009f;
float GRAD_DECAY=0.9f;
size_t BATCH_SIZE=30;
size_t REPLAY_SIZE=1000000;
float GAMMA=0.99f; 
int UPDATE_COUNT=50;
int STEPS_PER_EPISODE=10000; 
float OUT_OF_BOUNDS=-0.5f;
float COMPLETION_REWARD=1.0f;	
int EPOCH=0;
int OPTIMIZER=1;

//Network parameters
std::vector<size_t> LAYER_SIZES({81,128,128,4});
std::vector<std::string> ACTIVATION_FUNCTIONS({"ReLU","ReLU","Linear"});
