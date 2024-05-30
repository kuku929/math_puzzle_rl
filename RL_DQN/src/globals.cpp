#include "globals.h"
#include <cstdio>
//what are you changing:
float EPSILON=1.0f;
float INITIAL_LEARNING_RATE=0.0009f;
float GRAD_DECAY=0.9f;
size_t BATCH_SIZE=30;
size_t REPLAY_SIZE=1000000;
float GAMMA=0.5f; 
int UPDATE_COUNT=50;
int STEPS_PER_EPISODE=10000; 
float OUT_OF_BOUNDS=-0.5f;
float COMPLETION_REWARD=1.0f;	
int EPOCH=0;
int OPTIMIZER=1;

int INPUT_SIZE = 9;
int HIDDEN_SIZE = 20;
int OUTPUT_SIZE = 4;

//0.99 : q_values: 0.0228765 -0.644293 0.0334103 0.00411912 
//0.5 : q_values: -0.0816154 -0.0471954 -0.00572923 -0.0591394 
//0.1 : q_values: -0.0137376 -0.0473945 -0.0235255 -0.0213767 
//0.00141277 0.00251007 0.00194986 0.00170688

// model parameters
//float EPSILON=1.0f;
//float INITIAL_LEARNING_RATE=0.09f;
//float GRAD_DECAY=0.99f;
//size_t BATCH_SIZE=30;
//size_t REPLAY_SIZE=1000000;
//float GAMMA=0.99f; 
//int UPDATE_COUNT=50;
//int STEPS_PER_EPISODE=10000; 
//float OUT_OF_BOUNDS=-0.5f;
//float COMPLETION_REWARD=2.0f;	
//int EPOCH=0;
//int OPTIMIZER=0;

////Network parameters
//int INPUT_SIZE = 4;
//int HIDDEN_SIZE = 7;
//int OUTPUT_SIZE = 4;
