/*
 *@Author: Krutarth Patel                                           
 *@Date: 1st June 2024
 *@Description : hyperparameters for the neural network
 */

#include "globals.h"
#include <vector>
#include <cstdio>
#include <string>

//model parameters
float EPSILON=1.0f;
float INITIAL_LEARNING_RATE=0.0009f; //learning rate
float GRAD_DECAY=0.99f; //weights for the RMSProp optimizer
size_t BATCH_SIZE=30; //batch size for each training instance
size_t REPLAY_SIZE=1000000; //the maximum size of the experience replay
float GAMMA=0.7f; //weights for the expected reward
int UPDATE_COUNT=50; //no of moves after which to update the network
int STEPS_PER_EPISODE=10000; //no of moves done by the actor in one epoch/episode
float OUT_OF_BOUNDS=-0.5f; //reward recieved for doing an invalid move
float COMPLETION_REWARD=1.0f;	//reward for completing the puzzle
int EPOCH=0;
int OPTIMIZER=1; //use the optimizer optionally

//Network parameters
std::vector<size_t> LAYER_SIZES({4,32,32,4}); //architecture of the network
std::vector<std::string> ACTIVATION_FUNCTIONS({"Leaky","Leaky","Linear"}); //activation functions for the network
//following functions are supported : ReLU Leaky sigmoid Linear
