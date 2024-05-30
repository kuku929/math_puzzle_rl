#include <cstdio>
// model parameters
extern float EPSILON;
extern float INITIAL_LEARNING_RATE;
extern float GRAD_DECAY;
extern size_t BATCH_SIZE;
extern size_t REPLAY_SIZE;
extern float GAMMA;
extern int UPDATE_COUNT;
extern int STEPS_PER_EPISODE;
extern float OUT_OF_BOUNDS;
extern float COMPLETION_REWARD;	
extern int EPOCH;
extern int OPTIMIZER;

//Network parameters
extern int INPUT_SIZE;
extern int HIDDEN_SIZE;
extern int OUTPUT_SIZE;
