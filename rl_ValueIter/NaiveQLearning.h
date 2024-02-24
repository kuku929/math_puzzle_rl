#pragma once
#include "rl_utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <string>
#include <bitset>
#include <cmath>
using namespace std;
#define MAX_PRIME 524160 

int R(const state &curr_state,const state &next_state, vector<string> &policy, int &current_row);
void findMax(const state &curr_state, const vector<state> &connected_states, const vector<float>& value, const vector<string>& policy, int current_row, pair<string, float>& max_value);
float difference(vector<float> &value_next, vector<float> &value_curr);
void train(unordered_map<string, vector<state>>& possible_states, vector<float> &value_curr, vector<float> &value_next, vector<string> &policy, float threshold, int save_file, int current_row);
int play(int arr[16], vector<string>& policy, int current_row);
