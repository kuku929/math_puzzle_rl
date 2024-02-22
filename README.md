Aim: Implementing Different Reinforcement Learning algorithms for solving the 15 puzzle and its variants.

If you don't know about the 15-puzzle, try it here: https://15puzzle.netlify.app/

to run the code locally, follow the given instructions: 

1) download the repo on your local machine. 
2) create a build directory inside your project folder and cd into it
3) once inside, run cmake .. -G Ninja
4) then run ninja
5) you will now see an executable named rl
6) run the code using ./rl -p -t (float) -i (input filename)

allowed flags to be used:

-p : one can use an existing policy by putting this flage while running, 
     keep in mind the policy must be stored in the same directory as the executable and should be named "policy.txt"
     
-t : specifies the threshold for training the model, default value is 100.0. Pass this flag if you want to train the model further.

-i : input flag, passing it will prompt the model to play the 15 puzzle game with the initial state as described in the input.txt
     an example input.txt has been provided. Please refer to it for the format of the input.

