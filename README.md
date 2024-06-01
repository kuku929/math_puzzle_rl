# $N^2-1$ puzzle solver using different Reinforcement Learning techniques 

This project focusses on two different algorithms to find mostly optimal solutions to the 15-puzzle and in general to a $N^2-1$ puzzle.

I have explored the following algorithms:
- Value Iteration
- Deep Q-Learning


## About the puzzle

If you have not played the 15-puzzle, you can try it [here][https://15puzzle.netlify.app/]


the generalization to the 15-puzzle is an $N^2-1$ puzzle. The $N^2-1$ puzzle is a widely known puzzle. It is also an excellent benchmark to test heuristic search algorithms like A*. 

The puzzle is challenging to train RL algorithms on owing to its huge state space, the 15-puzzle has $15!/2$ "solvable" states which is $6.5\times10^11$. Thus, most algorithms focus on sub-optimal solutions which are fast to train. 

The $N^2-1$-puzzle is an NP-hard problem, thus optimal solutions in reasonable time are nearly impossible.

To read further about the puzzle and its solutions, here is an interesting [paper][https://pdfs.semanticscholar.org/d3f3/fa96e6414585900422467c0042c4665dd98b.pdf]
