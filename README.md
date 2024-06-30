# CS370Final-

# Pirate Intelligent Agent Project

## Overview
This project involves creating a pirate intelligent agent for a treasure hunt game, where the player needs to find the treasure before the pirates. As an AI developer, I designed an intelligent agent (the pirate) to navigate the game world and find the treasure before the human player, which is a pathfinding problem.

## Project Details
In this project, I implemented a deep Q-learning algorithm to train the pirate agent. The provided starter code included two Python files, `TreasureMaze.py` and `GameExperience.py`, along with a Jupyter Notebook `TreasureHuntGame.ipynb`. 

### Code Provided
- **TreasureMaze.py**: Represents the environment, including a maze object defined as a matrix, and methods to handle agent movements, rewards, and game status.
- **GameExperience.py**: Stores episodes, i.e., all states between the initial and terminal states, used by the agent for learning.
- **TreasureHuntGame.ipynb**: Contains the code skeleton for the project, including neural network setup and areas to implement the Q-learning algorithm.

### Code Created
- **Q-Training Algorithm**: Implemented the Q-learning algorithm using neural networks. This involved defining the reward structure, exploration and exploitation strategies, and training loops.
- **Design Defense**: Documented the approach to solving the problem, comparing human and machine methods, and evaluating the effectiveness of the algorithm.

## Learning Reflections

### Project Explanation
I created the Q-training algorithm from scratch using the pseudocode provided. This algorithm trained the pirate agent to navigate the maze and find the treasure efficiently. Key aspects included setting up the neural network, defining the reward structure, and implementing the training loop with exploration and exploitation balance.

### Connection to Computer Science
#### What Do Computer Scientists Do and Why Does It Matter?
Computer scientists develop algorithms and intelligent systems that solve complex problems, often using concepts from machine learning and AI. Their work is crucial in automating tasks, optimizing processes, and creating innovative solutions across various domains. This project, for example, showcases the application of reinforcement learning in game AI development, demonstrating how intelligent systems can be trained to perform specific tasks efficiently.

#### How Do I Approach a Problem as a Computer Scientist?
Approaching a problem as a computer scientist involves breaking down the problem, understanding the requirements, and methodically designing and implementing a solution. This often includes iterative testing and optimization to achieve the desired outcome. For this project, I started by understanding the environment and the provided code, then implemented and tested the Q-learning algorithm to train the pirate agent.

#### What Are My Ethical Responsibilities to the End User and the Organization?
Ethically, computer scientists must ensure their solutions are fair, transparent, and beneficial to end users and organizations. They must consider privacy, security, and the potential impact of their algorithms on society. In this project, ensuring that the intelligent agent behaves as expected without exploiting any unintended behaviors is crucial for maintaining the integrity of the game and providing a fair experience to the players.

## Implementation Details

### Q-Learning Algorithm
The deep Q-learning implementation used a neural network to approximate the Q-value function. The network consisted of input layers, hidden layers with ReLU activation, and output layers corresponding to possible actions. The agent learned optimal actions through a balance of exploitation (using known information) and exploration (trying new actions).

```python
# Neural network model setup
model = Sequential()
model.add(Dense(24, input_dim=state_size, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(action_size, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

# Q-learning training loop
def q_learning_train(maze, agent, episodes, gamma, epsilon):
    for episode in range(episodes):
        state = maze.reset()
        done = False
        while not done:
            action = choose_action(state, epsilon)
            next_state, reward, done = maze.step(action)
            target = reward + gamma * np.max(agent.predict(next_state))
            q_values = agent.predict(state)
            q_values[0][action] = target
            agent.fit(state, q_values, epochs=1, verbose=0)
            state = next_state
```

### Design Defense
In the design defense, I compared human problem-solving approaches with machine learning techniques, assessed the purpose of the intelligent agent in pathfinding, and evaluated the use of algorithms to solve complex problems. [View the Design Defense Document](https://github.com/EmpressCatbug/CS370Final-/blob/main/CS%20370%20Project%202%20Design%20Defense%20Thompson.pdf)

## Conclusion
This project provided valuable experience in applying reinforcement learning and neural networks to a real-world problem. The skills and concepts learned here are essential for developing intelligent systems and contribute significantly to my understanding of AI and its applications in computer science.
