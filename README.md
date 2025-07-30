# SoC_2025_SuperMario_RL
🕹️ Super Mario! Quest Using Reinforcement Learning

Project Name: Super Mario! Quest Using RL
Project ID: 75
Name: Arnav Pandit
Roll No.: 24B3948

Welcome to my SoC (Summer of Code) project repository! This project explores the use of Reinforcement Learning (RL) to teach an AI agent how to play the classic Super Mario Bros. game.

Here is how to navigate through this repository:

Week 2: Introduction to Reinforcement Learning with Taxi-v3

    Implemented and trained an agent using Q-Learning on the Taxi-v3 environment from OpenAI Gym.

    Learned fundamental RL concepts like state, action, reward, Q-table, and policy iteration.

    Demonstrated how an agent can learn to pick up and drop off passengers optimally.

Week 3 & 4: Exploring the Super Mario World

    Set up the gym-super-mario-bros environment.

    Applied wrappers for preprocessing (resizing, grayscale, frame stacking).

    Defined different action spaces (e.g., SIMPLE_MOVEMENT, COMPLEX_MOVEMENT).

    Explored the challenges of training in a visual and dynamic environment.

    Observed Mario's learning behavior and limitations using default settings.

Week 5 (Final): Mastering Mario with PPO

    Switched to Proximal Policy Optimization (PPO), a state-of-the-art policy-gradient algorithm.

    Used CnnPolicy to handle image-based input effectively.

    Optimized Mario’s performance using custom wrappers and tuned hyperparameters.

    Compared results between simple and complex movement strategies.

    Final trained model demonstrates Mario crossing difficult obstacles with learned jumping and movement patterns.

📌 Key Concepts Covered

    Q-Learning

    Deep Reinforcement Learning

    Environment Preprocessing (Gray-scaling, Frame Stacking)

    PPO Algorithm

    Action Space Design

    Reward Engineering


🧠 Tools & Libraries Used

    gym, gym-super-mario-bros

    stable-baselines3

    OpenAI Baselines

    NES-Py

    NumPy, Matplotlib for logging & visualization
