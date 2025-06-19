# 224R-Final-Project

## Follow instructions here to set up MiniWoB++: https://miniwob.farama.org/content/getting_started/
Create a python environment and then install required libraries within this environment. 

There are two files under q-learning folder. These are our initial milestone exploration files.
1. `simple_q_learning.py`: We run Q-learning on one of the following MiniWoB environments whose one action can be expressed as 'click one DOM element':
* miniwob/click-test
* miniwob/click-button
* miniwob/click-button-sequence
* miniwob/click-link

2. `sequential_q_learning.py`: We run Q-learning on one of the following MiniWoB environments:
* miniwob/click-test
* miniwob/click-button
* miniwob/click-button-sequence
* miniwob/click-link

Based on the feedback, we decided to expand the number of distinct tasks (more states/action spaces explored):
Click-test
Click-button
Click-button-sequence
focus-text
click-collapsible
Click-dialog
Login-user
Simple-arithmetic
Enter-text
Click-tab

Final approaches:
We decided to implement HER with the Q learning approach we explored for the milestone in `her_q_learning.py` (Pure RL).

RL for LLMs exploration
1. `gpt40_miniwob_agent.py` (Pure LLM)
2. `hrl_gpt4o_agent.py` (RL + LLM)


