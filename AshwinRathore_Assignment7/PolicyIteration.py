import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import random
gamma = 1 # discounting rate
rewardSize = -1
gridSize = 5
terminationStates = [[0,0], [gridSize-1, gridSize-1]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
numIterations = 100000

def actionRewardFunction(initialPosition, action):
    if initialPosition in terminationStates:
        return initialPosition, 0
    reward = rewardSize
    finalPosition = np.array(initialPosition) + np.array(action)
    if -1 in finalPosition or 5 in finalPosition: 
        finalPosition = initialPosition   
    return finalPosition, reward

valueMap = np.zeros((gridSize, gridSize))
valueMap=np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
          [5.0, 6.0, 7.0, 8.0, 9.0],
          [10.0, 11.0, 12.0, 13.0, 14.0],
          [15.0, 16.0, 17.0, 18.0, 19.0],
          [20.0, 21.0, 22.0, 23.0, 0.0]])
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
deltas = []
for it in range(1000000):
    copyValueMap = np.copy(valueMap)
    deltaState = []
    for state in states:
        if state ==[0,0] or state ==[4,4]:
            continue
        weightedRewards = 0
        for action in actions:
            finalPosition, reward = actionRewardFunction(state, action)
            weightedRewards += (1/len(actions))*(reward+(gamma*valueMap[finalPosition[0], finalPosition[1]]))
        deltaState.append(np.abs(copyValueMap[state[0], state[1]]-weightedRewards))
        copyValueMap[state[0], state[1]] = weightedRewards-1
    deltas.append(deltaState)
    if (np.array_equal(valueMap, copyValueMap)):
        print("Iteration {}".format(it+1))
        print(valueMap)
        print("")
        break
    valueMap = copyValueMap
    if it in [0,1,9, numIterations-1]:
        print("Iteration {}".format(it+1))
        print(valueMap)
        print("")








