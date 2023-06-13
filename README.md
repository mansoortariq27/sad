start here
names='''
help
outliers
pca
one_hot
label
perceptron_learning
mlp
kmeanM
kmeanS
maze
processing
mlpi
mlp_bp
ga
libraries
regressionS
'''
libraries='''
from sklearn import cluster, datasets, mixture
import cv2
from PIL import Image
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_blobs
import os
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.neural_network import MLPClassifier
'''
ga='''
#Function to generate a chromosome of length n
def generate_chromosome(n):
    chromosome = []
    for i in range(n):
        chromosome.append(random.randint(0, n-1))
    return chromosome
#Function to generate an initial population of size pop_size, where each chromosome has length n
def initial_population(pop_size, n):
    population = []
    for i in range(pop_size):
        chromosome = generate_chromosome(n)
        population.append(chromosome)
    return population
#Function to compute the fitness score of a given chromosome
def fitness(chromosome):
    n = len(chromosome)
    attacks = 0
    for i in range(n):
        for j in range(i+1, n):
            if chromosome[i] == chromosome[j]:  # Check for row conflicts
                attacks += 1
            elif abs(chromosome[i] - chromosome[j]) == abs(i - j):  # Check for diagonal conflicts
                attacks += 1
    fitness_score = n*(n-1)/2 - attacks  # Calculate fitness score based on number of conflicts
    return fitness_score
#Function to perform tournament selection on a given population
def selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)  # Choose tournament_size chromosomes at random
    fittest = tournament[0]
    for i in range(1, tournament_size):  # Iterate over remaining chromosomes to find fittest
        if fitness(tournament[i]) > fitness(fittest):
            fittest = tournament[i]
    return fittest
#Function to perform crossover between two parent chromosomes
def crossover(parent1, parent2):
    n = len(parent1)
    crossover_point = random.randint(1, n-1)  # Choose a random crossover point
    child = parent1[:crossover_point] + parent2[crossover_point:]  # Create child by combining parent segments
    return child
#Function to perform mutation on a given chromosome with a given probability
def mutation(chromosome, mutation_rate):
    n = len(chromosome)
    for i in range(n):
        if random.random() < mutation_rate:  # Perform mutation with probability mutation_rate
            chromosome[i] = random.randint(0, n-1)  # Choose a new random value for the gene
    return chromosome
#Function to convert a chromosome to a string representation of a chessboard
def chromosome_to_board(chromosome):
    n = len(chromosome)
    board = []
    for i in range(n):
        row = ["x"] * n
        row[chromosome[i]] = "Q"  # Place a queen at the position specified by the gene
        board.append(" ".join(row))
    return "
".join(board)

#Function to perform a genetic algorithm to solve the N-Queens problem
def genetic_algorithm(pop_size, n, tournament_size, mutation_rate, num_generations):
    population = initial_population(pop_size, n)  # Initialize population
    for i in range(num_generations):
        fitness_scores = [fitness(chromosome) for chromosome in population]  # Compute fitness scores for all chromosomes
        best_fitness = max(fitness_scores)  # Find the fittest chromosome in the population
        best_chromosome = population[fitness_scores.index(best_fitness)]
        print(f"Generation {i+1}: Best fitness = {best_fitness}")  # Print progress
        print(chromosome_to_board(best_chromosome))
        new_population = []
        for j in range(pop_size):
            parent1 = selection(population, tournament_size)  # Select two parents via tournament selection
            parent2 = selection(population, tournament_size)
            child = crossover(parent1, parent2)
            child = mutation(child, mutation_rate)
            new_population.append(child)
        population = new_population
    best_board = chromosome_to_board(best_chromosome)
    return best_chromosome, best_board

n=int(input('Enter queen size : '))
best_solution, best_board = genetic_algorithm(pop_size=50, n=n, tournament_size=5, mutation_rate=0.05, num_generations=100)

print("Best solution found:")
print(best_solution)
print("Corresponding board:")
print(best_board)
'''
mlp_bp='''
import numpy as np

#Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#Activation functions and their derivatives
def sigmoid_derivative(x):
    return x * (1 - x)

#Hyperparameters
input_nodes = 60
hidden_nodes = 32
output_nodes = 1
learning_rate = 0.01
epochs = 500

#Initialize weights and biases
W1 = np.random.randn(input_nodes, hidden_nodes)
b1 = np.zeros(hidden_nodes)

W2 = np.random.randn(hidden_nodes, output_nodes)
b2 = np.zeros(output_nodes)

#Training loop
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(x, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    output_layer_output = sigmoid(output_layer_input)

    # Backward propagation
    output_error = y - output_layer_output
    output_delta = output_error * sigmoid_derivative(output_layer_output)

    hidden_error = np.dot(output_delta, W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    W2 += np.dot(hidden_layer_output.T, output_delta) * learning_rate
    b2 += np.sum(output_delta, axis=0) * learning_rate

    W1 += np.dot(x.T, hidden_delta) * learning_rate
    b1 += np.sum(hidden_delta, axis=0) * learning_rate

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        loss = np.mean(np.square(output_error))
        print(f"Epoch {epoch}, Loss: {loss:.5f}")
'''
mlpi='''

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

#Load dataset
data_dir = "/content/drive/MyDrive/Colab Notebooks/AI/flowers"
categories = os.listdir(data_dir)
labels = [i for i in range(len(categories))]
label_dict = dict(zip(categories, labels))

img_size = 128
data = []
target = []

for category in categories:
    folder_path = os.path.join(data_dir, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        try:
            # Convert the image into grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize the grayscale image
            resized = cv2.resize(gray, (img_size, img_size))

            # Append the resized image and its label to the data and target lists
            data.append(resized)
            target.append(label_dict[category])

        except Exception as e:
            print('Exception:', e)

#Normalize the data
data = np.array(data) / 255.0

#Reshape the data
data = data.reshape(-1, img_size, img_size, 1)

#Convert the target list to a numpy array
target = np.array(target)

#Split the data into training, validation, and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data, target, test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.1, random_state=42)

#Define the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(img_size, img_size, 1)),
    keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)),
    keras.layers.Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)),
    keras.layers.Dense(len(categories), activation='softmax', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01))
])


#Define bias initializer
bias_initializer = tf.keras.initializers.Zeros()

#Set weights manually for the first layer
W1 = np.random.randn(img_size * img_size, 64) * 0.01
b1 = bias_initializer(shape=(64,))
model.layers[1].set_weights([W1, b1])

#Set weights manually for the second layer
W2 = np.random.randn(64, 32) * 0.01
b2 = bias_initializer(shape=(32,))
model.layers[2].set_weights([W2, b2])

#Set weights manually for the output layer
W3 = np.random.randn(32, len(categories)) * 0.01
b3 = bias_initializer(shape=(len(categories),))
model.layers[3].set_weights([W3, b3])

#Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Print model summary
model.summary()

#Take a sample input from the training data
sample_input = train_data[0]

#Flatten the input
flatten_input = sample_input.flatten()

#Pass the input through the first layer manually
layer1_output = np.dot(flatten_input, W1) + b1
layer1_activation = np.maximum(layer1_output, 0)

#Pass the output of the first layer through the second layer manually
layer2_output = np.dot(layer1_activation, W2) + b2
layer2_activation = np.maximum(layer2_output, 0)

#Pass the output of the second layer through the output layer manually
output = np.dot(layer2_activation, W3) + b3
predicted_class = np.argmax(output)


#Train the model
epochs = 10
batch_size = 32
learning_rate = 0.01

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for i in range(0, len(train_data), batch_size):
        batch_data = train_data[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = model(batch_data)
            loss = tf.keras.losses.sparse_categorical_crossentropy(batch_labels, predictions)
        # Backward pass
        gradients = tape.gradient(loss, model.trainable_variables)
        # Update weights and biases using stochastic gradient descent
        for j in range(len(model.trainable_variables)):
            model.trainable_variables[j].assign_sub(learning_rate * gradients[j])
    # Evaluate on validation data
    val_loss, val_accuracy = model.evaluate(val_data, val_labels)
    print(f"Val loss: {val_loss:.4f} - Val accuracy: {val_accuracy:.4f}")


#Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print(f"Test loss: {test_loss:.4f} - Test accuracy: {test_accuracy:.4f}")

#Predict the class labels for the test set
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)

#Calculate the classification report
from sklearn.metrics import classification_report

print(classification_report(test_labels, y_pred_classes, target_names=categories))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#Get the predicted probabilities for the test set
y_pred_prob = model.predict(test_data)

#Plot the ROC curve and calculate AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(categories)):
    fpr[i], tpr[i], _ = roc_curve(test_labels, y_pred_prob[:,i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

#Plot the ROC curves for each class
plt.figure(figsize=(8,6))
for i in range(len(categories)):
    plt.plot(fpr[i], tpr[i], label=f"{categories[i]} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend()
plt.show()
'''
help='''
import re
def give(x):
    for i in x:
        i=re.sub(r'<.*>','',i)
        i=i.replace('&lt;','<')
        i=i.replace('&gt;','>')
        print(i)
give(dt['help'].split('\n'))
'''
outliers='''

  #check for outliers using iqr
  Q1 = df.quantile(0.25)
  Q3 = df.quantile(0.75)
  IQR = Q3 - Q1
  print(IQR)
  #remove outliers
  df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
  df

  #check for outliers using z-score
  from scipy import stats
  import numpy as np
  z = np.abs(stats.zscore(df))
  #remove outliers
  df = df[(z < 3).all(axis=1)]
  df

  #check for outliers using boxplot
  import seaborn as sns
  sns.boxplot(x=df['Age'])
 
'''
pca='''

  #get the numeric columns
  df_numeric = df.select_dtypes(include=['float64', 'int64'])
  #pca
  from sklearn.decomposition import PCA
  pca = PCA(n_components=2)
  pca.fit(df_numeric)
  pca_data = pca.transform(df_numeric)
  pca_data

 
'''
one_hot='''

 #get the categorical columns
  df_categorical = df.select_dtypes(include=['object'])

  #numeric columns
  df_numeric = df.select_dtypes(include=['float64', 'int64'])

  #one hot encoding
  df_dummies = pd.get_dummies(df_categorical, drop_first=True)

  #concatenate the two dataframes : df_dummies, df_numeric
  df_new = pd.concat([df_dummies, df_numeric], axis=1)
'''
label='''

  #get the numeric columns
  df_numeric = df.select_dtypes(include=['float64', 'int64'])

  #get the categorical columns
  df_categorical = df.select_dtypes(include=['object'])

  #label encoding
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  df_categorical = df_categorical.apply(le.fit_transform)

  #concatenate the two dataframes : df_dummies, df_numeric
  df_new = pd.concat([df_dummies, df_numeric], axis=1)
'''
betaG='''

#Tic Tac Toe Player
  import copy
  import math
  import random


  X = "X"
  O = "O"
  D="D"
  EMPTY = None


  def initial_state():
      """
      Returns starting state of the board.
      """
      return [[EMPTY, EMPTY, EMPTY],
              [EMPTY, EMPTY, EMPTY],
              [EMPTY, EMPTY, EMPTY]]
     
  def draw_board(board):
      print("-------------")
      for i in range(3):
          print("| ", end="")
          for j in range(3):
              if board[i][j] == EMPTY:
                  print(" ", end="")
              else:
                  print(board[i][j], end="")
              print(" | ", end="")
          print()
          print("-------------")


  def player(board):
      """
      Returns player who has the next turn on a board.
      """
      count = 0
      for i in board:
          for j in i:
              if j:
                  count += 1
      if count % 2 != 0:
          return O
      return X


  def actions(board):
      """
      Returns set of all possible actions (i, j) available on the board.
      """
      res = set()
      board_len = len(board)
      for i in range(board_len):
          for j in range(board_len):
              if board[i][j] == EMPTY:
                  res.add((i, j))
      return res


  def result(board, action):
      """
      Returns the board that results from making move (i, j) on the board.
      """
      curr_player = player(board)
      result_board = copy.deepcopy(board)
      (i, j) = action
      result_board[i][j] = curr_player
      return result_board


  def get_horizontal_winner(board):
      # check horizontally
      winner_val = None
      board_len = len(board)
      for i in range(board_len):
          winner_val = board[i][0]
          for j in range(board_len):
              if board[i][j] != winner_val:
                  winner_val = None
          if winner_val:
              return winner_val
      return winner_val


  def get_vertical_winner(board):
      # check vertically
      winner_val = None
      board_len = len(board)
      for i in range(board_len):
          winner_val = board[0][i]
          for j in range(board_len):
              if board[j][i] != winner_val:
                  winner_val = None
          if winner_val:
              return winner_val
      return winner_val


  def get_diagonal_winner(board):
      # check diagonally
      winner_val = None
      board_len = len(board)
      winner_val = board[0][0]
      for i in range(board_len):
          if board[i][i] != winner_val:
              winner_val = None
      if winner_val:
          return winner_val

      winner_val = board[0][board_len - 1]
      for i in range(board_len):
          j = board_len - 1 - i
          if board[i][j] != winner_val:
              winner_val = None

      return winner_val


  def winner(board):
      """
      Returns the winner of the game, if there is one.
      """
      winner_val = get_horizontal_winner(board) or get_vertical_winner(board) or get_diagonal_winner(board) or None
      return winner_val


  def terminal(board):
      """
      Returns True if game is over, False otherwise.
      """
      if winner(board) != None:
          return True

      for i in board:
          for j in i:
              if j == EMPTY:
                  return False
      return True

  def utility(board):
      """
      Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
      """
      winner_val = winner(board)
      if winner_val == X:
          return 1
      elif winner_val == O:
          return -1
      return 0
         
  def play_game():
      board = initial_state()
      draw_board(board)
      while not terminal(board):
          print("Enter the row and column number")
          row, col = map(int, input().split())
          board = result(board, (row, col))
          if not terminal(board):
              action = minimax(board)
              board = result(board, action)
              draw_board(board)
      print("Winner is ", winner(board))

  #Return the maximum value of the board and the action that leads to it
  def max_value(board):
      if terminal(board):
          return utility(board), None
      v = -math.inf
      action = None
      for a in actions(board):
          v1 = min_value(result(board, a))[0]
          if v1 > v:
              v = v1
              action = a
      return v, action

  #Return the minimum value of the board and the action that leads to it
  def min_value(board):
      if terminal(board):
          return utility(board), None
      v = math.inf
      action = None
      for a in actions(board):
          v1 = max_value(result(board, a))[0]
          if v1 < v:
              v = v1
              action = a
      return v, action

  #Return the optimal action for the current player on the board
  def minimax(board):
     
      if terminal(board):
          return None
      if player(board) == X:
          return max_value(board)[1]
      else:
          return min_value(board)[1]

  play_game()
 
'''
maze='''

  #read the maze from the input file
maze = []
#read the maze
def read_maze():

    #open the input file
    with open('maze.txt') as f:

        #iterate through the lines in the file
        for line in f:

            #create a row to store the values in the line  
            row = []

            #iterate through the characters in the line
            for i in line.strip():
                if i == 'S':
                    row.append('S')
                elif i == 'G':
                    row.append('G')
                elif i == '0':
                    row.append(0)
                elif i == '1':
                    row.append(1)
                elif i=='2':
                    row.append(2)
                elif i==' ':
                    continue
                else:
                    raise ValueError('Invalid character in input file')
            maze.append(row)

        return maze

maze=read_maze()

#this fucntion takes maze as input and then returns the start and goal positions
def identify_start_goal(maze):

    #iterate thorugh the maze and find the start and goal positions
    for i in range(len(maze)):
        for j in range(len(maze[i])):

            #check if the current position is the start position
            if maze[i][j] == 'S':
                s = (i, j)
            elif maze[i][j] == 'G':   #check if the current position is the goal position
                e = (i, j)
    return s, e

#s and e are the start and goal positions
s,e =  identify_start_goal(maze)

 
'''
bfs='''


 #implement BFS recursively
from collections import deque

#this is a recursive function that takes maze, start position, goal position, queue, visited set and path as input
#return the path if the goal position is found else return None

#implement BFS recursively
def bfs(maze, S, G, queue=None, visited=None, path=None):

    #check if the queue is empty
    if not queue:

        #create a queue and add the start position to it
        queue = deque([(S, [])])
   
    #check if the visited set is empty
    if not visited:
        #create a visited set and add the start position to it
        visited = set([S])

    #check if the path is empty
    if not path:

        #create a path list
        path = []
       
    #pop the first element from the queue from the left
    curr_position, path = queue.popleft()

    #store the current indices in i and j
    i, j = curr_position

    #check if the current position is the goal position
    if curr_position == G:
        return path + [curr_position]
    for ni, nj in [(i-1, j), (i, j+1), (i+1, j), (i, j-1)]:
        #check if the neighbour is valid and not visited
        if 0 <= ni < len(maze) and 0 <= nj < len(maze[0]) and maze[ni][nj] != 1 and (ni, nj) not in visited:

            #add the neighbour to the queue
            queue.append(((ni, nj), path + [curr_position]))

            #add the neighbour to the visited set
            visited.add((ni, nj))

    #check if the queue is not empty
    if queue:
        return bfs(maze, S, G, queue, visited, path)
    else:
        return None


#check if bfs returns a path
def check_bfs_path(maze, s, e):

    #call the bfs function
    path=bfs(maze, s, e)

    #check if the path is not None
    if(path):

        #return the path
        return path
    else:
        return -1

        # run BFS and print the path
  bfs_path = bfs(maze, s, e)

  #check if bfs returns a path
  if bfs_path:
      print('BFS path :', check_bfs_path(maze, s, e))
  else:
      print('BFS found no path')

'''
dfs='''

#implement DFS recursively
def dfs_recursive(maze, curr_position, G, visited, path):

    #store the current indices in i and j
    i, j = curr_position

    #add the current to the visited set
    visited.add(curr_position)

    #check if the current position is the goal position
    if curr_position == G:

        # add the current node to the path and return the path
        return path + [curr_position]
   
    #check the neighbours of the current node
    for ni, nj in [(i+1, j), (i, j-1),(i-1, j), (i, j+1)]:

        #check if the neighbour is valid and not visited
        if 0 <= ni < len(maze) and 0 <= nj < len(maze[0]) and maze[ni][nj] != 1 and (ni, nj) not in visited:

            #call the dfs function recursively with the neighbour as the current node
            result = dfs_recursive(maze, (ni, nj), G, visited, path + [curr_position])

            #check if the result is not None
            if result:
                return result
               
    return None

#wrapper function
def dfs(maze, S, G):
    visited = set()
    return dfs_recursive(maze, S, G, visited, [])

#check if dfs returns a path
def check_dfs_path(maze, s, e):

    #call the dfs function
    path=dfs(maze, s, e)

    #check if the path is not None
    if path:
        return path
    else:
        return -1


      #run DFS and print the path
dfs_path = dfs(maze, s, e)
if dfs_path:
    print('DFS path :', check_dfs_path(maze, s, e))
else:
    print('DFS found no path')
 
'''
bfs_dfs_visualization='''

import tkinter as tk

def main2():
    #calling the dfs function
    path = check_dfs_path(maze, s, e)

    #calling the bfs function
    path2 = check_bfs_path(maze, s, e)

    # Define the colors we will use in RGB format
    BLACK = "#000000"
    WHITE = "#FFFFFF"
    RED = "#FF0000"
    GREEN = "#00FF00"
    BLUE = "#0000FF"
    PINK = "#FF00FF"
    YELLOW = "#FFFF00"

    # Define the size of each cell in the maze
    CELL_SIZE = 20

    def draw_maze(maze):
        for row in range(len(maze)):
            for col in range(len(maze[row])):
                if maze[row][col] == 1:
                    canvas.create_rectangle(col * CELL_SIZE, row * CELL_SIZE, (col + 1) * CELL_SIZE, (row + 1) * CELL_SIZE, fill=BLACK)
                else:
                    canvas.create_rectangle(col * CELL_SIZE, row * CELL_SIZE, (col + 1) * CELL_SIZE, (row + 1) * CELL_SIZE, fill=WHITE)
   
    def draw_path(path, path2):
        # draw the path for dfs
        for i in range(len(path)-1):
            row1, col1 = path[i]
            row2, col2 = path[i+1]
            canvas.create_line(col1 * CELL_SIZE + CELL_SIZE/2, row1 * CELL_SIZE + CELL_SIZE/2, col2 * CELL_SIZE + CELL_SIZE/2, row2 * CELL_SIZE + CELL_SIZE/2, fill=GREEN, width=2)

        # draw the path for bfs
        for i in range(len(path2)-1):
            row1, col1 = path2[i]
            row2, col2 = path2[i+1]
            canvas.create_line(col1 * CELL_SIZE + CELL_SIZE/2, row1 * CELL_SIZE + CELL_SIZE/2, col2 * CELL_SIZE + CELL_SIZE/2, row2 * CELL_SIZE + CELL_SIZE/2, fill=RED, width=2)

        # draw the start and goal nodes
        canvas.create_rectangle(0, 0, CELL_SIZE, CELL_SIZE, fill=PINK)
        canvas.create_rectangle(len(maze[0]) * CELL_SIZE - CELL_SIZE, len(maze) * CELL_SIZE - CELL_SIZE, len(maze[0]) * CELL_SIZE, len(maze) * CELL_SIZE, fill=YELLOW)

    # Create the main window and canvas
    root = tk.Tk()
    canvas = tk.Canvas(root, width=len(maze[0]) * CELL_SIZE, height=len(maze) * CELL_SIZE)
    canvas.pack()

    # Draw the maze and path
    draw_maze(maze)
    if path != -1:
        draw_path(path, path2)

    root.mainloop()

main2()

'''
kmeanM='''

  #find the optimal number of clusters using elbow method
  from sklearn.cluster import KMeans
  wcss = []
  for i in range(1, 11):
      kmeans = KMeans(n_clusters=i)
      kmeans.fit(X1)
      wcss.append(kmeans.inertia_)
  import matplotlib.pyplot as plt
  plt.plot(range(1, 11), wcss)
  plt.title('The Elbow Method')
  plt.xlabel('Number of clusters')
  plt.ylabel('WCSS')
  plt.show()

  #FIND THE OPTIMAL NUMBER OF CLUSTERS USING silhouette_score
  from sklearn.metrics import silhouette_score
  from sklearn.cluster import KMeans
  import matplotlib.pyplot as plt
  sil = []
  for k in range(2,6):
    kmeans = KMeans(n_clusters = k).fit(X1)
    labels = kmeans.labels_
    sil.append(silhouette_score(X1, labels, metric = 'euclidean'))
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_score(X1, labels, metric = 'euclidean'))
  plt.plot(range(2, 6), sil)
  plt.title('KMeans')
  plt.xlabel('Number of clusters')
  plt.ylabel('Silhouette Score')
  plt.show()

  #apply kmeans algorithm
  kmeans = KMeans(n_clusters=3)
  kmeans.fit(X1)
  y_kmeans = kmeans.predict(X1)

  #visualize the clusters
  plt.scatter(X1[:, 0], X1[:, 1], c=y_kmeans, s=50, cmap='viridis')
  centers = kmeans.cluster_centers_
  plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
  plt.show()
 
'''
regressionS='''
  #Create a target column by averaging the low + high. Drop only open, close columns.
  df['target'] = (df['Low'] + df['High'])/2
  df.drop(['Open','Close'],axis=1,inplace=True)
  Preprocess the date column. Convert it into long format.
  df['Date'] = pd.to_datetime(df['Date'])
  df['Date'] = df['Date'].astype(np.int64)
  #converting categorical data into numerical data
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  df['Symbol']=le.fit_transform(df['Symbol'])
  df.drop(['Name'],axis=1,inplace=True)
  df.head()
  #Apply Linear Regression and find the error both mean absolute and mean squared error.
  X = df.drop('target',axis=1)
  y = df['target']
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
  model = LinearRegression()
  model.fit(X_train,y_train)
  y_pred = model.predict(X_test)
  #4-Before applying machine learning divide the data into training and testing part with 70-30 aspect ratio.
  print('Mean Absolute Error: ',mean_absolute_error(y_test,y_pred))
  print('Mean Squared Error: ',mean_squared_error(y_test,y_pred))
  #Draw the linear regression line estimated by Model.
  import matplotlib.pyplot as plt
  plt.scatter(y_test,y_pred)
  plt.xlabel('Actual')
  plt.ylabel('Predicted')
  plt.show()
'''
kmeanS='''

  #Generate a synthetic dataset with 300 samples and 2 features
  data = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
  #Standardize the dataset
  scaler = StandardScaler()
  data_scaled = scaler.fit_transform(data[0])

 
class KMeans:
    def __init__(self, clusters=5, iterations=100):
        #number of clusters
        self.clusters = clusters
        #MAx Iterations
        self.iterations = iterations
    def distances(self, X):
        #calculating distance
        distance = np.linalg.norm(X[:, np.newaxis] - self.centroid, axis=2)
        return distance
    def centroids(self, X, labels):
        #updating centroid
        centroid = np.array([X[labels == k].mean(axis=0) for k in range(self.clusters)])
        return centroid
    def fit(self, X):
        self.centroid = X[np.random.choice(X.shape[0], self.clusters, replace=False)]
        iter=range(self.iterations)
        for i in iter:
            #getting distance calculated above
            distance = self.distances(X)
            #giving label on basis of distance
            labels = np.argmin(distance, axis=1)
            #now centroids are updayted based on labels
            new_cntroid = self.centroids(X, labels)
            #checking convergence
            if np.linalg.norm(new_cntroid - self.centroid) < 1e-4:
                break
            #updating on the basis of covergence
            self.centroid = new_cntroid
        self.labels_ = labels
  #Taking 3 clusters
  km = KMeans(clusters=3)
  km.fit(data_scaled)
  #Plot clusters
  plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=km.labels_, s=10)
  plt.scatter(km.centroid[:, 0], km.centroid[:, 1], s=100, marker='*', c='red')
  plt.title('K-Mean Clustering')
  plt.xlabel('X axis')
  plt.ylabel('Y axis')
  plt.show() 
'''
mlp='''

  from sklearn.neural_network import MLPClassifier
  from sklearn.datasets import make_classification
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score

  #Generate a random dataset for demonstration purposes
  X, y = make_classification(n_samples=100, n_features=10, n_classes=2)
  #Split the dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :62], df.iloc[:, -1], test_size=0.3)
  #Create an instance of the MLPClassifier with default parameters
  mlp = MLPClassifier()
  #Define some hyperparameters to tune
  hidden_layer_sizes = [(10,), (50,), (100,)]
  learning_rates = ['constant', 'invscaling', 'adaptive']
  #Train and evaluate the MLPClassifier on different hyperparameter combinations
  best_accuracy = 0
  best_mlp = None
  for layer_size in hidden_layer_sizes:
      for rate in learning_rates:
          mlp.set_params(hidden_layer_sizes=layer_size, learning_rate=rate)
          mlp.fit(X_train, y_train)
          y_pred = mlp.predict(X_test)
          accuracy = accuracy_score(y_test, y_pred)
          if accuracy > best_accuracy:
              best_accuracy = accuracy
              best_mlp = mlp
  #Predict the class labels of the test set using the best MLPClassifier
  y_pred = best_mlp.predict(X_test)
  #Calculate the accuracy of the classifier
  accuracy = accuracy_score(y_test, y_pred)
  print("Best MLPClassifier:", best_mlp)
  print("Accuracy:", accuracy)
'''
perceptron_learning='''
  #Implement OR and AND gates using perceptron learning scheme.
  #OR GAte
  #input
  X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  #output
  Y = np.array([0, 1, 1, 1])
  input_len = len(X)
  w = np.array([1, 1])
  lr = 0.1
  b = 0.5
  n = 100
  for i in range(n):
      for j in range(input_len):
          O = np.dot(X[j], w) + b
          if O >= 0:
              t = 1
          else:
              t = 0
          error = Y[j] - t
          w = w + (lr * error * X[j])
          b = b + (lr * error)
  #Print the new weights and bias
  print("New Weights:", w)
  print("New bias:", b)

  #Calculate the actual output using the updated weights and bias
  actual_output = np.zeros(input_len)
  for j in range(input_len):
      O = np.dot(X[j], w) + b
      if O >= 0:
          actual_output[j] = 1

  #Print the expected output and the actual output side by side
  print("Expected Output: ", Y)
  print("Actual Output:", actual_output) 
'''
processing='''
  #find the missing values
  df.isnull().sum()

  #if there is ? in the data, replace it with NaN
  import numpy as np
  df = df.replace('?', np.NaN)

  #drop the missing values
  df = df.dropna()

  #fill the missing values with the mean of the column
  df = df.fillna(df.mean())

  #fill the missing values with the median of the column
  df = df.fillna(df.median())

  #fill the missing values with the mode of the column
  df = df.fillna(df.mode())

  #fill the missing values with the constant of the column
  df = df.fillna(0)

  #fill the categorical missing values with the mode of the column
  df = df.fillna(df.mode().iloc[0])
#Clean the dataset.
  df=df.drop(['price'], axis=1)
#Removing zero rows
remove_all_zero_rows=df.iloc[:, 2:62].dropna(how='all')
remove_all_zero_rows
idx = df.index
#Remove rows with corresponding indices from columns 1, 2, and 64
df = df.drop(index=idx.difference(remove_all_zero_rows.index))
#Fill missing values with column means
df = df.fillna(df.mean())
#Update the original dataframe with the cleaned subset
df
#3. Convert labels into 0 or 1
df.loc[df['calc_position'] ==1, 'calc_position'] = 1
df.loc[df['calc_position'] >1, 'calc_position'] = 0
df
#4. Plot statistics to perform exploratory data analysis.
df.describe()
df.info()
#Histogram of a numerical column
plt.hist(df.iloc[:, 2:])
plt.title('Histogram of column_name')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

#Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

race_id_ = pd.get_dummies(df, columns = ['race_id'])
race_id_

df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].astype(np.int64)

df['target'] = (df['Low'] + df['High'])/2
df.drop(['Open','Close'],axis=1,inplace=True)

#converting categorical data into numerical data

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Symbol']=le.fit_transform(df['Symbol'])

df.drop(['Name'],axis=1,inplace=True)

df.head()

plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
'''
end here
