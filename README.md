**Problem 1 - Pikachu** - 
  **Problem Statement** 
 - 1. Given a state of the board, we need to make the best possible next move such that our player wins the game if it continues.
 - 2. The solution to this problem is to calculate the Minimax score for a given board state with the player given. If player is w, then it means w is treated as Max player and we need to find the best move for w. 
 - 3. Our Max node will look at all possible valid moves for player given,  from the given state of the board. 
 - 4. The next layer will be a Min, where the best possible moves by the other player will be calculated by taking the Min of the Max nodes below it(these can be thought of as all possible moves after the min player has responded to our move - that is for all responses to our move). Ideally, We need to keep building the game tree until we reach leaf nodes for all branches of the tree. We reach a leaf node, when we either win or lose the game, which in this game is given by all Max player's pieces being captured(-1) or all Min player's pieces being captured. 
 - 5. In practice, however, generating the whole game tree will take computationally long time. So, we do 2 things - **(1).** ***Alpha-beta Pruning*** to prune the nodes which we know will not have an effect on the decision made at a Min or Max node. **(2)** Measure the utility value of a given board state using an ***evaluation function***, so that we can stop early in our game tree and return this value instead of waiting to reach the terminal nodes and backing up those values. 
 
 **Implementation Details and challenges** - 
  - 1. We create 2 classes ***'pichu'*** and ***'Pikachu'*** for the type of pieces that can be present on the board. A class 'State' has the list of white pieces, black pieces and the board state. The properties of pichu and Pikachu are color(side), Character(text character on the board),position(row ,column) 
  - 2. When a board state is input, we convert it to 2d list and pass it to our function ***'GetNextMoves'***, tries to find all valid moves for each piece of the players color. We store each valid move with a ***Move*** class, which is used to denote the previous position of a piece, position after the move and captured piece position. Our ***UpdateBoard*** method takes care of changing the position of each piece and also removing and replacing captured and Promoted pieces. The pichu and Pikachu classes have a getvalid moves method which provides a list of ***Move*** class objects. This is checked for every piece to get all valid possible moves.  
  - 3. We built a recursive ***Minimax*** function, which took as input  - (State, player, alpha , beta and recursion depth). Our function returns 1 value representing the utility of 1 of the next states of our root node. This is returned for every next state of root node and the best node is picked. The function has with 3 if else code blocks. **1** to check the terminal state, which is recursion depth reached or no opponent pieces left **2** If passed level variable is 'Min' run the min code block with opponent player, and if level variable is passed with 'Max' then call Max part. We also implemented **alpha-beta pruning**, where we pass alpha and beta as -inf and +inf respectively. These values get set on the first node of the minimax tree computation. We prune subtrees by returning before computing utility values of all nodes in a subtree, when we know any node has utility value less than alpha, meaning that the min of all nodes in the subtree will have lesser or equal value to alpha, thus max node will not have any use of these nodes. Similarly, for beta values, any time we see a node in a subtree with higher value, we can return before getting utility values for other  nodes. 
  - 4. We also kept a ***recursion depth** variable to return utility values directly instead of recursing on minimax when the recursion depth reaches a certain value. We tried a few values of recursion depth, but with the time limit of 10 seconds, we are not able to get valid board state in 10 seconds if we keep recursion depth any higher than 3. Alpha-beta pruning, didnt seem to help us increase the recursion depth as we expected. 
  - 5. Due to the above step, we needed a strong **evaluation function**, such that the best score is given to a board state with the best chance of winning.
        - We ran our code against many other players to check the performance of our solution and on the basis of this, kept updating our evaluation function
        - We tried a combination of things **1.** Number of White Pieces  - Number of Black Pieces - the idea behind this was that anytime our player sees a capture 2 or 3 moves ahead it gives importance to that move and captures the opponent piece. **2** We defined a strength variable by looking at which variables have pieces 1 square behind, to the left and to the right of them. for e.g, if a piece has a same color piece behind it, then strength+=1, similarly for to the left of the piece and right of the piece - the idea being that a piece surrounded by same color pieces cannot be captured **3** For every piece we calculate how far the piece is from the centre of the board - the idea for this is that pieces on the edge are harder to capture as opposed to central ones, also having space in the centre files means our piece have room to capture opponent piece jumping to the left **4** Strength value is also increased by which type the piece is. If its a Pikachu, we add strength equal to length of board/2, since this piece can move the width and length of the board at max. **5** Sum of the row positions of the piece, so that pieces prefer to move forward, as thats the way to promote to a Pikachu and win quicker. 
         - Our final evaluation function after lots of experimenting was a combination(sum) of all of the above with **1** being most important for capturing opponent pieces, with a fraction multiplied to the other values.
**Challenges**
 - Due to the nature of the problem being a game, and the minimax recursive solution, it was very hard to debug coding errors. 
 - Due to the time limitation and ultimately reduction in recursive depth, we had to rely on improving the evaluation function. Our solution performed well vs basic opponents but not vs better functioning solutions. 
 - Alpha - beta pruning didnt seem to make the code faster
  - Lack of test cases provided meant we had to commit many times and check on server against other teams.
---------------------------------------------------------------------------------------------------------------------------------------------

**Problem 2 - Sebastian** - 
 - 1. Problem Statement - Given a die roll in 1 of 13 turns, we need to make a decision such that obtained score is maximized across all turns. The decision is made up of 2 parts
 **no. of dice to reroll** and which **subset of dice of chosen size** to reroll. 
 - 2. To obtain the best decision, we calculate the expected value of each decision and chose the one with the maximum expected value. This is done using Expectminimax but with no min layer since this is a 1 player game. Conceptually, our game tree starts with the given roll. **The root node layer is considered the 'MAX' layer**. We then produce a chance layer with one chance node representing 1 decision. for e.g, there will be a chance node for the decision - reroll die 1, chance node for decision reroll die 1,2, reroll 0, 2, 4, etc...
 - 3. Each chance node then predicts all possible outcomes for this decision given for its root. For eg., if first roll is 1, 2, 1, 2, 3,  then the decision to roll 1st die will produce the max nodes with outcomes - [1, 2, 1, 2, 3] , [2, 2, 1, 2, 3], [3, 2, 1, 2, 3], [4, 2, 1, 2, 3], [5, 2, 1, 2, 3], [6, 2, 1, 2, 3], that is the root node outcome with only 1st die outcome changed. Similarly another chance node will have the decision - reroll 2nd die. In general, **every chance node represents a decision**. The **first chance layer represents the expected values for the 2nd roll of the dice in the same turn***. Similarly the 2nd chance layer represents, the expected values for the 3rd roll of the dice in the same turn. 
 - 4. The outcomes from the chance node will also have a probability. The max nodes generated from the chance nodes each in turn generate more chance nodes. This process from step 1 - 4 above will happen until we hit a terminal or leaf level. In our game, since we have only 3 roll of dice, at the third max layer, we will stop generating chance nodes and return a value. The value returned by a max node in our game tree will be the highest score possible from that outcome(when its combined with original outcome - that is, if original roll is 1, 1, 1, 1, 4 and we reroll the 5th die, and get 1, 1, 1, 1, 1 then the value returned by this node will be 50 (**the highest category available**). **This value is multiplied by the probability of this outcome given the dice we rolled**. For same example, if we rolled only the 5th die and obtained 1, the probability of that outcome was 1/6, since only 1 die was rolled. Similarly, for 2 dice rolled and outcome of (1,2), the probability will be 1/18 (as 1,2 and 2, 1 are same combination
  - 5. The expected value of a chance node is the expected value of all its outcomes, that is, summation of (highest value possible from the max node multiplied by its probability). 
   - 6. The max node which is the parent of the chance nodes, will pick the chance node with the highest expected value. These values get backed up as the recursion values start returning and the max node at the root picks the decision which gives the maximum expected value. This value represents the strength of the decision for the 2nd roll of the dice, thus becoming our desired decision.  
    - 7. We will also have a chance node for the decision - dont reroll the dice again, since we could have already obtained the highest value possible. This chance node will produce 1 outcome(max node), the same as its root with probability as 1. The max node generated from this will, however, again generate all possible decisions. This represents the special case where a die is rolled 1st time, no roll chosen for 2nd attempt, but a different decision chosen for the 3rd roll ( this is in practice, a redundant node, since it will always pick the choice of not rolling on the 3rd level as well,if that was in fact the correct decision to begin with for the 2nd roll)
    - 8. Our game tree will be of the form -**max** -> chance1, chance2, chance3, chance4.... ->(each chance node)max1, max2, max3, max4->(each max node)chance1, chance2, chance3...->(each chance node)leaf1, leaf2,....

**Implementation Details** - 
 - 1. Our main function is ***Expectiminimax***. This function is called in every roll method - first_roll, second_roll(not in third, because there we need to make a decision about the category to choose). It takes as input  - **a**.current dice outcome - this is considered as root max node, **b**layer number/ roll number - this parameter is required since our terminal condition is that a roll is the third one in a turn.  **c** scorecard**, the current scorecard, this is required because in subsequent turns the no. of categories available will be lesser and lesser, so the computation of utility values in our max nodes will be from the highest score producing categories, **d** a string representing the type of the next layer - this is used to tell the ***if block*** condition in our expectiminimax definition to generate chance nodes. Similarly in the chance node, when expectiminimax is called, the value passed for this argument is 'max' so that the next layer is max layer.**e** rerolled_dice - this represents a list of dice which are to be rerolled - this parameter is required since our chance node will generate max nodes for only that particular decision -e.g the decision to reroll 1st die-> (max)all outcomes for this decision. In the first_roll method, this value is sent as blank, since it will be the chance nodes decide on the decisions. 
 - 2. We have a method called *PredictedRolls*. This method returns list of all the utility values along with their probabilities for the max node or leaf nodes that a chance node generates. This method is called in our ***if block*** for 'chance'. 
 - 3. The chance ***if block*** calculates the expected value of all the values returned from step 2 above by summing the returned probabilities with the **utility values** - these are calculated by sending the new predicted outcome of the dice to a ***Utility*** method. Once the chance nodes has geenerated a list of expected values of all decisions, it returns this list to the max node from which it was called. The max layer picks the highest value from that list and propagates it up to the chance node above it. If the max layer was a root node, then we are out of our recursion and our returned value is thus a value and a list of indexes of the dice to be rolled again(if any).
 - 4. In the first_roll method, we also compute the best category score of the original outcome. We compare the 2 values and pick the highest one. The list returned is this either empty - a decision to no roll again, or the list of indexes we obtained from ***Expectiminimax*** method. 
  - 5. ***Utility value computation*** - In the expectiminimax there is one ***if block*** which is true when the roll number is 3. This block calls the Utility function, passing an outcome of the 5 dice, obtained by some combination of rerolled dice and the other original dice. We also pass in the scorecard from the original call because while computing the score, we cannot considered the categories picked in the previous turns.  The utility function records scores for all categories and picks the highest score, since this is the value which will get us the best expectation at our chance node. Our utility function also checks whether the bonus flag has already been used, if not, it compares the sum of **(the 'Numbers' category of the original scorecard and the maximum value of the 'Numbers' category in the scorecard calculated by our utility function)**. If this number exceeds 63, we return the max score + 35. This is because, our current scorecard may have had a sum of the numbers category equal to some value, we check whether the best Numbers category score in this outcome will take the sum to 63. If it does, then we can expect a bonus from this outcome, thus adding 35 to the returned value. 
   - 6. **Probability and Score calculation** - Instead of computing a score everytime it is called, the utility function makes use of a dictionary ***ScoreMap***, which we generate in the constructor of SebastianAutoPlayer.py. Since we know the number of dice will always be 5 and number of outcomes from every single die is 1-6, we generate all possible permutations in a GenerateProbabilityMap function. The GenerateProbabilityMap function in turn calls ***GetProbability*** Method, which is a function which uses a formula from combinations to get the probability - **[(No. of dice rolled)!/product of - (no. of duplicate)!]* 1/(6 raised to no. of dice rolled)**. We return 2 dictionaries, a dictionary of scorecards for every outcome and a dictionary of probabilities for every outcome - ProbabilityMap and ScoreMap. In our chance node, when computing the expected value, we simply look up the outcome and its probability, that is, it 2 dice are rerolled and the outcome is (1, 2) - we look up the probability of the outcome to be 1/18 in our dictionary. Similarly we look up the scorecard in our Utility function. Thus, the Utility function will now have a scorecard for the outcome passed to it. However, we will set to 0, all the categories which have already been recorded on the original scorecard and consider the max value from the remaining available ones. These 2 dictionaries are done for speeding up the code. It did achieve some success in doing that(more on that below)

**Challenges and Design Decisions** - 
 - 1. Used Itertools and its methods for generating combinations instead of looping and replacing existing list elements to get new all possible combination of dice to be rolled.
 - 2. As mentioned above, used a precomputed table of probabilities. This sped up the computation time in our expectiminimax function, however, this map is called on every new Sebastian Game, even though the table is constant. 
 - 3. We do have an intuition that we have to optimize the scores not only across rolls but also across Turns. Since getting a low score on a potentially high scoring category, even though its the best category for that turn, means a lower overall score. However, we couldnt act on this inuition and write any good code in the Third_roll function, which just returns the max scoring category after the 3rd roll of the dice.
 - 4. Most importantly, its taking a long time for a single roll of dice when we take all the subsets and all combinations of indexes. This is because there are in total 1431 different outcomes possible. This needs to be across 2 levels, making it in the order of a lakh nodes. To work around this, we tried **(1)** to take only shorter subsets(e.g only 0 or 1 or 2 dice allowed to be rerolled) in the first reroll and then all the subsets in the 2nd reroll (since the 2nd reroll is all leaf nodes). **(2)** Direcly only computing expected values in the first lavel itself. This means our tree would be only 2 levels deep in total instead of 3. We found that **(1)** **(2)** dont give significantly different scores on average. Also, on average the scores are around 210, whereas expected is around 250
 
---------------------------------------------------------------
**PROBLEM 3 -DOCUMENT CLASSIFICATION**


**1)PROBLEM STATEMENT:** The problem makes use of the concept of Naives Bayes implementation for classifying tweets and calculating the accurracy.Naive Bayes technique can be used to classify several textual objects such as documents,emails,tweets into two specific cateogories such as spam vs non spam,important vs unimportant,acceptable vs inapproriate and so on.This technique uses a "bag of words" model.In this problem we have considered a textual object D which contains words w1,w2,w3....wn. There are two classes A and B. The classifier decides which class the object belongs to. 

**2)IMPLEMENTATION**:Firstly the testing and training data has been provided to us in form of tweets with labels East Coast and West Coast. These tweets are unordered.Firstly to improve the accuracy I had cleaned the data which means I have removed the special characters,puncuations and so on. For this I stored test_data[objects] into a variable called unfiltered data. A loop has then been set to traverse through unfiltered data. When a special character or a character is observed it is replaced by ''.

For "cleaning the data", the stopwords had to removed as well. Stopwords are generally referred to as a "single set of words" which is different for each application.Stopwords can be determiners,conjunctions and prepositions.It can also be domain specific. In case of this problem,we remove stopwords from nltk library.This nltk library can be downloaded. and this library is then imported into the program and is then we used to remove stopwords.

Then after the data has been cleaned the probabilty and the classification is carried about. For doing this two maps called east_coast and west_coast is created.If the test_data[labels] is east coast the element is added to the east_coast map. Similarly if thr label is west coast, it is moved to west coast map. 

After I performed this I noticed that it wasnt accurate enough. I had obtained an accuracy of 50.1. 

To correct above problem, we tried additive smoothing with many values of alpha - 0.1, 0.2, 1. We found Laplacian smoothing to be the best performing, giving an accuracy of **82.11**. We also tried removing words in our maps which didnt occur more than 2 or 3 times, however this decreased the accuracy, so we decided to remove that line of code.  
