# turbo-garbanzo
AI-prog siste øving

Evaluation for  Project 3  (SOM for TSP and MNIST)

At the demo session, you will earn up to 30 points.  There is no written report for this project.  The points will be distributed as follows:

	•	A fully-functioning SOM for solving TSP (3 points)
	•	
	•	Graphic visualization of the SOM’s progress as it solves the TSP: 
	the standard 2-d picture that shows the output “ring” gradually fitting 
	to the city locations. (2 points)
	•	
	•	TSP Performance Test (10 points) – You will be given 3 cases from the 
	8 that you were already given.  In addition, you will get 2 more cases of 
	the same file format and containing no more than 150 cities.  
	Each case will be worth 2 points.  You need to get an answer within 10% of
	 the optimum to get both points.  Do not count on partial credit for these.
	•	
	•	A fully-functioning  2-d SOM for classifying MNIST cases (3 points)
	•	
	•	Graphic visualization of the 2-d SOM as it gradually learns to correctly
	 classify the training cases (3 points)
	•	
	•	MNIST Performance (5 points) – Using any dimensions of the SOM that you 
	find to be appropriate, train on a collection of at least 500 randomly-selected
	 MNIST cases.  Once training has finished, run each training case through the 
	 SOM one final time and keep track of the number of correct classifications 
	 (i.e. the training accuracy).  Next, run a set of at least 100 randomly-selected
	  MNIST cases (NONE of which can occur in the training set) and calculate the 
	  testing accuracy.  Training accuracy must be 85% or better to get full credit (3 points),
	   while test accuracy must be 75% or better to get full credit (2 points).
	•	
	•	Oral Discussion (4 points) – Discuss the key parameters for the SOM and 
	your experiences using them.  A good discussion will include results from different
	 runs (of both TSP and MNIST) using different values of the key parameters.  
	 By now, you should know what these key parameters are.


Additional Comments

1)
 If you are having trouble with speed (or lack thereof), be sure to check the numpy 
 documentation for matrix or vector operations that could do many (e.g. 784) 
 simple arithmetic operations in one function call.  
 The numpy operators are written in C and are thus very fast.  
 One numpy vector subtraction, for example, is much faster than a Python for loop
  that does all of the individual element-by-element subtractions in Python.

2) 

The 2-dimensional MNIST SOM will perform a classification task, so you need a way to map input vectors (of 784 pixels) to classes (0 through 9).  Here’s a simple way to do so:

On every kth learning epoch, turn on a special mode that forces the network to keep track of the cases that each node wins.  At the end of this special epoch, each node can compute its class by taking the majority label of its cases.  For example, if a node has 10 cases labeled “5” and 4 cases labeled “6”, then the node’s class becomes a “5”.

Although I made an earlier suggestion to compute an average of the labels, when I got the chance to implement it myself, I found that the “majority rule” works better than the average (which can give some misleading results for small data sets).  

Once each node has an associated class, testing becomes trivial.  You simply turn off learning, run each case through the network, and record the class associated with the winner node for each case.  The percentage of those classes that match the true labels of the corresponding cases constitutes the accuracy.  This accuracy can be computed for both the training cases and the test cases.

3)

In addition, the classes associated with each node allow you to color-code a diagram of your output layer.  As shown below, these color-coded diagrams give a nice visualization of learning progress in the SOM.


Above is a color-code view of a 10x10 Kohonen net for MNIST classification.  The net is poorly trained, as evidenced by the nearly random distribution of colors.


The diagram above shows a well-trained SOM.  Note the large patches of similar color, which illustrate the effects of cooperation inherent in the SOM algorithm.

This type of diagram is relatively easy to produce using the networkX package for Python.  I’ve used hexagonal connectivity and diamond shape, but you are free to use a simple lattice structure where each node has 4 neighbors (east, west, north and south) or 8 neighbors (east, southeast, south, southwest, west, northwest, north, northeast). 

 This is only one suggested visualization technique; there are many others that convey both the neighborhood structure of the output layer and the general behavior of the output nodes.  Be sure to choose a diagram type that clearly illustrates the clustering effect (i.e. neighboring nodes detect similar patterns) that is the hallmark of the SOM.
