/****************************************************************************************
*This program is a simple artificial neural network with 3 layers (input, hidden, output)
*Feed forward and back propagation will be preformed in order to train the network
*Optimal weights will be stored in a file (weights.txt) inside folder results
*****************************************************************************************/


//HEADERS
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <math.h>

//NAMESPACES
using namespace std;


//CONSTANTS
//File name for the data set that will be used to train the network
#define INPUT_FILE_NAME "data-set/train.txt"
//Output file to print the weights in
#define WEIGHTS_FILE_NAME "results/weights.txt"
//Output file to print the weights with details in
#define WEIGHTS_WITH_DETAILS_FILE_NAME "results/weights_with_details.txt"
//Minimum MSE that will be used for the break condition
#define MINIMUM_MSE 0.5
//Max number of iterations to be preformed during training
#define MAX_ITERATIONS_FOR_TRAINING 1
//Indicate if we want to train the neural network or not
#define ALLOW_TRAINING true
//Indicate if we want to test the nerual network or not
#define ALLOW_TESTING true



//Class to hold Neural Network data and operation that will be preformed
class NeuralNetwork
{
private:

	//Variable to hold the MSE of the data set
	float mean_square_error;
	//The last index of data we will use for training
	int training_data_max_index;
	//The starting index of data we will use for testing
	int testing_data_start_index;
	
	//SIZES
	//Number of nodes in input layer
	int input_layer_nodes;
	//Number of nodes in hidden layer
	int hidden_layer_nodes;
	//Number of nodes in output layer
	int output_layer_nodes;
	//Number of rows in the data set
	int data_set_size;

	//MATRICES
	//The matrix that is holding the weights between the input nodes layer and hidden nodes layer
	float ** input_hidden_weights;
	//The matrix that is holding the weights between the hidden nodes layer and output nodes layer
	float ** hidden_output_weights;
	//The data set will be taken from the file and stored in a matrix
	float ** data_set;
	//The actual output values for the data set that will be used in computing the MSE
	float **actual_values;
	//Matrix to hold the final output values for the neural network taken from the output layer nodes
	float ** output_values;

	//ARRAYS
	//Array to hold the input values for the neural network
	float * input_values;
	//Array to hold the output values for the hidden layer nodes
	float * hidden_values;
	
	/*
	*Array to hold the error values for each test
	*Will be used later to calculate the MSE
	*/
	float * square_error;
		
	//PRIVATE FUNCTIONS
	//Seed random function
	void seedRand()
	{
		//Declare variable to hold seconds on clock.
		time_t seconds;

		//Get value from system clock and	place in seconds variable
		time(&seconds);

		//Convert seconds to a unsigned integer
		srand((unsigned int)seconds);
	}

	//Function to Calculate the MSE
	void calculateMSE(string type)
	{
		mean_square_error = 0.0;
		int i = 0, max_index = 0;
		if (type == "test")
		{
			i = testing_data_start_index;
			max_index = data_set_size;

		}
		else
		{
			max_index = training_data_max_index;
		}
		for (; i < max_index; i++)
		{
			mean_square_error += square_error[i];
		}
		mean_square_error /= training_data_max_index;
	}

	//Function to initialize the matrices and arrays
	void initializeResources()
	{

		//Initialize the matrix of weights between the input and hidden layer
		input_hidden_weights = new float*[hidden_layer_nodes];
		for (int i = 0; i<hidden_layer_nodes; i++)
			input_hidden_weights[i] = new float[input_layer_nodes];

		//Initialize the matrix of weights between the hidden and output layer
		hidden_output_weights = new float*[output_layer_nodes];
		for (int i = 0; i<output_layer_nodes; i++)
			hidden_output_weights[i] = new float[hidden_layer_nodes];

		//Initialize the weight matrices with random floats
		initializeWeights();
		//Initialize the data set matrix
		data_set = new float*[data_set_size];
		for (int i = 0; i<data_set_size; i++)
			data_set[i] = new float[input_layer_nodes];

		//Initialize the final output values matrix
		output_values = new float*[data_set_size];
		for (int i = 0; i < data_set_size; i++)
			output_values[i] = new float[output_layer_nodes];

		//Initialize the input values array
		input_values = new float[input_layer_nodes];

		//Initialize the hidden  layer output values array
		hidden_values = new float[hidden_layer_nodes];

		

		//Initialize the actual values array for the data set
		actual_values = new float*[data_set_size];
		for (int i = 0; i < data_set_size; i++)
			actual_values[i] = new float[output_layer_nodes];

		//Initialize the mean square error (MSE) array
		square_error = new float[data_set_size];

	}

	//Function to randomly generate initial weights for the neural network
	void initializeWeights()
	{
		//Initialize weights between input layer and hidden layer
		for (int i = 0; i < hidden_layer_nodes; i++)
		{
			for (int j = 0; j< input_layer_nodes; j++)
			{
				input_hidden_weights[i][j] = generateRandomFloat();

			}

		}

		//Initialize weights between hidden layer and output layer
		for (int i = 0; i < output_layer_nodes; i++)
		{
			for (int j = 0; j< hidden_layer_nodes; j++)
			{
				hidden_output_weights[i][j] = generateRandomFloat();

			}

		}

	}

	//Function to generate a random float number for the initial weights between -1 and 1
	float generateRandomFloat()
	{	
		//Generate a random float between 0 - 1
		float random_float = (rand() % 1001) / 1000.0;
		if(rand()%100<=50)
		return random_float;
		return random_float * -1;

	}

	//Activation function (Sigmoid)
	void activationFunction(float &value)
	{
		float denom = 1 + exp(value*-1);
	   value = 1.0/denom;
	}

	//Function to preform matrix vector multiplication
	void sumOfProduct(float * &matrix, float * &vector,int &vector_size,float &sum_of_product)
	{  
		
		sum_of_product = 0.0;
		
		for (int i = 0; i < vector_size; i++)
		{
			//cout << "weight: " << matrix[i] << "  vector: " << vector[i] << endl;
			sum_of_product += (matrix[i] * vector[i]);
		}
		
	}

	//Function to calculate the square error for a certain test case
	void calculateSquareErrorForCurrentTestCase(int current_test_case)
	{   //Initialize the value 
		square_error[current_test_case] = 0.0;
		//Iterate over all the output values that was calculated from the output layer 
		for (int i = 0; i < output_layer_nodes; i++)
		{  //square error = Segma (actual-produced)^2
			square_error[current_test_case] += (actual_values[current_test_case][i] - output_values[current_test_case][i]) * (actual_values[current_test_case][i] - output_values[current_test_case][i]);
		}
	}

	//Function to set the input vector
	void setInputValues(int current_test_case)
	{
		for (int i = 0; i < input_layer_nodes; i++)
		{
			input_values[i] = data_set[current_test_case][i];
			
		}

	}
	
	//Output the weights with details
	void outputWeightsToFileWithDetails()
	{
		ofstream output_file;
		output_file.open(WEIGHTS_WITH_DETAILS_FILE_NAME);
		output_file << "Weights of the Neural Network : " << endl;

		output_file << "Weights of matrix between input and hidden layer : " << endl;

		for (int i = 0; i < hidden_layer_nodes; i++)
		{
			output_file << "Weights of each input node to hidden node #" << i + 1 << " : ";
			for (int j = 0; j < input_layer_nodes; j++)
			{
				output_file << input_hidden_weights[i][j];
				if (j < input_layer_nodes - 1)
					output_file << " , ";
			}
			output_file << endl;
		}

		output_file << "Weights of matrix between hidden and output layer : " << endl;

		for (int i = 0; i < output_layer_nodes; i++)
		{
			output_file << "Weights of each hidden node to output node #" << i + 1 << " : ";
			for (int j = 0; j < hidden_layer_nodes; j++)
			{
				output_file << hidden_output_weights[i][j];
				if (j < hidden_layer_nodes - 1)
					output_file << " , ";
			}
			output_file << endl;
		}
		output_file.close();
	}

	//Output the weights without details
	void outputWeightstoFile()
	{
		ofstream output_file;
		output_file.open(WEIGHTS_FILE_NAME);
		for (int i = 0; i < hidden_layer_nodes; i++)
		{
			for (int j = 0; j < input_layer_nodes; j++)
			{
				output_file << input_hidden_weights[i][j];
				if (j < input_layer_nodes - 1)
					output_file << " ";
			}
			output_file << endl;
		}
		for (int i = 0; i < output_layer_nodes; i++)
		{
			for (int j = 0; j < hidden_layer_nodes; j++)
			{
				output_file << hidden_output_weights[i][j];
				if (j < hidden_layer_nodes - 1)
					output_file << " ";
			}
			output_file << endl;
		}
		output_file.close();
	}


public:

	//PUBLIC FUNCTIONS
	//Function return the MSE
	float getMSE()
	{
		
		return mean_square_error;
	}
	//Function to read the data set from file
	void getInputFromFile()
	{
		//Create and open file variable to read input from
		fstream input_file;
		input_file.open(INPUT_FILE_NAME);
		//Seed
		seedRand();
		//First input is the the number of nodes in the input layer
		input_file >> input_layer_nodes;
		//Second input is the number of nodes in the hidden layer
		input_file >> hidden_layer_nodes;
		//Third input is the number of nodes in the output layer
		input_file >> output_layer_nodes;
		//Fourth input is the data set size
		input_file >> data_set_size;

        //Half of the data set will be used for training and the other half for testing 
		training_data_max_index = data_set_size / 2;
		testing_data_start_index = data_set_size / 2;
		

		//Initialize the matrices and arrays
		initializeResources();
		

		//Iterate over the whole file to store the data set in our program
		for (int i = 0; i<data_set_size; i++)
		{
			//Read the input values for each data row in the data set
			for (int j = 0; j<input_layer_nodes; j++)
			{
				input_file >> data_set[i][j];

			}
			for(int j=0;j<output_layer_nodes;j++)
			input_file >> actual_values[i][j];
		}
		//Close the input file
		input_file.close();
	}

	//Print the details of the neural network
	void printNetworkDetails()
	{
		cout << "Number of nodes in the input layer: " << input_layer_nodes << endl;
		cout << "Number of nodes in the hidden layer: " << hidden_layer_nodes << endl;
		cout << "Number of nodes in the output layer: " << output_layer_nodes << endl;
		cout << "Size of the data set: " << data_set_size << endl;

		cout << "The data set: " << endl;
		for (int i = 0; i<data_set_size; i++)
		{
			cout << "Data row number " << i + 1 << " : ";
			for (int j = 0; j<input_layer_nodes; j++)
			{
				cout << data_set[i][j] << " ";

			}
			for(int j=0;j<output_layer_nodes;j++)
			cout << actual_values[i][j] << " ";
			cout << endl;

		}
		cout << endl << "Input hidden layer weights matrix : " << endl;
		for (int i = 0; i < hidden_layer_nodes; i++)
		{
			for (int j = 0; j< input_layer_nodes; j++)
			{
				cout << input_hidden_weights[i][j] << " ";

			}
			cout << endl;

		}

		cout << endl << "Hidden output layer weights matrix : " << endl;
		for (int i = 0; i < output_layer_nodes; i++)
		{
			for (int j = 0; j< hidden_layer_nodes; j++)
			{
				cout << hidden_output_weights[i][j] << " ";

			}
			cout << endl;
		}
		cout << endl;

	}

	//Print values and MSE of the network after training
	void printResultsOfTraining()
	{
		
		cout << "MSE = " << mean_square_error << endl;

		cout << "Results of the network compared to actual : " << endl;

		for (int i = 0; i < training_data_max_index; i++)
		{
			cout << "data set #" << i + 1 << " actual output: ";
			for (int j = 0; j < output_layer_nodes; j++)
			{
				cout << actual_values[i][j] << " ";
			}
			cout << "network output: ";
			for (int j = 0; j < output_layer_nodes; j++)
			{
				cout << output_values[i][j] << " ";
			}
			cout << endl;
		}
		
	}

	//Print values and MSE of the network after testing
	void printResultsOfTesting()
	{

		cout << "MSE = " << mean_square_error << endl;

		cout << "Results of the network compared to actual : " << endl;

		for (int i = testing_data_start_index; i < data_set_size; i++)
		{
			cout << "data set #" << i + 1 << " actual output: ";
			for (int j = 0; j < output_layer_nodes; j++)
			{
				cout << actual_values[i][j] << " ";
			}
			cout << "network output: ";
			for (int j = 0; j < output_layer_nodes; j++)
			{
				cout << output_values[i][j] << " ";
			}
			cout << endl;
		}

	}


	//Destructor
	~NeuralNetwork()
	{
		delete[] input_values;
		delete[] hidden_values;
		delete[] square_error;

		for (int i = 0; i < output_layer_nodes; i++)
			delete[] output_values[i];
		delete[] output_values;

		for (int i = 0; i < output_layer_nodes; i++)
			delete[] actual_values[i];
		delete[]actual_values;

		for (int i = 0; i<hidden_layer_nodes; i++)
			delete[] input_hidden_weights[i];
		delete[] input_hidden_weights;

		for (int i = 0; i<output_layer_nodes; i++)
			delete[] hidden_output_weights[i];
		delete[] hidden_output_weights;

		for (int i = 0; i<data_set_size; i++)
			delete[] data_set[i];
		delete[] data_set;

	}

	
	//Feed forward function
	void feedForward(string type)
	{ //iterator for the algorithm
		int i = 0,max_index=0;
		if (type == "test")
		{
			i = testing_data_start_index;
			max_index = data_set_size;

		}
		else
		{
			max_index = training_data_max_index;
		}
		
		for (; i < max_index; i++)
		{
			
			//Set the input vector with the data row of the current iteration
			setInputValues(i);
			
			/*
			 *First step is to do sum of product on each hidden layer node
			 *Second step is to apply activation function and store the output values to be used as an input for next layer
			 */
			
			for (int j = 0; j < hidden_layer_nodes; j++)
			{   
			
				//Preform sum of product on each hidden layer node
				sumOfProduct(input_hidden_weights[j], input_values, input_layer_nodes, hidden_values[j]);
				
				//Calculate the output of each hidden layer node by applying activiation function (Sigmoid)
				activationFunction(hidden_values[j]);
				
				
			}

			
			/*
			*First step is to do sum of product on each output layer node
			*Second step is to apply activation function and store the output values to be used to caluclate the error
			*/
			for (int j = 0; j < output_layer_nodes; j++)
			{
				
				//Preform sum of product on each output layer node
				sumOfProduct(hidden_output_weights[j], hidden_values, hidden_layer_nodes, output_values[i][j]);
				
				
				//Calculate the output of each output layer node by applying activiation function (Sigmoid)
				activationFunction(output_values[i][j]);
			}
			
			//Calculate the square error
			calculateSquareErrorForCurrentTestCase(i);
		}
		//Calculate the MSE
		calculateMSE(type);
	}

	/*
	*Function to output best weights to file 
	*First function (outputWeightstoFile) will output the weights without details so we can use those weights later
	*Second function (outputWeightsToFileWithDetails) will output the weights with details for clarification
	*/
	void outputWeightsToFiles()
	{
		outputWeightstoFile();
		outputWeightsToFileWithDetails();
		
	}
	
	//Function to load weights of matrices from file
	void loadWeightsFromFile()
	{

		fstream weights_file;
		weights_file.open(WEIGHTS_FILE_NAME);
		
		for (int i = 0; i < hidden_layer_nodes; i++)
		{
			for (int j = 0; j < input_layer_nodes; j++)
			{
				weights_file >> input_hidden_weights[i][j];
			}
		}
		for (int i = 0; i < output_layer_nodes; i++)
		{
			for (int j = 0; j < hidden_layer_nodes; j++)
			{
				weights_file >> hidden_output_weights[i][j];
			}
		}
		weights_file.close();

	}

};

//GLOBAL VARIABLES
//Our neural network
NeuralNetwork neuralnetwork;

//FUNCTION PROTOTYPES
void trainNetwork();
void testNetwork();

//Entry point of our program
int main()
{	system("color 70");

	//Get input (training or testing) from file	
	neuralnetwork.getInputFromFile();

	//Train the neural network
    #if ALLOW_TRAINING
	trainNetwork();
    #endif
	//Test the neural network
    #if ALLOW_TESTING
	testNetwork();
    #endif
	system("pause");
	return 0;
}

//FUNCTION BODIES
//Function to allow us to train the neural network when we want
void trainNetwork()
{
	
	for (int i = 0; i < MAX_ITERATIONS_FOR_TRAINING; i++)
	{
		neuralnetwork.feedForward("train");
		
		if (neuralnetwork.getMSE()<= MINIMUM_MSE)
		{
			cout << "achieved MSE less than the required minimum" << endl;
			/*neuralnetwork.printResultsOfTraining();
			neuralnetwork.outputWeightsToFiles();*/
			return;
		}
		//Backpropagation goes here
		//neuralnetwork.backPropagation();
	}
	cout << "Couldnt get a mean square error lower than the required minimum" << endl;
	/*neuralnetwork.printResultsOfTraining();*/
	neuralnetwork.outputWeightsToFiles();
	return;
}

//Function to allow us to test the neural network when we want
void testNetwork()
{
	neuralnetwork.loadWeightsFromFile();
	neuralnetwork.feedForward("test");
	neuralnetwork.printNetworkDetails();
	neuralnetwork.printResultsOfTesting();

}