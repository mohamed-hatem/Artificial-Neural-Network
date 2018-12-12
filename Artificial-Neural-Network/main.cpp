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
//Max error that will be used for the break condition
#define MAX_TOTAL_ERROR 0.000005
//Minimum error that will be used for the break condition
#define MIN_TOTAL_ERROR -0.00005
//Max number of iterations to be preformed during training
#define MAX_EPOCHS 500
//Learning rate constant
#define LEARNING_RATE 0.5


//Class to hold Neural Network data and operation that will be preformed
class NeuralNetwork
{
private:

	//Variable to hold the current total error for the current data record
	float error_of_current_data_record;
	//The last index of data we will use for training
	int training_data_max_index;
	//The starting index of data we will use for testing
	int testing_data_start_index;
	
	//SIZES
	//Number of nodes in input layer
	int input_layer_neurons;
	//Number of nodes in hidden layer
	int hidden_layer_neurons;
	//Number of nodes in output layer
	int output_layer_neurons;
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
	//Array to hold the input value of a certain record for the neural network
	float * input_values;
	//Array to hold the output value of a certain record for the hidden layer nodes
	float * hidden_values;
	/*
	*Array to hold the error values for each output node for each data record in the data set
	*Error values will be used later to calculate the total error and backpropagation
	*/
	float * record_error;
			
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

	//Function to initialize the matrices and arrays
	void initializeResources()
	{

		//Initialize the matrix of weights between the input and hidden layer
		input_hidden_weights = new float*[hidden_layer_neurons];
		for (int i = 0; i<hidden_layer_neurons; i++)
			input_hidden_weights[i] = new float[input_layer_neurons];

		//Initialize the matrix of weights between the hidden and output layer
		hidden_output_weights = new float*[output_layer_neurons];
		for (int i = 0; i<output_layer_neurons; i++)
			hidden_output_weights[i] = new float[hidden_layer_neurons];

		//Initialize the weight matrices with random floats
		initializeWeights();
		//Initialize the data set matrix
		data_set = new float*[data_set_size];
		for (int i = 0; i<data_set_size; i++)
			data_set[i] = new float[input_layer_neurons];

		//Initialize the final output values matrix
		output_values = new float*[data_set_size];
		for (int i = 0; i < data_set_size; i++)
			output_values[i] = new float[output_layer_neurons];

		//Initialize the input values array
		input_values = new float[input_layer_neurons];

		//Initialize the hidden  layer output values array
		hidden_values = new float[hidden_layer_neurons];

		

		//Initialize the actual values array for the data set
		actual_values = new float*[data_set_size];
		for (int i = 0; i < data_set_size; i++)
			actual_values[i] = new float[output_layer_neurons];

		//Initialize the record error array
			record_error= new float[output_layer_neurons];

	}

	//Function to randomly generate initial weights for the neural network
	void initializeWeights()
	{
		//Initialize weights between input layer and hidden layer
		for (int i = 0; i < hidden_layer_neurons; i++)
		{
			for (int j = 0; j< input_layer_neurons; j++)
			{
				input_hidden_weights[i][j] = generateRandomFloat();

			}

		}

		//Initialize weights between hidden layer and output layer
		for (int i = 0; i < output_layer_neurons; i++)
		{
			for (int j = 0; j< hidden_layer_neurons; j++)
			{
				hidden_output_weights[i][j] = generateRandomFloat();

			}

		}

	}

	//Function to generate a random float number for the initial weights between -10 and 10
	float generateRandomFloat()
	{	
		//Generate a random float between 0 - 10
		float random_float = (rand() % 1001) / 100.0;
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

	//Function to preform Sum of product for a certain node 
	void sumOfProduct(float * &weights, float * &inputs,int &vector_size,float &sum_of_product)
	{  
		
		sum_of_product = 0.0;
		
		for (int i = 0; i < vector_size; i++)
		{
			
			sum_of_product += (weights[i] * inputs[i]);
		}
		
	}

	//Function to calculate the error for a certain data record
	void calculateErrorForCurrentTestCase(int current_data_record)
	{
		
		
		//Iterate over all the output values that was calculated from the output layer 
		for (int i = 0; i < output_layer_neurons; i++)
		{  // error = (actual-produced)
			record_error[i] = 0.0;
			record_error[i] += (actual_values[current_data_record][i] - output_values[current_data_record][i]);
			
		}
		
		
	}

	//Function to set the input vector
	void setInputValues(int current_data_record)
	{
		for (int i = 0; i < input_layer_neurons; i++)
		{
			input_values[i] = data_set[current_data_record][i];
			
		}

	}
	
	//Output the weights with details
	void outputWeightsToFileWithDetails()
	{
		ofstream output_file;
		output_file.open(WEIGHTS_WITH_DETAILS_FILE_NAME);
		output_file << "Weights of the Neural Network : " << endl;

		output_file << "Weights of matrix between input and hidden layer : " << endl;

		for (int i = 0; i < hidden_layer_neurons; i++)
		{
			output_file << "Weights of each input node to hidden node #" << i + 1 << " : ";
			for (int j = 0; j < input_layer_neurons; j++)
			{
				output_file << input_hidden_weights[i][j];
				if (j < input_layer_neurons - 1)
					output_file << " , ";
			}
			output_file << endl;
		}

		output_file << "Weights of matrix between hidden and output layer : " << endl;

		for (int i = 0; i < output_layer_neurons; i++)
		{
			output_file << "Weights of each hidden node to output node #" << i + 1 << " : ";
			for (int j = 0; j < hidden_layer_neurons; j++)
			{
				output_file << hidden_output_weights[i][j];
				if (j < hidden_layer_neurons - 1)
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
		for (int i = 0; i < hidden_layer_neurons; i++)
		{
			for (int j = 0; j < input_layer_neurons; j++)
			{
				output_file << input_hidden_weights[i][j];
				if (j < input_layer_neurons - 1)
					output_file << " ";
			}
			output_file << endl;
		}
		for (int i = 0; i < output_layer_neurons; i++)
		{
			for (int j = 0; j < hidden_layer_neurons; j++)
			{
				output_file << hidden_output_weights[i][j];
				if (j < hidden_layer_neurons - 1)
					output_file << " ";
			}
			output_file << endl;
		}
		output_file.close();
	}

	//Function to Calculate the Total Error for a data record
	void calculateTotalError()
	{
		error_of_current_data_record = 0.0;

		for (int i = 0; i < output_layer_neurons; i++)
		{
			error_of_current_data_record += (record_error[i] * record_error[i]);
		}

		error_of_current_data_record /= 2;
	}


public:

	//PUBLIC FUNCTIONS
	//Function return the Total Error of a data record
	float getTotalErrorOfCurrentRecord()
	{
		
		return error_of_current_data_record;
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
		input_file >> input_layer_neurons;
		//Second input is the number of nodes in the hidden layer
		input_file >> hidden_layer_neurons;
		//Third input is the number of nodes in the output layer
		input_file >> output_layer_neurons;
		//Fourth input is the data set size
		input_file >> data_set_size;

        //Only 10% of the data set will be used for testing  
		training_data_max_index = data_set_size *0.9;
		testing_data_start_index = training_data_max_index;
		

		//Initialize the matrices and arrays
		initializeResources();
		

		//Iterate over the whole file to store the data set in our program
		for (int i = 0; i<data_set_size; i++)
		{
			//Read the input values for each data row in the data set
			for (int j = 0; j<input_layer_neurons; j++)
			{
				input_file >> data_set[i][j];

			}
			for(int j=0;j<output_layer_neurons;j++)
			input_file >> actual_values[i][j];
		}
		//Transform the actual output values using the activation function
		for (int i = 0; i < data_set_size; i++)
		{
			for (int j = 0; j < output_layer_neurons; j++)
				activationFunction(actual_values[i][j]);
		}

		//Close the input file
		input_file.close();
	}

	//Print values of the network after training
	void printResultsOfTraining(int max_index)
	{
		float mse = 0.0;

		for (int i = 0; i < max_index; i++)
		{
			for (int j = 0; j < output_layer_neurons; j++)
			{
				mse = mse + ((actual_values[i][j] - output_values[i][j])*(actual_values[i][j] - output_values[i][j]));

			}
		}

		mse = mse / ((max_index));
		

		cout << "Results of the network compared to actual : " << endl;
		cout << "MSE of training = " << mse << endl;
		for (int i = 0; i < max_index; i++)
		{
			cout << "data set #" << i + 1 << " actual output: ";
			for (int j = 0; j < output_layer_neurons; j++)
			{
				cout << actual_values[i][j] << " ";
			}
			cout << "network output: ";
			for (int j = 0; j < output_layer_neurons; j++)
			{
				cout << output_values[i][j] << " ";
			}
			cout << endl;
		}
		
	}

	//Print values of the network after testing
	void printResultsOfTesting()
	{
		float mse = 0.0;

			for (int i = testing_data_start_index; i < data_set_size; i++)
			{
				for (int j = 0; j < output_layer_neurons; j++)
				{
					mse = mse + ((actual_values[i][j] - output_values[i][j])*(actual_values[i][j] - output_values[i][j]));
			
				}
			}
			
			mse = mse/((data_set_size - testing_data_start_index));
		

		cout << "Results of the network compared to actual : " << endl;
		cout << "MSE of testing = " << mse << endl;
		for (int i = testing_data_start_index; i < data_set_size; i++)
		{
			cout << "data set #" << i + 1 << " actual output: ";
			for (int j = 0; j < output_layer_neurons; j++)
			{
				cout << actual_values[i][j] << " ";
			}
			cout << "network output: ";
			for (int j = 0; j < output_layer_neurons; j++)
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
		delete[] record_error;

		for (int i = 0; i < output_layer_neurons; i++)
			delete[] output_values[i];
		delete[] output_values;

		for (int i = 0; i < output_layer_neurons; i++)
			delete[] actual_values[i];
		delete[]actual_values;

		for (int i = 0; i<hidden_layer_neurons; i++)
			delete[] input_hidden_weights[i];
		delete[] input_hidden_weights;

		for (int i = 0; i<output_layer_neurons; i++)
			delete[] hidden_output_weights[i];
		delete[] hidden_output_weights;

		for (int i = 0; i<data_set_size; i++)
			delete[] data_set[i];
		delete[] data_set;

	}

	
	//Feed forward function
	void feedForward(int data_record_index)
	{ 
		
		
			
			//Set the input vector with the data set values of the current data record
			setInputValues(data_record_index);
			
			/*
			 *First step is to do sum of product on each hidden layer node
			 *Second step is to apply activation function and store the output values to be used as an input for the output layer
			 */
			
			for (int j = 0; j < hidden_layer_neurons; j++)
			{   
			
				//Preform sum of product on each hidden layer node
				sumOfProduct(input_hidden_weights[j], input_values, input_layer_neurons, hidden_values[j]);
				
				//Calculate the output of each hidden layer node by applying activiation function (Sigmoid)
				activationFunction(hidden_values[j]);
				
				
			}

			
			/*
			*First step is to do sum of product on each output layer node
			*Second step is to apply activation function and store the output values to be used to calculate the error
			*/
			for (int j = 0; j < output_layer_neurons; j++)
			{
				
				//Preform sum of product on each output layer node
				sumOfProduct(hidden_output_weights[j], hidden_values, hidden_layer_neurons, output_values[data_record_index][j]);
				
				
				//Calculate the output of each output layer node by applying activiation function (Sigmoid)
				activationFunction(output_values[data_record_index][j]);
			}
			
			//Calculate the  error for each output neuron against actual value
			calculateErrorForCurrentTestCase(data_record_index);
		   //Calculate total error
			calculateTotalError();
		
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
		
		for (int i = 0; i < hidden_layer_neurons; i++)
		{
			for (int j = 0; j < input_layer_neurons; j++)
			{
				weights_file >> input_hidden_weights[i][j];
			}
		}
		for (int i = 0; i < output_layer_neurons; i++)
		{
			for (int j = 0; j < hidden_layer_neurons; j++)
			{
				weights_file >> hidden_output_weights[i][j];
			}
		}
		weights_file.close();

	}
	
	//Function to get the last index of training data set
	int getMaxTrainingIdex()
	{
		return training_data_max_index;
	}

	//Function to get the data set size
	int getDataSetSize()
	{
		return data_set_size;
	}

	//Function to do backpropagation for a certain data record
	void backPropagation(int data_record_index)
	{   //Matrix that will be used to store old weights of hidden_output layers
		float ** old_weight_values = new float*[output_layer_neurons];
		for (int i = 0; i < output_layer_neurons; i++)
			old_weight_values[i] = new float[hidden_layer_neurons];

		for (int i = 0; i < output_layer_neurons; i++)
		{
			for (int j = 0; j < hidden_layer_neurons; j++)
				old_weight_values[i][j] = hidden_output_weights[i][j];
		}

		float * delta_change_output = new float[output_layer_neurons],delta_weight_change=0.0, delta_change;
		//Backpropagate throught the output layer first and change the weights according to the rules
		for (int i = 0; i < output_layer_neurons; i++)
		{
			delta_change_output[i] = (-1 * getTotalErrorOfCurrentRecord())*(output_values[data_record_index][i] *(1 - output_values[data_record_index][i]));
			for (int j = 0; j < hidden_layer_neurons; j++)
			{
				delta_weight_change = LEARNING_RATE * delta_change_output[i] * hidden_values[j];
				hidden_output_weights[i][j] = old_weight_values[i][j] - delta_weight_change;
			}
		}
		float error_weight_sum_of_product;

		//Backpropagate throught the hidden layer and change the weights according to the rules
		for (int i = 0; i < hidden_layer_neurons; i++)
		{
			error_weight_sum_of_product = 0.0;
			delta_change = 0.0;
			for (int j = 0; j < input_layer_neurons; j++)
			{ 
				for (int k = 0; k< output_layer_neurons; k++)
				{
					error_weight_sum_of_product += (delta_change_output[k] * hidden_output_weights[k][i]);
				}
				

				
				delta_change = (hidden_values[i] * (1 - hidden_values[i]))*error_weight_sum_of_product;
				delta_weight_change = LEARNING_RATE * delta_change * input_values[j];
				input_hidden_weights[i][j] = input_hidden_weights[i][j] - delta_weight_change;
			}
		}



		for (int i = 0; i < output_layer_neurons; i++)
			delete[] old_weight_values[i];
		delete[] old_weight_values;
		delete[] delta_change_output;


	}


};

//GLOBAL VARIABLES
//Our neural network
NeuralNetwork neuralnetwork;

//FUNCTION PROTOTYPES
bool trainNetwork();
void testNetwork();

//Entry point of our program
int main()
{	system("color 70");

	//Get input (training or testing) from file	
	neuralnetwork.getInputFromFile();

	//Train the neural network

	bool achieved_acceptable_error = trainNetwork();

	//Test the neural network
	if(achieved_acceptable_error)
	testNetwork();

	system("pause");
	return 0;
}

//FUNCTION BODIES
//Function to allow us to train the neural network when we want
bool trainNetwork()
{
	
	for (int i = 0; i < MAX_EPOCHS; i++)
	{
		for (int j = 0; j < neuralnetwork.getMaxTrainingIdex(); j++)
		{
			neuralnetwork.feedForward(j);
			
			if (neuralnetwork.getTotalErrorOfCurrentRecord() < MAX_TOTAL_ERROR && neuralnetwork.getTotalErrorOfCurrentRecord() > MIN_TOTAL_ERROR)
			{
				cout << "achieved Total Error less than the required minimum" << endl;
				cout << "Total Error : " << neuralnetwork.getTotalErrorOfCurrentRecord() << endl << "i = " << j << endl;
				neuralnetwork.printResultsOfTraining(j+1);
				neuralnetwork.outputWeightsToFiles();
				return true;
			}
			
			neuralnetwork.backPropagation(j);
		}
		
		
		
	}
	cout << "Couldnt get an error value lower than the required minimum" << endl;
	neuralnetwork.printResultsOfTraining(neuralnetwork.getMaxTrainingIdex());
	neuralnetwork.outputWeightsToFiles();
	return false;
}

//Function to allow us to test the neural network when we want
void testNetwork()
{
	neuralnetwork.loadWeightsFromFile();
	for(int j=neuralnetwork.getMaxTrainingIdex();j<neuralnetwork.getDataSetSize();j++)
	neuralnetwork.feedForward(j);
	neuralnetwork.printResultsOfTesting();

}