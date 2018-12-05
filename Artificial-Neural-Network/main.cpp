/*
*This program is a simple artificial neural network
*It will preform feed forward and back propagation in order to train the network
*/

//HEADERS
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

//NAMESPACES
using namespace std;

//CONSTANTS

//File name for the data set that will be used to train the network
#define INPUT_FILE_NAME "data-set/train.txt"
//Output file to print the results in
#define OUPUT_FILE_NAME "Result.txt"
//Minimum MSE that will be used for the break condition
#define MSE 0.05
//Max number of iterations to be preformed
#define MAX_ITERATIONS 500



//Class to hold Neural Network data and operation that will be preformed
class NeuralNetwork
{
private:
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

	//ARRAYS
	//Array to hold the input values for the neural network
	float * input_values;
	//Array to hold the output values for the hidden layer nodes
	float * hidden_values;
	//Array to hold the final output values for the neural network
	float * output_values;
	//The actual values for the data set that will be used in computing the MSE
	float *actual_values;
	//Array to hold the MSE values
	float * mean_square_error;
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

		//Initialize the input values array
		input_values = new float[input_layer_nodes];

		//Initialize the hidden  layer output values array
		hidden_values = new float[hidden_layer_nodes];

		//Initialize the final output values array
		output_values = new float[output_layer_nodes];

		//Initialize the actual values array for the data set
		actual_values = new float[data_set_size];

		//Initialize the mean square error (MSE) array
		mean_square_error = new float[data_set_size];

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
	//Function to generate a random float number for the initial weights
	float generateRandomFloat()
	{
		float random_float = (rand() % 1001) / 1000.0;

		return random_float;

	}

public:
	//Function to read the data set from file
	void getInputFromFile()
	{
		//Create and open file variable to read input from
		fstream input_file;
		input_file.open(INPUT_FILE_NAME);

		//First input is the the number of nodes in the input layer
		input_file >> input_layer_nodes;
		//Second input is the number of nodes in the hidden layer
		input_file >> hidden_layer_nodes;
		//Third input is the number of nodes in the output layer
		input_file >> output_layer_nodes;
		//Fourth input is the data set size
		input_file >> data_set_size;

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
			input_file >> actual_values[i];
		}
		//Close the input file
		input_file.close();
	}


	//Print the data set
	void print()
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
			cout << actual_values[i] << endl;

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


	}

	//Function to deallocate resources that were created dynamically
	void dealocateResources()
	{

		delete[] input_values;
		delete[] hidden_values;
		delete[] output_values;
		delete[] actual_values;
		delete[] mean_square_error;

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


};



int main()
{
	system("color 70");
	srand(time(NULL));
	NeuralNetwork nn;
	nn.getInputFromFile();
	nn.print();
	nn.dealocateResources();

	return 0;
}
