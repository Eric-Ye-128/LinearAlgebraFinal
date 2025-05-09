import java.util.*;

class Main {
	public static void main(String[] args) {
		System.out.println(activationFunc(-2.0));
		System.out.println(activationFunc(-1.0));
		System.out.println(activationFunc(0.0));
		System.out.println(activationFunc(1.0));
		System.out.println(activationFunc(2.0));

		// inputs and outputs
		int inputSize = 2, outputSize = 2;
		double[] input = new double[inputSize];
		input[0] = 1;
		input[1] = 1;
		double[] output = new double[outputSize];
		
		// creating hidden layers
		ArrayList<HiddenLayer> hiddenLayers = new ArrayList<HiddenLayer>();
		int hiddenCount = 2, hiddenSize = 3;
		for (int i = 0; i < hiddenCount; i++) {
			hiddenLayers.add(new HiddenLayer(new double[hiddenSize]));
		}

		// creating weight layers
		ArrayList<WeightLayer> weightLayers = new ArrayList<WeightLayer>();
		weightLayers.add(new WeightLayer(new HiddenLayer(input), hiddenLayers.get(0)));
		for (int i = 0; i < hiddenCount - 1; i++) {
			weightLayers.add(new WeightLayer(hiddenLayers.get(i), hiddenLayers.get(i + 1)));
		}
		weightLayers.add(new WeightLayer(hiddenLayers.get(hiddenLayers.size() - 1), new HiddenLayer(output)));

		// randomizing initial weights
		for (int i = 0; i < weightLayers.size(); i++) {
			weightLayers.get(i).initializeWeights();
		}

		// applies weights
		for (int i = 0; i < weightLayers.size(); i++) {
			weightLayers.get(i).applyWeights();
		}
		output = weightLayers.get(weightLayers.size() - 1).getNext().get();

		//----------------------------------------------------

		System.out.println("weight layers: ");
		for (int i = 0; i < weightLayers.size(); i++) {
			double[][] temp = weightLayers.get(i).getWeights();
			
			for (double[] row : temp) System.out.println(Arrays.toString(row));
			System.out.println();
		}

		System.out.println("hidden layers: ");
		for (int i = 0; i < hiddenLayers.size(); i++) {
			System.out.println(Arrays.toString(hiddenLayers.get(i).get()));
		}

		System.out.println("output layer: ");
		System.out.println(Arrays.toString(output));
	}
	
	public static double[][] multiplyMatrix(double[][] A, double[][] B) {
		double[][] out = new double[A.length][B[0].length];

		for (int row = 0; row < out.length; row++) {
			for (int col = 0; col < out[0].length; col++) {
				for (int index = 0; index < A[0].length; index++) {
					out[row][col] += A[row][index] * B[index][col];
				}
			}
		}

		return out;
	}

	public static double[][] transposeMatrix(double[][] A) {
		double[][] T = new double[A[0].length][A.length];
		for (int i = 0; i < A.length; i++) {
			for (int j = 0; j < A[0].length; j++) {
				T[j][i] = A[i][j];
			}
		}
		return T;
	}

	public static double activationFunc(double num) {
		return (1 / (1 + Math.exp(-num)));
	}
}
