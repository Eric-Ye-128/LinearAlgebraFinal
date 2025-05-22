import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.util.*;

class Main {
	public static void main(String[] args) {
		double eta = 0.1;

		// inputs and outputs
		File file = null;
		Scanner scan = null;
		String[] line = null;

		try {
			file = new File("trainingSet.txt");
			scan = new Scanner(file);
		} catch (Exception e) {
			System.out.println("FIle not found");
            e.printStackTrace();
			return;
		}

		ArrayList<double[]> inputs = new ArrayList<double[]>();
		ArrayList<double[]> targets = new ArrayList<double[]>();
		ArrayList<double[]> outputs = new ArrayList<double[]>();

		while (scan.hasNext()) {
			line = scan.nextLine().split(" ");
			double[] input = new double[line.length];
			for (int i = 0; i < input.length; i++) input[i] = Double.parseDouble(line[i]);
			inputs.add(input);

			line = scan.nextLine().split(" ");
			double[] target = new double[line.length];
			for (int i = 0; i < target.length; i++) target[i] = Double.parseDouble(line[i]);
			targets.add(target);

			if (scan.hasNextLine()) scan.nextLine();
		}

		// System.out.println("inputs: ");
		// for (int i = 0; i < inputs.size(); i++) System.out.println(Arrays.toString(inputs.get(i)));

		// System.out.println("targets: ");
		// for (int i = 0; i < targets.size(); i++) System.out.println(Arrays.toString(targets.get(i)));


		double[] currInput = new double[inputs.get(0).length];
		double[] currOutput = new double[targets.get(0).length];
		
		// creating hidden layers
		ArrayList<HiddenLayer> hiddenLayers = new ArrayList<HiddenLayer>();
		int hiddenCount = 1, hiddenSize = 3;
		for (int i = 0; i < hiddenCount; i++) {
			hiddenLayers.add(new HiddenLayer(new double[hiddenSize]));
		}

		// creating weight layers
		ArrayList<WeightLayer> weightLayers = new ArrayList<WeightLayer>();
		weightLayers.add(new WeightLayer(new HiddenLayer(currInput), hiddenLayers.get(0)));
		for (int i = 0; i < hiddenCount - 1; i++) {
			weightLayers.add(new WeightLayer(hiddenLayers.get(i), hiddenLayers.get(i + 1)));
		}
		weightLayers.add(new WeightLayer(hiddenLayers.get(hiddenLayers.size() - 1), new HiddenLayer(currOutput)));

		// randomizing initial weights
		for (int i = 0; i < weightLayers.size(); i++) weightLayers.get(i).initialize();

		// applies weights
		for (int i = 0; i < inputs.size(); i++) {
			weightLayers.get(0).getLast().set(inputs.get(i));
			for (int j = 0; j < weightLayers.size(); j++) {
				weightLayers.get(j).applyWeights();
			}
			outputs.add(weightLayers.get(weightLayers.size() - 1).getNext().get());

			System.out.printf("%-15s | %-45s | %-15s\n", Arrays.toString(inputs.get(i)), 
				Arrays.toString(outputs.get(outputs.size() - 1)), Arrays.toString(targets.get(i)));
		}


		System.out.println("weight layers: ");
		for (int i = 0; i < weightLayers.size(); i++) {
			double[][] temp = weightLayers.get(i).getWeights();
			
			for (double[] row : temp) System.out.println(Arrays.toString(row));
			System.out.println();

			System.out.println(Arrays.toString(weightLayers.get(i).getBias()));
			System.out.println();
		}


		System.out.println(loss(targets, outputs));

		/*
		System.out.println("hidden layers: ");
		for (int i = 0; i < hiddenLayers.size(); i++) {
			System.out.println(Arrays.toString(hiddenLayers.get(i).get()));
		}
		*/
	}
	
	public static double[][] addMatrix(double[][] A, double[][] B) {
		double[][] out = new double[A.length][A[0].length];

		for (int row = 0; row < out.length; row++) {
			for (int col = 0; col < out[0].length; col++) {
				out[row][col] = A[row][col] + B[row][col];
			}
		}

		return out;
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

	public static double activationFunc(double x) {
		return (1 / (1 + Math.exp(-x)));
	}

	public static double activationFuncDerivative(double x) {
		return Math.exp(-x) / Math.pow(1 + Math.exp(-x), 2);
	}

	public static void updateWeights(ArrayList<WeightLayer> weightLayers, ArrayList<double[]> targets, double eta) {
		double[][][] deltas = new double[weightLayers.size()][weightLayers.get(0).getNext().getSize()][1];
		// double[][] deltaBias = new double[weightLayers.size()][weightLayers.get(0).getNext()];

		for (int i = weightLayers.size() - 1; i >= 0; i++) {
			double[][] sigma = new double[weightLayers.get(i).getWeights().length][weightLayers.get(i).getWeights().length];
			for (int index = 0; index < sigma.length; index++) {
				sigma[index][index] = activationFuncDerivative(weightLayers.get(i).getNext().getPreActivation()[index]);
			}

			for (int n = 0; n < targets.size(); n++) {
				if (i == weightLayers.size() - 1) {
					double[][] cost = new double[weightLayers.get(0).getWeights().length][1];
					for (int index = 0; index < cost.length; index++) {
						cost[index][0] = weightLayers.get(i).getNext().getIndex(index) - targets.get(n)[index];
					}
				} else {
					double[][] cost = multiplyMatrix(transposeMatrix(weightLayers.get(i + 1).getWeights()), deltas[i + 1]);
				}

				double[][] temp = multiplyMatrix(sigma, cost);
				for (row = 0; row < deltas[0].length; row++) {
					deltas[i][row][0] += temp[row][0];
				}
			}

			// for (int col = 0; col < deltaWeights[0][0].length; col++) {
			// 	for (int row = 0; row < deltaWeights[0].length; row++) {
			// 		deltaWeights[i][row][col] *= weightLayers.get(i).getLast().getIndex(col);
			// 	}
			// }
		}

		for (int i = 0; i < deltas.length; i++) {
			for (int row = 0; row < deltas[0].length; row++) {
				deltas[i][row][0] /= targets.size();
			}
		}

		for (int i = 0; i < weightLayers.size(); i++) {
			double[][] deltaBias = deltas[i];
			for (int index = 0; index < deltaBias.length; index++) {
				deltaBias[index][0] *= eta;
			}
			weightLayers.get(i).setBias(transposeMatrix(addMatrix(weightLayers.get(i).getAllBias(), deltaBias))[0]);

			double[][] deltaWeights = multiplyMatrix(deltas[i], transposeMatrix(weightLayers.get(i).getLast().getAll()));
			for (int row = 0; row < deltaWeights.lengthl row++) {
				for (int col = 0; col < deltaWeights[0].length; col++) {
					deltaWeights[row][col] *= eta;
				}
			}
			weightLayers.get(i).setWeight(addMatrix(weightLayers.get(i).getWeights(), deltaWeights));
		}
	}

	public static double loss(ArrayList<double[]> targets, ArrayList<double[]> outputs) {
		double totalLoss = 0.0;
		
		for (int i = 0; i < targets.size(); i++) {
			double loss = 0.0;
			for (int j = 0; j < targets.get(i).length; j++) {
				loss += 0.5 * (targets.get(i)[j] - outputs.get(i)[j]) * (targets.get(i)[j] - outputs.get(i)[j]);
			}

			totalLoss += loss;
		}

		totalLoss /= targets.size();

		return totalLoss;
	}
}
