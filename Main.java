import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.util.*;

class Main {
	public static void main(String[] args) {
		double eta = 0.01;	// the learning rate
		int epoch = 100;	// number of passes through training set

		// inputs and outputs
		ArrayList<int[][]> images = new ArrayList<int[][]>();		// training set images as matrices
		ArrayList<double[]> inputs = new ArrayList<double[]>();		// flattened training set images
		ArrayList<double[]> targets = new ArrayList<double[]>();	// desired outputs
		ArrayList<double[]> outputs = new ArrayList<double[]>();	// actual outputs

		// obtains PNG files from Training set
		File folder = new File("Training set");
		File[] listOfFiles = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".png"));

		// checks if training set is empty
		if (listOfFiles == null || listOfFiles.length == 0) {
			System.out.println("No training PNGs found.");
			return;
		}

		// sorts training set
		Arrays.sort(listOfFiles, (f1, f2) -> {
			String name1 = f1.getName().replaceAll("\\D+", "");
			String name2 = f2.getName().replaceAll("\\D+", "");
			int num1 = name1.isEmpty() ? 0 : Integer.parseInt(name1);
			int num2 = name2.isEmpty() ? 0 : Integer.parseInt(name2);
			return Integer.compare(num1, num2);
		});

		// turns images to gray scale
		for (File file : listOfFiles) {
			try {
				images.add(convertGrayScale(ImageIO.read(file)));
			} catch (Exception e) {
				System.out.println("Error processing file " + file.getName());
				e.printStackTrace();
				return;
			}
		}

		// properly formats the images to have the same size and bounded values
		inputs = formatImages(images);

		// obtains desired outputs from Training set
		File file = null;
		Scanner scan = null;
		String[] line = null;
		File[] targetFiles = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".txt"));
		try {
			file = targetFiles[0];
			scan = new Scanner(file);
		} catch (Exception e) {
			System.out.println("FIle not found");
            e.printStackTrace();
			return;
		}

		// categorizes possible targets
		String[] targetTypes = scan.nextLine().split(" ");
		scan.nextLine();
		while (scan.hasNext()) {
			double[] target = new double[targetTypes.length];
			String targetType = scan.nextLine();
			for (int i = 0; i < target.length; i++) {
				if (targetType.equals(targetTypes[i])) {
					target[i] = 1.0;
					break;
				}
			}
			targets.add(target);
		}
		
		// creating hidden layers
		ArrayList<HiddenLayer> hiddenLayers = new ArrayList<HiddenLayer>();
		int hiddenCount = 5, hiddenSize = 10;
		for (int i = 0; i < hiddenCount; i++) {
			hiddenLayers.add(new HiddenLayer(new double[hiddenSize]));
		}

		// creating weight layers
		ArrayList<WeightLayer> weightLayers = new ArrayList<WeightLayer>();
		weightLayers.add(new WeightLayer(new HiddenLayer(new double[inputs.get(0).length]), hiddenLayers.get(0)));
		for (int i = 0; i < hiddenCount - 1; i++) {
			weightLayers.add(new WeightLayer(hiddenLayers.get(i), hiddenLayers.get(i + 1)));
		}
		weightLayers.add(new WeightLayer(hiddenLayers.get(hiddenLayers.size() - 1), new HiddenLayer(new double[targets.get(0).length])));

		// randomizing initial weights
		for (int i = 0; i < weightLayers.size(); i++) weightLayers.get(i).initialize();

		// passes through training set
		for (int trial = 0; trial < epoch; trial++) {
			outputs.clear();

			// feedfowards
			for (int i = 0; i < inputs.size(); i++) {
				weightLayers.get(0).getLast().set(inputs.get(i));
				for (int j = 0; j < weightLayers.size(); j++) {
					weightLayers.get(j).applyWeights();
				}
				outputs.add(weightLayers.get(weightLayers.size() - 1).getNext().get());
			}

			if ((trial + 1) % (epoch / 5) == 0) {
				System.out.println("Error at epoch " + (trial + 1) + "/" + epoch + ": " + loss(targets, outputs));
			}
			
			// backpropagation
			// if (trial > 10) eta *= Math.exp(-0.5);
			weightLayers = backpropagation(weightLayers, targets, eta);
		}

		System.out.println("\nFinal error: " + loss(targets, outputs) + "\n");
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

	// the sigmoid function is used as the activation function
	public static double activationFunc(double x) {
		return (1 / (1 + Math.exp(-x)));
	}

	public static double activationFuncDerivative(double x) {
		return Math.exp(-x) / Math.pow(1 + Math.exp(-x), 2);
	}

	public static ArrayList<WeightLayer> backpropagation(ArrayList<WeightLayer> weightLayers, ArrayList<double[]> targets, double eta) {
		// initializes deltas
		ArrayList<double[][]> deltas = new ArrayList<>();
		for (int i = 0; i < weightLayers.size(); i++) {
			deltas.add(new double[weightLayers.get(i).getNext().getSize()][1]);
		}

		for (int i = weightLayers.size() - 1; i >= 0; i--) {
			// finds change in neuron output wrt neuron inputs
			double[][] sigma = new double[weightLayers.get(i).getWeights().length][weightLayers.get(i).getWeights().length];
			for (int index = 0; index < sigma.length; index++) {
				sigma[index][index] = activationFuncDerivative(weightLayers.get(i).getNext().getPreActivation()[index]);
			}

			// finds change in error wrt neuron outputs
			for (int n = 0; n < targets.size(); n++) {
				double[][] cost = null;

				// checks if current layer contains inner or output neurons
				if (i == weightLayers.size() - 1) {
					cost = new double[targets.get(0).length][1];
					for (int index = 0; index < cost.length; index++) {
						cost[index][0] = weightLayers.get(i).getNext().getIndex(index) - targets.get(n)[index];
					}
				} else {
					cost = multiplyMatrix(transposeMatrix(weightLayers.get(i + 1).getWeights()), deltas.get(i + 1));
				}

				deltas.set(i, addMatrix(deltas.get(i), multiplyMatrix(sigma, cost)));
			}
		}

		// averages the deltas across entire training set
		for (int i = 0; i < deltas.size(); i++) {
			for (int row = 0; row < deltas.get(i).length; row++) {
				deltas.get(i)[row][0] /= targets.size();
			}
		}

		for (int i = 0; i < weightLayers.size(); i++) {
			// applies learning rate and updates biases
			double[][] deltaBias = deltas.get(i);
			for (int index = 0; index < deltaBias.length; index++) {
				deltaBias[index][0] *= -eta;
			}
			weightLayers.get(i).setBias(transposeMatrix(addMatrix(weightLayers.get(i).getAllBias(), deltaBias))[0]);

			// finds change in neuron inputs wrt weights and applies learning rate, then updates weights
			double[][] deltaWeights = multiplyMatrix(deltas.get(i), transposeMatrix(weightLayers.get(i).getLast().getAll()));
			for (int row = 0; row < deltaWeights.length; row++) {
				for (int col = 0; col < deltaWeights[0].length; col++) {
					deltaWeights[row][col] *= -eta;
				}
			}
			weightLayers.get(i).setWeight(addMatrix(weightLayers.get(i).getWeights(), deltaWeights));
		}

		return weightLayers;
	}

	// the square error function is used as the loss function
	public static double loss(ArrayList<double[]> targets, ArrayList<double[]> outputs) {
		double totalLoss = 0.0;
		
		for (int i = 0; i < targets.size(); i++) {
			double loss = 0.0;
			for (int j = 0; j < targets.get(i).length; j++) {
				loss += 0.5 * (targets.get(i)[j] - outputs.get(i)[j]) * (targets.get(i)[j] - outputs.get(i)[j]);
			}

			totalLoss += loss;
		}

		return (totalLoss /= targets.size());
	}

	public static int[][] convertGrayScale(BufferedImage image) {
		int[][] grayScale = new int[image.getHeight()][image.getWidth()];

		for (int i = 0; i < grayScale.length; i++) {
			for (int j = 0; j < grayScale[0].length; j++) {
				Color color = new Color(image.getRGB(j, i));
				grayScale[i][j] = (int) (0.299 * color.getRed() + 0.587 * color.getGreen() + 0.114 * color.getBlue());
			}
		}

		return grayScale;
	}

	public static ArrayList<double[]> formatImages(ArrayList<int[][]> images) {
		int maxRows = 0;
        int maxCols = 0;

		// finds largest image dimensions
        for (int[][] image : images) {
            maxRows = Math.max(maxRows, image.length);
            maxCols = Math.max(maxCols, image[0].length);
        }

        ArrayList<double[]> out = new ArrayList<>();
        for (int[][] image : images) {
			// pads the edges of images with 0s
			int[][] padded = new int[maxRows][maxCols];
	        for (int i = 0; i < image.length; i++) {
    	        System.arraycopy(image[i], 0, padded[(maxRows - image.length) / 2 + i], (maxCols - image[0].length) / 2, image[i].length);
        	}

			// flattens the images and bounds entries to be from 0 to 1
			double[] formatted = new double[padded.length * padded[0].length];
			for (int i = 0; i < formatted.length; i++) {
				formatted[i] = padded[i / padded[0].length][i % padded[0].length] /= 255.0;
			}

			out.add(formatted);
        }

        return out;
	}
}
