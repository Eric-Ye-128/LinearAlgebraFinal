import java.util.*;

class WeightLayer {
	private double[][] weights;
	private double[][] bias;
	private HiddenLayer last;
	private HiddenLayer next;
	
	public WeightLayer(HiddenLayer last, HiddenLayer next) {
		this.weights = new double[next.getSize()][last.getSize()];
		this.bias = new double[next.getSize()][1];
		this.last = last;
		this.next = next;
	}
	
	public HiddenLayer getLast() {
		return last;
	}
	
	public void setLast(HiddenLayer newLast) {
		last = newLast;
	}
	
	public HiddenLayer getNext() {
		return next;
	}
	
	public void setNext(HiddenLayer newNext) {
		next = newNext;
	}
	
	public double[][] getWeights() {
		return weights;
	}
	
	public void setWeight(int input, int output, double value) {
    	weights[output][input] = value;
	}

	public void setWeight(double[][] newWeights) {
		weights = newWeights;
	}

	public double getBias(int index) {
		return bias[index][0];
	}

	public double[] getBias() {
		return Main.transposeMatrix(bias)[0];
	}

	public double[][] getAllBias() {
		return bias;
	}

	public void setBias(int index, double value) {
		bias[index][0] = value;
	}

	public void setBias(double[] newBias) {
		double[][] T = Main.transposeMatrix(bias);
		T[0] = newBias;
		bias = Main.transposeMatrix(T);
	}

	public void initialize() {
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[0].length; j++) {
				weights[i][j] = gaussian() / Math.sqrt(next.getSize());
			}
		}

		for (int i = 0; i < bias.length; i++) {
			bias[i][0] = gaussian();
		}
	}

	public void reset() {
		this.weights = new double[next.getSize()][last.getSize()];
		this.bias = new double[next.getSize()][1];
	}

	public void applyWeights() {
		double[] arr = Main.transposeMatrix(Main.addMatrix(Main.multiplyMatrix(weights, last.getAll()), bias))[0];
		this.next.setPreActivation(arr);
		for (int i = 0; i < arr.length; i++) arr[i] = Main.activationFunc(arr[i]);
		this.next.set(arr);
	}

	public double gaussian() {
		double v1, v2, s;
    	do {
       		v1 = 2 * Math.random() - 1;
       		v2 = 2 * Math.random() - 1;
       		s = v1 * v1 + v2 * v2;
     	} while (s >= 1 || s == 0);

     	return v1 * StrictMath.sqrt(-2 * StrictMath.log(s) / s);
	}
}
