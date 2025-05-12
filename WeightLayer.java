class WeightLayer {
	private double[][] weights;
	private double[][] bias;
	private double[][] deltas;
	private HiddenLayer last;
	private HiddenLayer next;
	
	public WeightLayer(HiddenLayer last, HiddenLayer next) {
		this.weights = new double[next.getSize()][last.getSize()];
		this.bias = new double[next.getSize()][1];
		this.deltas = new double[weights.length][weights[0].length];
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

	public double[][] getDeltas() {
		return deltas;
	}

	public void setDelta(int input, int output, double value) {
		deltas[output][input] = value;
	}

	public void setDelta(double[][] newDelta) {
		deltas = newDelta;
	}

	public void initialize() {
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[0].length; j++) {
				weights[i][j] = Math.random() - 0.5;
			}
		}

		for (int i = 0; i < bias.length; i++) {
			bias[i][0] = Math.random() - 0.5;
		}
	}

	public void reset() {
		this.weights = new double[next.getSize()][last.getSize()];
		this.bias = new double[next.getSize()][1];
	}

	public void applyWeights() {
		double[] arr = Main.transposeMatrix(Main.addMatrix(Main.multiplyMatrix(weights, last.getAll()), bias))[0];
		for (int i = 0; i < arr.length; i++) arr[i] = Main.activationFunc(arr[i]);
		this.next.set(arr);
	}
}
