class WeightLayer {
	private double[][] weights;
	private double[][] deltas;
	private HiddenLayer last;
	private HiddenLayer next;
	
	public WeightLayer(HiddenLayer last, HiddenLayer next) {
		this.weights = new double[next.getSize()][last.getSize()];
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

	public double[][] getDeltas() {
		return deltas;
	}

	public void setDelta(int input, int output, double value) {
		deltas[output][input] = value;
	}

	public void initializeWeights() {
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[0].length; j++) {
				weights[i][j] = Math.random();
			}
		}
	}

	public void resetWeights() {
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[0].length; j++) {
				weights[i][j] = 0;
			}
		}
	}

	public void applyWeights() {
		double[] arr = Main.transposeMatrix(Main.multiplyMatrix(weights, last.getAll()))[0];
		for (int i = 0; i < arr.length; i++) {
			arr[i] = Main.activationFunc(arr[i]);
		}
		this.next.set(arr);
	}
}
