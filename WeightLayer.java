class WeightLayer {
	double[][] weights;
	HiddenLayer last;
	HiddenLayer next;
	
	public WeightLayer(HiddenLayer last, HiddenLayer next) {
		this.weights = new double[next.getSize()][last.getSize()];
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
    	weights[input][output] = value;
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
		

		this.next.set(Main.transposeMatrix(Main.multiplyMatrix(weights, last.getAll()))[0]);
	}
}
