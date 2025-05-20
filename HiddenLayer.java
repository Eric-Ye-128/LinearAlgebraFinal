class HiddenLayer {
	private double[][] layer;
	private double[][] preActivation;
	
	public HiddenLayer(double[] input) {
		this.layer = new double[1][input.length];
		this.layer[0] = input;
		this.layer = Main.transposeMatrix(this.layer);
		this.preActivation = new double[layer.length][layer[0].length];
	}
	
	public int getSize() {
		return layer.length;
	}

	public double[] get() {
		return Main.transposeMatrix(layer)[0];
	}

	public double[][] getAll() {
		return layer;
	}

	public void set(double[] newLayer) {
		double[][] T = Main.transposeMatrix(layer);
		T[0] = newLayer;
		layer = Main.transposeMatrix(T);
	}

	public double getIndex(int index) {
		return layer[index][0];
	}
	
	public void setIndex(int index, double value) {
		layer[index][0] = value;
	}

	public double[] getPreActivation() {
		return Main.transposeMatrix(preActivation)[0];
	}

	public void setPreActivation(double[] preActivations) {
		double[][] T = Main.transposeMatrix(preActivation);
		T[0] = preActivations;
		this.preActivation = Main.transposeMatrix(T);
	}
}
