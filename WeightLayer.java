class WeightLayer {
	double[][] weights;
	HiddenLayer last;
	HiddenLayer next;
	
	public WeightLayer(HiddenLayer last, double[][] weights, HiddenLayer next) {
		this.weights = weights;
		this.last = last;
		this.next = next;
	}
	
	public getLast() {
		return last;
	}
	
	public setLast(HiddenLayer newLast) {
		last = newLast;
	}
	
	public getNext() {
		return Next;
	}
	
	public setLast(HiddenLayer newNext) {
		next = newNext;
	}
	
	public getWeights() {
		return weights;
	}
	
	public setWeights(int input, int output, double value) {
    	weights[input][output] = value;
	}
}
