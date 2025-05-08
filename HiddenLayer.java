class HiddenLayer {
	int[] layer;
	
	public HiddenLayer(int[] layer) {
		this.layer = layer;
	}
	
	public getIndex(int index) {
		return layer[index];
	}
	
	public setIndex(int index, int value) {
		layer[index] = value;
	}
}
