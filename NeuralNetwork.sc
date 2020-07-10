NeuralNetworkLayer {
	classvar econst = 2.71828;
	var parent, <size, <weights, biases, <>values, activation, previousLayer, <>error;

	*new {
		arg parent, size, previousLayer, activation = "relu";
		^super.new.init(parent, size, previousLayer, activation);
	}

	init {
		arg parent_, size_, previousLayer_, activation_ = "relu";
		parent = parent_;
		size = size_;
		activation = activation_;
		previousLayer = previousLayer_;

		values = Matrix.fill(size,1,{0});

		if(previousLayer.notNil,{
			// this is not the input layer
			weights = Matrix.fill(size,previousLayer.size,{1.0.rand});
			biases = Matrix.fill(size,1,{0.5.rand2});
		});
	}

	feedForward {
		values = ((weights * previousLayer.values) + biases).collect({
			arg val;
			this.activationFunc(val);
		});
		^values;
	}

	activationFunc {
		arg val;
		activation.switch(
			"relu",{
				//"relu val: %".format(val).postln;
				^max(0,val);
			},
			"sigmoid",{
				^(1+econst.pow(val * -1)).reciprocal;
			},
			"linear",{
				^val;
			},
			"tanh",{
				^tanh(val);
			}
		);
	}

	// https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
	derivativeActivationFunc {
		arg val;
		activation.switch(
			"relu",{
				var return;
				//"d relu val: %".format(val).postln;
				if(val < 0,{return = 0.0},{return = 1.0});
				^return;
			},
			"sigmoid",{
				^(val * (1 - val));
			},
			"linear",{
				^1.0;
			},
			"tanh",{
				^(1-val.pow(2))
			}
		);
	}

	backProp {
		var gradient = values.collect({
			arg val, row, col;
			this.derivativeActivationFunc(val) * error.at(row,col) * parent.learningRate;
		});

		weights = weights + (gradient * previousLayer.values.flop);
		biases = biases + gradient;
	}
}

NeuralNetwork {
	var inputSize, layers, <learningRate;

	*new {
		arg inputSize, learningRate = 0.1;
		^super.new.init(inputSize,learningRate);
	}

	init {
		arg inputSize_, learningRate_ = 0.1;
		inputSize = inputSize_;
		learningRate = learningRate_;

		layers = List.new;
		layers.add(NeuralNetworkLayer(this,inputSize));
	}

	addLayer {
		arg size, activation;
		layers.add(NeuralNetworkLayer(this,size,layers.last,activation));
	}

	feedForward {
		arg in;
		var out;
		layers[0].values_(Matrix.withFlatArray(in.size,1,in));

		layers[1..].do({
			arg layer;
			out = layer.feedForward;
		});

		^out;
	}

	train1 {
		arg inputs, targets;
		var return_e;
		targets = Matrix.withFlatArray(targets.size,1,targets);

		// calc errors
		layers.last.error = targets - this.feedForward(inputs);
		((layers.size-2)..1).do({
			arg layerI;
			var layer = layers[layerI];
			layer.error = layers[layerI + 1].weights.flop * layers[layerI + 1].error;
		});

		// back prop
		((layers.size-1)..1).do({
			arg layerI;
			layers[layerI].backProp;
		});
		return_e = layers.last.error.flatten;
		//return_e.postln;
		^(return_e.pow(2).sum / return_e.size);
	}

	train {
		arg trainingData, nEpochs;
		nEpochs.do({
			arg epoch;
			var err = 0;
			trainingData.scramble.do({
				arg trainingPair;
				err = err + this.train1(trainingPair[0],trainingPair[1]);
			});
			"epoch: %".format(epoch).postln;
			"error: %\n".format(err).postln;
		});
	}

	trainAndTest {
		arg trainingData, trainPercent;
		var trainingN = (trainingData.size * trainPercent).floor.asInteger;
		var trainingSet = trainingData[0..(trainingN-1)];
		var testingSet = trainingData[trainingN..];
		var nCorrect = 0;
		this.train(trainingSet);
		testingSet.do({
			arg testingPair;
			if(this.feedForward(testingPair[0]).maxIndex == testingPair[1].maxIndex,{
				nCorrect = nCorrect + 1;
			});
		});
		^(nCorrect / testingSet.size);
	}
}