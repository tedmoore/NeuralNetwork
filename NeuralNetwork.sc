NeuralNetwork {
	classvar activation_functions, derivative_activation_functions, activation_name2i, activation_i2name;
	classvar e = 2.71828;
	var <>net, activation_i;

	*initClass {
		activation_i2name = [
			'identity',
			'sigmoid',
			'relu',
			'tanh'
		];
		activation_name2i = IdentityDictionary.new;
		activation_i2name.do{
			arg name, i;
			activation_name2i[name] = i;
		};

		activation_functions = [
			{arg val; val}, // identity
			{arg val; (1+e.pow(val * -1)).reciprocal}, // sigmoid
			{arg val; max(0,val)}, // relu
			{arg val; tanh(val)} // tanh
		];

		derivative_activation_functions = [
			{arg val; 1.0}, // identity
			{arg val; val * (1 - val)}, // sigmoid
			{arg val; (val < 0).not.asInteger}, // relu
			{arg val; 1-val.pow(2)} // tanh
		];
	}

	*new {
		arg shape,activation = 'sigmoid';
		^super.new.init(shape,activation);
	}

	save {
		arg path;
		var save = ();
		save.net = net;
		save.activation_i = activation_i;
		save.writeArchive(path);
	}

	*load {
		arg path;
		var save, newNet;
		save = Object.readArchive(path);
		newNet = super.new.init(shape:nil,activation:activation_i2name[save.activation_i]);
		newNet.net = save.net;
		^newNet;
	}

	init {
		arg shape, activation = 'sigmoid';
		activation_i = activation_name2i[activation];

		net = shape.collect({
			arg nNeurons, i;
			var data = (
				vals:Array.fill(nNeurons,{0}),
			);
			if(i > 0,{
				// not input layer;
				data.biases = Array.fill(nNeurons,{rrand(-1.0,1.0)});
				data.weights = Array.fill(shape[i],{
					Array.fill(shape[i-1],{rrand(-1.0,1.0)});
				});
			});
			data;
		});
	}

	feedforward {
		arg input;

		net[0].vals = input;

		(1..(net.size-1)).do({
			arg layer;
			net[layer].vals.size.do({
				arg i;
				// multiply the previous layer values by the weights;
				net[layer].vals[i] = (net[layer-1].vals * net[layer].weights[i]).sum;
			});

			// add the bias
			net[layer].vals = net[layer].vals + net[layer].biases;
			// run activation
			net[layer].vals = net[layer].vals.collect({
				arg val;
				activation_functions[activation_i].(val);
			});
		});

		^net.last.vals;
	}

	train1 {
		arg inputs,targets,learnRate;

		if(((inputs.size != net[0].vals.size) || (targets.size != net.last.vals.size)).not,{
			// find all the errors
			net.last.errors = targets - this.feedforward(inputs);
			//"last layer errors: %".format(net.last.errors).postln;

			if(net.size > 2,{
				((net.size-2)..1).do({
					arg layer;
					//"layer: %".format(layer).postln;
					net[layer].errors = Array.fill(net[layer].vals.size,{0});
					//"errors: %".format(net[layer].errors).postln;

					net[layer].vals.size.do({
						arg neuron;
						var error;
						error = 0;
						net[layer+1].errors.do({
							arg nextLayerErr, i;
							error = error + (nextLayerErr * net[layer+1].weights[i][neuron]);
						});
						net[layer].errors[neuron] = error;
						//"neuron: %    error: %".format(neuron,error).postln;
					});
				});
			});

			// do gradient descent
			((net.size-1)..1).do({
				arg layer;
				//"layer: %".format(layer).postln;

				net[layer].vals.size.do({
					arg neuron;

					net[layer-1].vals.size.do({
						arg i;
						var deltaWeight = learnRate * net[layer].errors[neuron] * derivative_activation_functions[activation_i].(net[layer].vals[neuron]);

						net[layer].biases[neuron] = net[layer].biases[neuron] + deltaWeight;

						deltaWeight = deltaWeight * net[layer-1].vals[i];
						net[layer].weights[neuron][i] = net[layer].weights[neuron][i] + deltaWeight;
					});
				});
			});

			^net.last.vals;
		},{
			"input vector size: %\nNN input vector expects: %\ntarget vector size: %\nNN target vector expects: %".format(
				inputs.size,
				net[0].vals.size,
				targets.size,
				net.last.vals.size
			).postln;

			inputs.postln;
			net[0].postln;
			Error("input vector or target vector are wrong size").throw;
		});
	}

	train {
		arg trainingData, learnRate = 0.01, epochs = 100;
		var i = 0, avgError = inf;

		while({
			i < epochs;
		},{
			var totalError = 0;
			avgError = 0;
			//"a".postln;
			trainingData = trainingData.scramble;

			//trainingData.postln;

			trainingData.do({
				arg trainingExample;
				var inputVector, targetVector, output;
				inputVector = trainingExample[0];
				targetVector = trainingExample[1];

				//"input vector : %".format(inputVector).postln;
				//"target vector: %".format(targetVector).postln;

				output = this.train1(inputVector,targetVector,learnRate);
				totalError = totalError + (output-targetVector).squared.sum;
			});

			avgError = totalError / trainingData.size;

			i = i + 1;

			"epoch: %    avg error: %".format(i.asStringff(6),avgError).postln;
		});
	}


	/*	getWeightsAt {
	arg layer;
	if((layer == 0) || (layer > (net.size-1)),{
	Error("The layer provided (%) doesn't exist or doesn't have weights".format(layer)).throw;
	},{
	^net[layer].weights
	});
	}

	setWeightsAt {
	arg layer, weights;
	if((layer == 0) || (layer > (net.size-1)),{
	Error("The layer provided (%) doesn't exist or doesn't have weights".format(layer)).throw;
	},{
	if(net[layer].weights.size != weights.size,{
	Error("Provided weights has wrong number of neurons.\nExpected: %\nReceived: %\n\n".format(
	net[layer].weights.size,
	weights.size
	)).throw;
	},{
	var nWeights = net[layer].weights[0].size;
	weights.do({
	arg connections, i;
	if(connections.size != nWeights,{
	Error("A provided neuron (%) has wrong number of weights.\nExpected: %\nReceived: %\n\n".format(
	i,
	nWeights,
	connections.size
	)).throw;
	});
	});
	net[layer].weights = weights;
	});
	});
	}*/
}