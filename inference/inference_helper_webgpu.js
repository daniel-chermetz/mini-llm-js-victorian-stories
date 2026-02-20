globalThis.readFloat32Buffer = async function(ctx, srcBuffer, byteSize) {
  const readBuffer = ctx.webgpuDevice.createBuffer({
    size: byteSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = ctx.webgpuDevice.createCommandEncoder();
  encoder.copyBufferToBuffer(
    srcBuffer,
    0,
    readBuffer,
    0,
    byteSize
  );

  ctx.webgpuDevice.queue.submit([encoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);

  const arrayBuffer = readBuffer.getMappedRange();
  const result = new Float32Array(arrayBuffer.slice(0));

  readBuffer.unmap();
  readBuffer.destroy();

  return result;
}

globalThis.initTransformerBuffers = function(ctx, dimensions, heads, L, ffnDimMultiplier) {
	const F = Float32Array.BYTES_PER_ELEMENT;
	const dimL = dimensions * L * F;
	const headL = heads * L * F;
	const headLL = heads * L * L * F;
	const ffnDim = dimensions * ffnDimMultiplier;
	const ffnL = ffnDim * L * F;

	const makeBuf = (size) => ctx.webgpuDevice.createBuffer({
		size,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});

	ctx.preBuffers = {
		rms1: makeBuf(dimL),
		rms2: makeBuf(dimL),
		rms3: makeBuf(dimL),
		matMulV: makeBuf(dimL),
		matMulK: makeBuf(dimL),
		matMulQ: makeBuf(dimL),
		ropeK: makeBuf(dimL),
		ropeQ: makeBuf(dimL),
		ktq: makeBuf(headLL),
		colMax: makeBuf(headL),
		colSum: makeBuf(headL),
		softmax: makeBuf(headLL),
		valsAttention: makeBuf(dimL),
		outputProj: makeBuf(dimL),
		residual1: makeBuf(dimL),
		ffn1a: makeBuf(ffnL),
		silu: makeBuf(ffnL),
		ffn1b: makeBuf(ffnL),
		hadamard: makeBuf(ffnL),
		ffn2: makeBuf(dimL),
		residual2: makeBuf(dimL),
	};
};

globalThis.executeDimDimWeightLoading = function(ctx, flatInput, dimensions, shaderCode) {
	if (!ctx.dimDim_weight_loading_pipeline) {
		const dimDimShaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });

		ctx.dimDim_weight_loading_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: "auto",
			compute: { module: dimDimShaderModule, entryPoint: 'main' },
		});
	}

	const inputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	ctx.webgpuDevice.queue.writeBuffer(inputBuffer, 0, flatInput);

	const outputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.dimDim_weight_loading_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: outputBuffer } },
		],
	});

	const commandEncoder = ctx.webgpuDevice.createCommandEncoder();
	const passEncoder = commandEncoder.beginComputePass();
	passEncoder.setPipeline(ctx.dimDim_weight_loading_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(dimensions / 8);
	const workgroupsY = Math.ceil(dimensions / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);
	passEncoder.end();

	ctx.webgpuDevice.queue.submit([commandEncoder.finish()]);

	return {
		buffer: outputBuffer,
	};
};

globalThis.executeRmsWeightLoading = function(ctx, flatInput, dimensions, shaderCode) {
	if (!ctx.rms_weight_loading_pipeline) {
		const rmsShaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
		
		ctx.rms_weight_loading_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: rmsShaderModule, entryPoint: 'main' },
		});
	}

	const inputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	ctx.webgpuDevice.queue.writeBuffer(inputBuffer, 0, flatInput);

	const outputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.rms_weight_loading_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: outputBuffer } },
		],
	});

	const commandEncoder = ctx.webgpuDevice.createCommandEncoder();
	const passEncoder = commandEncoder.beginComputePass();
	passEncoder.setPipeline(ctx.rms_weight_loading_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroups = Math.ceil(dimensions / 64);
	passEncoder.dispatchWorkgroups(workgroups);
	passEncoder.end();

	ctx.webgpuDevice.queue.submit([commandEncoder.finish()]);

	return {
		buffer: outputBuffer,
	};
};

globalThis.executeFfn1WeightLoading = function(ctx, flatInput, numRows, numCols, shaderCode) {
	if (!ctx.ffn1_weight_loading_pipeline) {
		const ffn1ShaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.ffn1_weight_loading_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: ffn1ShaderModule, entryPoint: 'main' },
		});
	}

	const inputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	ctx.webgpuDevice.queue.writeBuffer(inputBuffer, 0, flatInput);

	const outputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.ffn1_weight_loading_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: outputBuffer } },
		],
	});

	const commandEncoder = ctx.webgpuDevice.createCommandEncoder();
	const passEncoder = commandEncoder.beginComputePass();
	passEncoder.setPipeline(ctx.ffn1_weight_loading_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(numCols / 8);
	const workgroupsY = Math.ceil(numRows / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);
	passEncoder.end();

	ctx.webgpuDevice.queue.submit([commandEncoder.finish()]);
	
	return {
		buffer: outputBuffer,
	};
};

globalThis.executeFfn2WeightLoading = function(ctx, flatInput, numRows, numCols, shaderCode) {
	if (!ctx.ffn2_weight_loading_pipeline) {
		const ffn2ShaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
			
		ctx.ffn2_weight_loading_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: ffn2ShaderModule, entryPoint: 'main' },
		});
	}

	const inputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	ctx.webgpuDevice.queue.writeBuffer(inputBuffer, 0, flatInput);

	const outputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.ffn2_weight_loading_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: outputBuffer } },
		],
	});

	const commandEncoder = ctx.webgpuDevice.createCommandEncoder();
	const passEncoder = commandEncoder.beginComputePass();
	passEncoder.setPipeline(ctx.ffn2_weight_loading_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(numCols / 8);
	const workgroupsY = Math.ceil(numRows / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);
	passEncoder.end();

	ctx.webgpuDevice.queue.submit([commandEncoder.finish()]);

	return {
		buffer: outputBuffer,
	};
};

globalThis.executeTokenEmbeddingsWeightLoading = function(ctx, flatInput, numRows, numCols, shaderCode) {
	if (!ctx.token_embeddings_weight_loading_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
		
		ctx.token_embeddings_weight_loading_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const inputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	ctx.webgpuDevice.queue.writeBuffer(inputBuffer, 0, flatInput);

	const outputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.token_embeddings_weight_loading_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: outputBuffer } },
		],
	});

	const commandEncoder = ctx.webgpuDevice.createCommandEncoder();
	const passEncoder = commandEncoder.beginComputePass();
	passEncoder.setPipeline(ctx.token_embeddings_weight_loading_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(numCols / 8);
	const workgroupsY = Math.ceil(numRows / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);
	passEncoder.end();

	ctx.webgpuDevice.queue.submit([commandEncoder.finish()]);
	
	return {
		buffer: outputBuffer,
	};
};

globalThis.executeSetTeacherMode = function(ctx, teacherMode) {
	if (!ctx.teacherModeBuffer) {
		ctx.teacherModeBuffer = ctx.webgpuDevice.createBuffer({
			size: Uint32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});
	}
	const teacherModeVal = teacherMode ? 1 : 0;
	ctx.webgpuDevice.queue.writeBuffer(ctx.teacherModeBuffer, 0, new Uint32Array([teacherModeVal]));

	return {
		buffer: ctx.teacherModeBuffer,
	};
};

globalThis.executeSetRightEndIndex = function(ctx, rightEndIndex) {
	if (!ctx.rightEndIndexBuffer) {
		ctx.rightEndIndexBuffer = ctx.webgpuDevice.createBuffer({
			size: Uint32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});
	}
	ctx.webgpuDevice.queue.writeBuffer(ctx.rightEndIndexBuffer, 0, new Uint32Array([rightEndIndex]));

	return {
		buffer: ctx.rightEndIndexBuffer,
	};
};

globalThis.executeSetInputTokenEmbeddings = function(ctx, indicesArray, tokenEmbeddingsBuffer, dimensions, L, shaderCode) {	
	if (!ctx.executeSetInputTokenEmbeddings_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeSetInputTokenEmbeddings_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});

		ctx.indicesBuffer_executeSetInputTokenEmbeddings = ctx.webgpuDevice.createBuffer({
			size: L * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

		ctx.outputBuffer_executeSetInputTokenEmbeddings = ctx.webgpuDevice.createBuffer({
			size: dimensions * L * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		ctx.bindGroup_executeSetInputTokenEmbeddings = ctx.webgpuDevice.createBindGroup({
			layout: ctx.executeSetInputTokenEmbeddings_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: ctx.indicesBuffer_executeSetInputTokenEmbeddings } },
				{ binding: 1, resource: { buffer: tokenEmbeddingsBuffer } },
				{ binding: 2, resource: { buffer: ctx.rightEndIndexBuffer } },
				{ binding: 3, resource: { buffer: ctx.outputBuffer_executeSetInputTokenEmbeddings } },
			],
		});	
	}

	ctx.webgpuDevice.queue.writeBuffer(ctx.indicesBuffer_executeSetInputTokenEmbeddings, 0, indicesArray);

	passEncoder.setPipeline(ctx.executeSetInputTokenEmbeddings_pipeline);
	passEncoder.setBindGroup(0, ctx.bindGroup_executeSetInputTokenEmbeddings);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(dimensions / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return { buffer: ctx.outputBuffer_executeSetInputTokenEmbeddings };
};

globalThis.executeRMSNorm = function(ctx, xInputsBuffer, rmsGammaBuffer, dimensions, L, shaderCode, rmsOutputBuffer) {
	if (!ctx.executeRMSNorm_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
		
		ctx.executeRMSNorm_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeRMSNorm_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: xInputsBuffer } },
			{ binding: 1, resource: { buffer: rmsGammaBuffer } },
			{ binding: 2, resource: { buffer: ctx.rightEndIndexBuffer } },
			{ binding: 3, resource: { buffer: rmsOutputBuffer } },
		],
	});

	passEncoder.setPipeline(ctx.executeRMSNorm_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(dimensions / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return { buffer: rmsOutputBuffer };
};

globalThis.executeMatMul_dim_L_dim_dim = function(ctx, aBuffer, bBuffer, dimensions, L, shaderCode, outputBuffer) {
	if (!ctx.executeMatMul_Dim_L_Dim_Dim_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
		
		ctx.executeMatMul_Dim_L_Dim_Dim_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeMatMul_Dim_L_Dim_Dim_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: aBuffer } },
			{ binding: 1, resource: { buffer: bBuffer } },
			{ binding: 2, resource: { buffer: ctx.rightEndIndexBuffer } },			
			{ binding: 3, resource: { buffer: outputBuffer } },
		],
	});

	passEncoder.setPipeline(ctx.executeMatMul_Dim_L_Dim_Dim_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(dimensions / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return { buffer: outputBuffer };
};

globalThis.executeLoadPrecomputedTheta = function(ctx, flatData, rows, cols) {
	const bufferSize = flatData.byteLength;
	const outputBuffer = ctx.webgpuDevice.createBuffer({
		size: bufferSize,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
	});
	
	ctx.webgpuDevice.queue.writeBuffer(outputBuffer, 0, flatData);
	
	return {
		buffer: outputBuffer,
	};
};

globalThis.executeRoPE = function(ctx, inputBuffer, thetaBuffer, headDim, heads, L, shaderCode, ropeOutputBuffer) {
	if (!ctx.executeRoPE_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeRoPE_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeRoPE_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: thetaBuffer } },
			{ binding: 2, resource: { buffer: ctx.rightEndIndexBuffer } },
			{ binding: 3, resource: { buffer: ropeOutputBuffer } },
		],
	});

	passEncoder.setPipeline(ctx.executeRoPE_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(headDim / 8);
	const workgroupsZ = heads;
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);

	return {
		buffer: ropeOutputBuffer,
	};
};

globalThis.executeKtQ = function(ctx, heads, shaderCode) {
	if (!ctx.executeKtQ_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeKtQ_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});

		ctx.executeKtQ_bindGroup = ctx.webgpuDevice.createBindGroup({
			layout: ctx.executeKtQ_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: ctx.preBuffers.ropeK } },
				{ binding: 1, resource: { buffer: ctx.preBuffers.ropeQ } },
				{ binding: 2, resource: { buffer: ctx.rightEndIndexBuffer } },
				{ binding: 3, resource: { buffer: ctx.preBuffers.ktq } },
			],
		});
	}

	passEncoder.setPipeline(ctx.executeKtQ_pipeline);
	passEncoder.setBindGroup(0, ctx.executeKtQ_bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(globalThis.LSequence / 8);
	const workgroupsZ = heads;
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);

	return { buffer: ctx.preBuffers.ktq };
};

globalThis.executeColMax = (ctx, heads, L, shaderCode) => {
	if (!ctx.executeColMax_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeColMax_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});

		ctx.executeColMax_bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeColMax_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: ctx.preBuffers.ktq } },
				{ binding: 1, resource: { buffer: ctx.rightEndIndexBuffer } },			
				{ binding: 2, resource: { buffer: ctx.preBuffers.colMax } },
			],
		});
	}

	passEncoder.setPipeline(ctx.executeColMax_pipeline);
	passEncoder.setBindGroup(0, ctx.executeColMax_bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(L * heads / 32));

	return { buffer: ctx.preBuffers.colMax };
}

globalThis.executeColSum = (ctx, heads, L, shaderCode) => {
	if (!ctx.executeColSum_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeColSum_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});

		ctx.executeColSum_bindGroup = ctx.webgpuDevice.createBindGroup({
			layout: ctx.executeColSum_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: ctx.preBuffers.ktq } },
				{ binding: 1, resource: { buffer: ctx.preBuffers.colMax } },
				{ binding: 2, resource: { buffer: ctx.rightEndIndexBuffer } },			
				{ binding: 3, resource: { buffer: ctx.preBuffers.colSum } },
			],
		});
	}

	passEncoder.setPipeline(ctx.executeColSum_pipeline);
	passEncoder.setBindGroup(0, ctx.executeColSum_bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(L * heads / 32));

	return { buffer: ctx.preBuffers.colSum };
}

globalThis.executeSoftmaxByHead = (ctx, heads, shaderCode) => {
	if (!ctx.executeSoftmaxByHead_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeSoftmaxByHead_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});

		ctx.executeSoftmaxByHead_bindGroup = ctx.webgpuDevice.createBindGroup({
			layout: ctx.executeSoftmaxByHead_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: ctx.preBuffers.ktq } },
				{ binding: 1, resource: { buffer: ctx.preBuffers.colMax } },
				{ binding: 2, resource: { buffer: ctx.preBuffers.colSum } },
				{ binding: 3, resource: { buffer: ctx.rightEndIndexBuffer } },				
				{ binding: 4, resource: { buffer: ctx.preBuffers.softmax } },
			],
		});
	}

	passEncoder.setPipeline(ctx.executeSoftmaxByHead_pipeline);
	passEncoder.setBindGroup(0, ctx.executeSoftmaxByHead_bindGroup);
	passEncoder.dispatchWorkgroups(
		Math.ceil(globalThis.LSequence / 8), 
		Math.ceil(globalThis.LSequence / 8), 
		heads
	);

	return { buffer: ctx.preBuffers.softmax };
};

globalThis.executeMatMulValsAttention = (ctx, headDim, heads, shaderCode) => {
	if (!ctx.executeMatMulValsAttention_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
		
		ctx.executeMatMulValsAttention_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});

		ctx.executeMatMulValsAttention_bindGroup = ctx.webgpuDevice.createBindGroup({
			layout: ctx.executeMatMulValsAttention_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: ctx.preBuffers.matMulV } },
				{ binding: 1, resource: { buffer: ctx.preBuffers.softmax } },
				{ binding: 2, resource: { buffer: ctx.preBuffers.valsAttention } },
			],
		});
	}

	passEncoder.setPipeline(ctx.executeMatMulValsAttention_pipeline);
	passEncoder.setBindGroup(0, ctx.executeMatMulValsAttention_bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(headDim / 8);
	const workgroupsZ = heads;
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);

	return { buffer: ctx.preBuffers.valsAttention };
};

// TODO: clean signature, some params not longer needed
globalThis.executeElementWiseAdd = (ctx, aBuffer, bBuffer, dimensions, L, shaderCode, outputBuffer) => {
	if (!ctx.executeElementWiseAdd_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
		
		ctx.executeElementWiseAdd_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeElementWiseAdd_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: aBuffer } },
			{ binding: 1, resource: { buffer: bBuffer } },
			{ binding: 2, resource: { buffer: outputBuffer } },
		],
	});

	passEncoder.setPipeline(ctx.executeElementWiseAdd_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(dimensions / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, 1);

	return { buffer: outputBuffer };
};

// TODO: clean signature, some params not longer needed
globalThis.executeMatMulFFN1 = (ctx, weightsBuffer, inputBuffer, ffnDim, dimensions, L, shaderCode, ffnOutputBuffer) => {
	if (!ctx.executeMatMulFFN1_pipeline) {	
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
		
		ctx.executeMatMulFFN1_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeMatMulFFN1_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: weightsBuffer } },
			{ binding: 1, resource: { buffer: inputBuffer } },
			{ binding: 2, resource: { buffer: ffnOutputBuffer } },
		],
	});

	passEncoder.setPipeline(ctx.executeMatMulFFN1_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(ffnDim / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, 1);

	return { buffer: ffnOutputBuffer };
};

// TODO: clean signature, some params not longer needed
globalThis.executeSilu = (ctx, inputBuffer, ffnDim, L, shaderCode) => {
	if (!ctx.executeSilu_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeSilu_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});

		ctx.executeSilu_bindGroup = ctx.webgpuDevice.createBindGroup({
			layout: ctx.executeSilu_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: ctx.preBuffers.ffn1a } },
				{ binding: 1, resource: { buffer: ctx.preBuffers.silu } },
			],
		});
	}

	passEncoder.setPipeline(ctx.executeSilu_pipeline);
	passEncoder.setBindGroup(0, ctx.executeSilu_bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(ffnDim / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return { buffer: ctx.preBuffers.silu };
};

// TODO: clean signature, some params not longer needed
globalThis.executeHadamard = (ctx, aBuffer, bBuffer, ffnDim, L, shaderCode) => {
	if (!ctx.executeHadamard_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeHadamard_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});

	 	ctx.executeHadamard_bindGroup = ctx.webgpuDevice.createBindGroup({
			layout: ctx.executeHadamard_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: ctx.preBuffers.silu } },
				{ binding: 1, resource: { buffer: ctx.preBuffers.ffn1b } },
				{ binding: 2, resource: { buffer: ctx.preBuffers.hadamard } },
			],
		});		
	}

	passEncoder.setPipeline(ctx.executeHadamard_pipeline);
	passEncoder.setBindGroup(0, ctx.executeHadamard_bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(ffnDim / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return { buffer: ctx.preBuffers.hadamard };
};

// TODO: clean signature, some params not longer needed
globalThis.executeMatMulFFN2 = (ctx, weightsBuffer, inputBuffer, dimensions, ffnDim, L, shaderCode) => {
	if (!ctx.executeMatMulFFN2_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeMatMulFFN2_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	ctx.executeMatMulFFN2_bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeMatMulFFN2_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: weightsBuffer } },
			{ binding: 1, resource: { buffer: ctx.preBuffers.hadamard } },
			{ binding: 2, resource: { buffer: ctx.preBuffers.ffn2 } },
		],
	});

	passEncoder.setPipeline(ctx.executeMatMulFFN2_pipeline);
	passEncoder.setBindGroup(0, ctx.executeMatMulFFN2_bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(dimensions / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return { buffer: ctx.preBuffers.ffn2 };
};

// TODO: clean signature, some params not longer needed
globalThis.executeMatMulVocab = (ctx, embeddingsBuffer, inputBuffer, vocabSize, dimensions, L, shaderCode) => {
	if (!ctx.executeMatMulVocab_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeMatMulVocab_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});

		ctx.outputBuffer_executeMatMulVocab = ctx.webgpuDevice.createBuffer({
			size: vocabSize * L * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		ctx.bindGroup_executeMatMulVocab = ctx.webgpuDevice.createBindGroup({
			layout: ctx.executeMatMulVocab_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: embeddingsBuffer } },
				{ binding: 1, resource: { buffer: ctx.preBuffers.rms3 } },
				{ binding: 2, resource: { buffer: ctx.teacherModeBuffer } },
				{ binding: 3, resource: { buffer: ctx.rightEndIndexBuffer } },
				{ binding: 4, resource: { buffer: ctx.outputBuffer_executeMatMulVocab } },
			],
		});
	}

	passEncoder.setPipeline(ctx.executeMatMulVocab_pipeline);
	passEncoder.setBindGroup(0, ctx.bindGroup_executeMatMulVocab);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(vocabSize / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return { buffer: ctx.outputBuffer_executeMatMulVocab };
};

globalThis.executeLogitSoftmax = (ctx, logitsBuffer, vocabSize, L, shaderCode) => {
	if (!ctx.executeLogitSoftmax_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeLogitSoftmax_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});

		ctx.outputBuffer_executeLogitSoftmax = ctx.webgpuDevice.createBuffer({
			size: vocabSize * L * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});
		
		ctx.bindGroup_executeLogitSoftmax = ctx.webgpuDevice.createBindGroup({
			layout: ctx.executeLogitSoftmax_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: logitsBuffer } },
				{ binding: 1, resource: { buffer: ctx.teacherModeBuffer } },
				{ binding: 2, resource: { buffer: ctx.rightEndIndexBuffer } },			
				{ binding: 3, resource: { buffer: ctx.outputBuffer_executeLogitSoftmax } },
			],
		});		
	}

	passEncoder.setPipeline(ctx.executeLogitSoftmax_pipeline);
	passEncoder.setBindGroup(0, ctx.bindGroup_executeLogitSoftmax);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(vocabSize / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return {
		buffer: ctx.outputBuffer_executeLogitSoftmax,
	};
};

globalThis.executeExtractPredictions = (ctx, softmaxBuffer, vocabSize, L, rightEndIndex, shaderCode) => {
	if (!ctx.executeExtractPredictions_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
		
		ctx.executeExtractPredictions_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});

		ctx.outputBuffer_executeExtractPredictions = ctx.webgpuDevice.createBuffer({
			size:  vocabSize * 2 * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		ctx.bindGroup_executeExtractPredictions = ctx.webgpuDevice.createBindGroup({
			layout: ctx.executeExtractPredictions_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: softmaxBuffer } },
				{ binding: 1, resource: { buffer: ctx.rightEndIndexBuffer } },
				{ binding: 2, resource: { buffer: ctx.outputBuffer_executeExtractPredictions } },
			],
		});		
	}		

	passEncoder.setPipeline(ctx.executeExtractPredictions_pipeline);
	passEncoder.setBindGroup(0, ctx.bindGroup_executeExtractPredictions);
	
	const workgroupsX = Math.ceil(vocabSize / 64);
	passEncoder.dispatchWorkgroups(workgroupsX);

	return {
		buffer: ctx.outputBuffer_executeExtractPredictions,
	};
};
