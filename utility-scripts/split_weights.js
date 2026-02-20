#!/usr/bin/env node
/**
 * split_weights.js
 * 
 * Splits a CUDA-format .bin weights file into two parts (A and B) for GitHub's <100MB limit.
 * 
 * File A: tokenEmbeddings, finalRMSNormGamma, transformer blocks 0 to (numTransformers/2 - 1)
 * File B: transformer blocks (numTransformers/2) to (numTransformers - 1)
 * 
 * Usage: node split_weights.js <input_file.bin>
 * Output: <input_file>_A.bin, <input_file>_B.bin
 */

const fs = require('fs');
const path = require('path');

// ============================================================================
// CONFIGURATION - Edit these values before each run
// ============================================================================
const NetworkMeta = {
    dimensions: 512,
    heads: 4,
    ropeDenomBase: 10000,
    ffnDimMultiplier: 4,
    numTransformers: 8,
};

const vocabSize = 10096;

// ============================================================================
// Derived values
// ============================================================================
const dim = NetworkMeta.dimensions;
const ffnDim = dim * NetworkMeta.ffnDimMultiplier;
const numTransformers = NetworkMeta.numTransformers;
const halfTransformers = Math.floor(numTransformers / 2);

console.log('=== Weight Splitter Configuration ===');
console.log(`Dimensions: ${dim}`);
console.log(`FFN Dim: ${ffnDim}`);
console.log(`Vocab Size: ${vocabSize}`);
console.log(`Total Transformers: ${numTransformers}`);
console.log(`Split: ${halfTransformers} + ${numTransformers - halfTransformers}`);
console.log('');

// Get input filename from command line
const inputFile = process.argv[2];
if (!inputFile) {
    console.error('Usage: node split_weights.js <input_file.bin>');
    process.exit(1);
}

if (!fs.existsSync(inputFile)) {
    console.error(`Error: File not found: ${inputFile}`);
    process.exit(1);
}

// Generate output filenames
const baseName = inputFile.replace(/\.bin$/, '');
const outputFileA = `${baseName}_A.bin`;
const outputFileB = `${baseName}_B.bin`;

console.log(`Input: ${inputFile}`);
console.log(`Output A: ${outputFileA}`);
console.log(`Output B: ${outputFileB}`);
console.log('');

// Read the input file
console.log('Reading input file...');
const fileBuffer = fs.readFileSync(inputFile);
const dataView = new DataView(fileBuffer.buffer, fileBuffer.byteOffset, fileBuffer.byteLength);

// Parse header
const headerLength = Number(dataView.getBigUint64(0, true));
const headerStart = 8;
const headerEnd = headerStart + headerLength;

if (headerEnd > fileBuffer.length) {
    console.error('Error: Header length exceeds file size');
    process.exit(1);
}

const headerBuffer = fileBuffer.slice(headerStart, headerEnd);
const metadata = JSON.parse(headerBuffer.toString('utf-8'));

console.log('Original metadata structure:');
console.log(`  - tokenEmbeddings: shape ${JSON.stringify(metadata.tokenEmbeddings.shape)}`);
console.log(`  - finalRMSNormGamma: shape ${JSON.stringify(metadata.finalRMSNormGamma.shape)}`);
console.log(`  - transformerBlocks: ${metadata.transformerBlocks.length} blocks`);
console.log('');

// Calculate data offset (with alignment padding)
const ALIGNMENT = 8;
const paddingNeeded = (ALIGNMENT - (headerEnd % ALIGNMENT)) % ALIGNMENT;
let dataOffset = headerEnd + paddingNeeded;

// Helper to calculate tensor size in bytes
function getTensorByteSize(tensorMeta) {
    const numElements = tensorMeta.shape.reduce((acc, val) => acc * val, 1);
    const bytesPerElement = tensorMeta.dtype === 'float32' ? 4 : 8;
    return numElements * bytesPerElement;
}

// Helper to extract tensor data from the file
function extractTensorData(tensorMeta, currentOffset) {
    const byteSize = getTensorByteSize(tensorMeta);
    const data = fileBuffer.slice(currentOffset, currentOffset + byteSize);
    return { data, nextOffset: currentOffset + byteSize };
}

// ============================================================================
// Extract all tensor data from the original file
// ============================================================================
console.log('Extracting tensor data...');

let currentOffset = dataOffset;

// Token embeddings
const tokenEmbeddingsResult = extractTensorData(metadata.tokenEmbeddings, currentOffset);
const tokenEmbeddingsData = tokenEmbeddingsResult.data;
currentOffset = tokenEmbeddingsResult.nextOffset;
console.log(`  - tokenEmbeddings: ${tokenEmbeddingsData.length} bytes`);

// Final RMS norm gamma
const finalRMSResult = extractTensorData(metadata.finalRMSNormGamma, currentOffset);
const finalRMSData = finalRMSResult.data;
currentOffset = finalRMSResult.nextOffset;
console.log(`  - finalRMSNormGamma: ${finalRMSData.length} bytes`);

// Transformer blocks
const transformerData = [];
for (let t = 0; t < metadata.transformerBlocks.length; t++) {
    const blockMeta = metadata.transformerBlocks[t];
    const blockData = {};
    
    // Read tensors in the order they appear in metadata
    for (const tensorName of Object.keys(blockMeta)) {
        const result = extractTensorData(blockMeta[tensorName], currentOffset);
        blockData[tensorName] = result.data;
        currentOffset = result.nextOffset;
    }
    
    transformerData.push(blockData);
    
    const blockSize = Object.values(blockData).reduce((sum, d) => sum + d.length, 0);
    console.log(`  - transformerBlock[${t}]: ${blockSize} bytes`);
}

console.log('');

// ============================================================================
// Build File A: tokenEmbeddings, finalRMSNormGamma, first half of transformers
// ============================================================================
console.log('Building File A...');

const metadataA = {
    tokenEmbeddings: metadata.tokenEmbeddings,
    finalRMSNormGamma: metadata.finalRMSNormGamma,
    transformerBlocks: metadata.transformerBlocks.slice(0, halfTransformers)
};

const headerStringA = JSON.stringify(metadataA);
const headerBytesA = Buffer.from(headerStringA, 'utf-8');
const headerLengthA = headerBytesA.length;
const paddingNeededA = (ALIGNMENT - (headerLengthA % ALIGNMENT)) % ALIGNMENT;

// Calculate total size for file A
let totalSizeA = 8 + headerLengthA + paddingNeededA; // header length (8 bytes) + header + padding
totalSizeA += tokenEmbeddingsData.length;
totalSizeA += finalRMSData.length;
for (let t = 0; t < halfTransformers; t++) {
    for (const tensorName of Object.keys(transformerData[t])) {
        totalSizeA += transformerData[t][tensorName].length;
    }
}

console.log(`  Header size: ${headerLengthA} bytes (+ ${paddingNeededA} padding)`);
console.log(`  Total size: ${totalSizeA} bytes (${(totalSizeA / 1024 / 1024).toFixed(2)} MB)`);

// Write File A
const bufferA = Buffer.alloc(totalSizeA);
let writeOffsetA = 0;

// Write header length (8 bytes, little-endian)
bufferA.writeBigUInt64LE(BigInt(headerLengthA), writeOffsetA);
writeOffsetA += 8;

// Write header
headerBytesA.copy(bufferA, writeOffsetA);
writeOffsetA += headerLengthA;

// Write padding
writeOffsetA += paddingNeededA;

// Write tokenEmbeddings
tokenEmbeddingsData.copy(bufferA, writeOffsetA);
writeOffsetA += tokenEmbeddingsData.length;

// Write finalRMSNormGamma
finalRMSData.copy(bufferA, writeOffsetA);
writeOffsetA += finalRMSData.length;

// Write first half of transformer blocks
for (let t = 0; t < halfTransformers; t++) {
    for (const tensorName of Object.keys(transformerData[t])) {
        transformerData[t][tensorName].copy(bufferA, writeOffsetA);
        writeOffsetA += transformerData[t][tensorName].length;
    }
}

fs.writeFileSync(outputFileA, bufferA);
console.log(`  Written to: ${outputFileA}`);
console.log('');

// ============================================================================
// Build File B: second half of transformers only
// ============================================================================
console.log('Building File B...');

const metadataB = {
    transformerBlocks: metadata.transformerBlocks.slice(halfTransformers)
};

const headerStringB = JSON.stringify(metadataB);
const headerBytesB = Buffer.from(headerStringB, 'utf-8');
const headerLengthB = headerBytesB.length;
const paddingNeededB = (ALIGNMENT - (headerLengthB % ALIGNMENT)) % ALIGNMENT;

// Calculate total size for file B
let totalSizeB = 8 + headerLengthB + paddingNeededB;
for (let t = halfTransformers; t < numTransformers; t++) {
    for (const tensorName of Object.keys(transformerData[t])) {
        totalSizeB += transformerData[t][tensorName].length;
    }
}

console.log(`  Header size: ${headerLengthB} bytes (+ ${paddingNeededB} padding)`);
console.log(`  Total size: ${totalSizeB} bytes (${(totalSizeB / 1024 / 1024).toFixed(2)} MB)`);

// Write File B
const bufferB = Buffer.alloc(totalSizeB);
let writeOffsetB = 0;

// Write header length (8 bytes, little-endian)
bufferB.writeBigUInt64LE(BigInt(headerLengthB), writeOffsetB);
writeOffsetB += 8;

// Write header
headerBytesB.copy(bufferB, writeOffsetB);
writeOffsetB += headerLengthB;

// Write padding
writeOffsetB += paddingNeededB;

// Write second half of transformer blocks
for (let t = halfTransformers; t < numTransformers; t++) {
    for (const tensorName of Object.keys(transformerData[t])) {
        transformerData[t][tensorName].copy(bufferB, writeOffsetB);
        writeOffsetB += transformerData[t][tensorName].length;
    }
}

fs.writeFileSync(outputFileB, bufferB);
console.log(`  Written to: ${outputFileB}`);
console.log('');

// ============================================================================
// Summary
// ============================================================================
console.log('=== Split Complete ===');
console.log(`Original file: ${fileBuffer.length} bytes (${(fileBuffer.length / 1024 / 1024).toFixed(2)} MB)`);
console.log(`File A: ${totalSizeA} bytes (${(totalSizeA / 1024 / 1024).toFixed(2)} MB)`);
console.log(`File B: ${totalSizeB} bytes (${(totalSizeB / 1024 / 1024).toFixed(2)} MB)`);
console.log(`Combined: ${totalSizeA + totalSizeB} bytes (${((totalSizeA + totalSizeB) / 1024 / 1024).toFixed(2)} MB)`);
console.log('');
console.log('To load in JS:');
console.log(`  loadModelWeightsV3('./model/${path.basename(outputFileA)}', './model/${path.basename(outputFileB)}')`);
