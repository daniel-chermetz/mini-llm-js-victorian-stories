// Letters: A–Z and a–z
const letters = {};
"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz".split("").forEach(ch => {
  letters[ch] = true;
});

// Digits: 0–9
const digits = {};
"0123456789".split("").forEach(ch => {
  digits[ch] = true;
});

// Common punctuation
const punctuation = {};
",.;:!?\"'`-–—()[]{}<>/\\|@#$%^&*_+=~".split("").forEach(ch => {
  punctuation[ch] = true;
});

const divideTextIntoWords = (letterSeq) => {
	const words = [];

	let currentWord = '';
	for (let i = 0; i < letterSeq.length; i++) {
		let wordEnd = false;

		if (!currentWord.length) {
			currentWord = currentWord + letterSeq[i];
			continue;
		}

		if (punctuation[letterSeq[i]] && currentWord.at(-1) !== ' ') {
			if (i === letterSeq.length - 1) {
				wordEnd = true;
			}
			if (i < (letterSeq.length - 1) && (!letters[letterSeq[i + 1]] && !digits[letterSeq[i + 1]])) {
				wordEnd = true;
			}
			if (!letters[currentWord.at(-1)] && !digits[currentWord.at(-1)]) {
				wordEnd = true;
			}
			if (letterSeq[i] === '.' && (currentWord === '.' || currentWord === ' .' || currentWord === '..' || currentWord === ' ..')) {
				wordEnd = false;
			}
		}

		if (letters[letterSeq[i]] || digits[letterSeq[i]]) {
			if (currentWord.length === 1 && (punctuation[currentWord[0]] || currentWord[0] === '\n' || currentWord[0] === '\t')) {
				wordEnd = true;
			}
			if (currentWord.length === 2 && currentWord[0] === ' ' && punctuation[currentWord[1]]) {
				wordEnd = true;
			}
			if (currentWord.length >= 2 && currentWord.at(-1) === '.' && currentWord.at(-2) === '.') {
				wordEnd = true;
			}			
		}

		if (letterSeq[i] === ' ') {
			wordEnd = true;
		}
		if (letterSeq[i] === '\n') {
			wordEnd = true;
		}
		if (letterSeq[i] === '\t') {
			wordEnd = true;
		}

		if (wordEnd) {
			words.push(currentWord);
			currentWord = '';
			i--;
		} else {
			currentWord += letterSeq[i];
		}
	}
	if (currentWord.length) {
		words.push(currentWord);
	}

	return words;
}

const checkIfTokenComboIsPresent = (targetComboStr, words) => {
	for (let wordIndex = 0; wordIndex < words.length; wordIndex++) {
		const currentWord = words[wordIndex];
		for (let subWordIndex = 0; subWordIndex < currentWord.length - 1; subWordIndex++) {
			let part1 = currentWord[subWordIndex];
			if (Array.isArray(part1)) {
				part1 = part1[0];
			}
			let part2 = currentWord[subWordIndex + 1];
			if (Array.isArray(part2)) {
				part2 = part2[0];
			}

			const combo = part1 + part2;
			if (combo === targetComboStr) {
				return true;
			}
		}
	}

	return false;
}

const mergePerHighestPriority = (mergePriorityTokenComboStr, words) => {
	for (let wordIndex = 0; wordIndex < words.length; wordIndex++) {
		const currentWord = words[wordIndex];

		for (let subWordIndex = 0; subWordIndex < currentWord.length - 1; subWordIndex++) {
			let part1 = currentWord[subWordIndex];
			if (Array.isArray(part1)) {
				part1 = part1[0];
			}
			let part2 = currentWord[subWordIndex + 1];
			if (Array.isArray(part2)) {
				part2 = part2[0];
			}

			const comparisonTargetComboStr = mergePriorityTokenComboStr ? mergePriorityTokenComboStr : mergePriority[mergePriority.length - 1];
			if (part1 + part2 === comparisonTargetComboStr) {
				let wordPostMerge = [];
				for (let i = 0; i < currentWord.length; i++) {
					if (i < subWordIndex) {
						wordPostMerge.push(currentWord[i]);
					}
					if (i > subWordIndex + 1) {
						wordPostMerge.push(currentWord[i]);
					}
					if (i === subWordIndex) {
						wordPostMerge.push([part1 + part2]);
					}
				}

				words[wordIndex] = wordPostMerge;
				wordIndex--; // check same word again for another repition of the same combo
				break;
			}
		}
	}

	return words;
}

globalThis.bpeTextSequence = (textSeq) => {
	let textWordsArr = divideTextIntoWords(textSeq);
	for (let vocabIndex = 0; vocabIndex < globalThis.vocab.length; vocabIndex++) {
		if (checkIfTokenComboIsPresent(globalThis.vocab[vocabIndex], textWordsArr)) {
			textWordsArr = mergePerHighestPriority(globalThis.vocab[vocabIndex], textWordsArr)
		}
	}

	const flattenedTokensArr = [];
	for (let wordIndex = 0; wordIndex < textWordsArr.length; wordIndex++) {
		const currentWord = textWordsArr[wordIndex];
		for (let subWordIndex = 0; subWordIndex < currentWord.length; subWordIndex++) {
			if (Array.isArray(currentWord[subWordIndex])) {
				flattenedTokensArr.push(currentWord[subWordIndex][0]);
			} else {
				// iterate over unmerged string
				const unmergedStr = currentWord[subWordIndex];
				for (let charIndex = 0; charIndex < unmergedStr.length; charIndex++) {
					flattenedTokensArr.push(unmergedStr[charIndex]);
				}
			}
		}
	}
	console.log(textWordsArr);
	console.log(flattenedTokensArr);

	return flattenedTokensArr;
}
