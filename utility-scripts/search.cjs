const fs = require('fs').promises;
const path = require('path');

// Directory where the JSON files are stored
const DIR_PATH = path.join(__dirname, '../tokenizedVictorianStories');

async function main() {
    // 1. Get keywords from terminal arguments (ignoring 'node' and 'search.js')
    const keywords = process.argv.slice(2).map(kw => kw.toLowerCase());
    
    if (keywords.length === 0) {
        console.error('❌ Please provide at least one keyword to search for.');
        console.log('Usage: node search.js <keyword1> <keyword2> ...');
        process.exit(1);
    }

    console.log(`🔍 Searching for stories containing all keywords: [${keywords.join(', ')}]\n`);

    try {
        // 2. Read the directory and filter for matching JSON files
        const files = await fs.readdir(DIR_PATH);
        const storyFiles = files
            .filter(file => file.startsWith('tokenizedStories_') && file.endsWith('.json'))
            .sort(); // Ensures they are processed in ascending order

        if (storyFiles.length === 0) {
            console.log('No files matching "tokenizedStories_*.json" found in the directory.');
            return;
        }

        let matchCount = 0;

        // 3. Process each file
        for (const file of storyFiles) {
            const filePath = path.join(DIR_PATH, file);
            
            // Extract the file number using regex (e.g., "0001" -> 1)
            const fileNumMatch = file.match(/tokenizedStories_(\d+)\.json/);
            if (!fileNumMatch) continue;
            
            const fileNumber = parseInt(fileNumMatch[1], 10);
            
            // Read and parse the JSON file
            const fileContent = await fs.readFile(filePath, 'utf-8');
            const stories = JSON.parse(fileContent);

            // 4. Search through stories
            stories.forEach((story, localIndex) => {
                // Lowercase all tokens in the story once for performance
                const lowerCaseStoryTokens = story.map(token => token.toLowerCase());

                // Check if EVERY keyword is present in the story
                // (Either as an exact match or as a substring of a token)
                const hasAllKeywords = keywords.every(keyword => 
                    lowerCaseStoryTokens.some(token => token.includes(keyword))
                );

                if (hasAllKeywords) {
                    matchCount++;
                    
                    // Note: I used (fileNumber - 1) so that file 0001 starts at global index 0. 
                    // If you strictly want (fileNumber - 0), change the `- 1` to `- 0`.
                    const globalIndex = (fileNumber - 1) * 1000 + localIndex;

                    console.log(`✅ Match found!`);
                    console.log(`   File:         ${file}`);
                    console.log(`   Local Index:  ${localIndex}`);
                    console.log(`   Global Index: ${globalIndex}`);
                    console.log('--------------------------------------------------');
                }
            });
        }

        console.log(`🎉 Search complete. Total matches found: ${matchCount}`);

    } catch (error) {
        console.error('❌ An error occurred:', error.message);
        if (error.code === 'ENOENT') {
            console.error(`Ensure the folder "${DIR_PATH}" exists in the same directory as this script.`);
        }
    }
}

main();