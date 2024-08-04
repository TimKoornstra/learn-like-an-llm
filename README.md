## Learn Like An LLM

Learn Like An LLM is a project designed to help users understand and engage with language models by simulating the experience of guessing masked words in sentences. The initial implementation includes a BERT-based model for generating probable words, and the `all-MiniLM-L6-v2` model to generate cosine similarity between user input and the original text. Future expansions aim to incorporate more sophisticated models, improve the scoring system, improve the difficulty system, and provide a user-friendly frontend interface.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Future Plans](#future-plans)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Learn Like An LLM offers an interactive way to engage with language models by allowing users to guess masked words in sentences. The project demonstrates how language models can be used to learn a new language by providing more contextual feedback than just "correct" or "incorrect." This is a fun and educational tool for those interested in natural language processing and machine learning.

## Installation

To run this project, you'll need Python 3.11 or higher and the required dependencies. Follow these steps to set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/learn-like-an-llm.git
   cd learn-like-an-llm
   ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the NLTK data required for tokenization:
    ```python
    import nltk
    nltk.download('punkt')
    ```

## Usage

To start the word masking and guessing game, run the `main.py` script:

```bash
python src/main.py
```

You will be prompted to choose a language (e.g., 'english', 'spanish', 'french'). The program will then load the corresponding text corpus from `data/corpus/<language>` and begin the interactive guessing game. You will guess the missing word in masked sentences and receive feedback on your guesses, including a perplexity score, which measures how well the language model predicts the missing word based on the context of the sentence. The lower the perplexity score, the better the model (in this case: you) performs.

## Features

- **Language Models:** Use BERT and MiniLM models for word prediction and similarity analysis.
- **Interactive Game:** Engage with the language model by guessing masked words in sentences.
- **Custom Corpora:** Load and preprocess text corpora for analysis.
- **User-Uploaded Corpora:** Upload and play with your own corpora by uploading text files in the `data/corpus/<language>` directory.
- **Automatic Translation:** Automatic translation of the original and user input text to English after guessing the masked word.

## Future Plans

- **Frontend Interface:** Develop a user-friendly frontend for easier interaction.
- **Performance Metrics:** Provide detailed performance metrics and analysis.

## Contributing

I welcome contributions from the community! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push your branch to your fork.
4. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for more details.
