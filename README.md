## Learn Like An LLM

Learn Like An LLM is a project designed to help users understand and engage with language models by simulating the experience of guessing masked words in sentences. The initial implementation includes an n-gram model for generating sentence probabilities and calculating perplexity. Future expansions aim to incorporate more sophisticated models like RoBERTa, enable users to upload their own corpora, and provide a user-friendly frontend interface.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Future Plans](#future-plans)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Learn Like An LLM provides an interactive way to engage with language models by allowing users to guess masked words in sentences. The project demonstrates how n-gram models can be used to calculate sentence probabilities and perplexity scores. This is a fun and educational tool for those interested in natural language processing and machine learning.

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

You will be prompted to guess the missing word in masked sentences and receive feedback on your guesses, including a perplexity score.

## Features

- **N-gram Model:** Calculate sentence probabilities and perplexity using an n-gram model.
- **Interactive Game:** Engage with the language model by guessing masked words in sentences.
- **Custom Corpora:** Load and preprocess text corpora for analysis.

## Future Plans

- **Advanced Models:** Integrate more sophisticated models like RoBERTa.
- **User-Uploaded Corpora:** Allow users to upload and analyze their own text corpora.
- **Frontend Interface:** Develop a user-friendly frontend for easier interaction.
- **Performance Metrics:** Provide detailed performance metrics and analysis.
- **Automatic Translation:** Translate sentences to other languages for multilingual support.

## Contributing

I welcome contributions from the community! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push your branch to your fork.
4. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for more details.
