# micro-llms.ts

`micro-llms.ts` is a project dedicated to providing lightweight, dependency-free, and pure TypeScript implementations of various "micro" Large Language Models (LLMs). Inspired by Andrej Karpathy's educational work, this project aims to make the core concepts of LLMs accessible and understandable through atomic implementations, perfect for learning, experimentation, and building foundational knowledge. It runs efficiently using Bun.

## Core Components

The project's architecture is designed for clarity and modularity:

*   **`src/core/`**: Contains the fundamental building blocks for LLMs, including matrix operations, optimization algorithms (like Adam), and a tokenizer, all essential for the training and inference processes.
*   **`src/utils/`**: Provides utility functions for data loading, managing training loops, and performing inference operations.

## Implemented Models

`micro-llms.ts` currently includes "micro" implementations of the following LLM architectures:

*   **GPT**: A Generative Pre-trained Transformer model, showcasing the foundational architecture for many modern large language models.
*   **Llama**: An implementation inspired by the Llama family of models, known for their efficiency and performance.
*   **Deepseek**: An implementation based on the Deepseek architecture, offering insights into its unique design choices.

## Installation

To get started with `micro-llms.ts`, first install the project dependencies using Bun:

```bash
bun install
```

## Usage

The `main.ts` script allows you to train and perform inference with the implemented models via command-line arguments.

### Training a Model

To train a model, specify the model name (`gpt`, `llama`, or `deepseek`) and the `train` command:

```bash
bun run main.ts gpt train
bun run main.ts llama train
bun run main.ts deepseek train
```
After training, the model will perform an inference step and save its state to the `models/` directory (e.g., `models/gpt.bin`).

### Performing Inference

To perform inference with a pre-trained model, specify the model name and the `infere` command. The script will automatically load the saved model from the `models/` directory:

```bash
bun run main.ts gpt infere
bun run main.ts llama infere
bun run main.ts deepseek infere
```

This project was created using `bun init` in bun v1.3.5. [Bun](https://bun.com) is a fast all-in-one JavaScript runtime.
