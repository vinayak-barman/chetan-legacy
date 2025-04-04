# Chetan legacy agent library
<a href="https://discord.gg/pUpVqA6g">![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white)
</a>

This is a preliminary version of the under-development [chetan](https://github.com/snayu-ai/chetan) agent system framework.


## Features
- LLMs
  - ✅ OpenAI
  - ✅ AzureOpenAI
  - ✅ Groq
  - ✅ vLLM
  - ❌ Ollama
  - ❌ LM Studio
- Actions
  - ✅ Ask user
  - ✅ Tell user
  - ✅ Code execution (StdIO redirect)
  - ✅ Crawl4AI web crawler
  - ✅ Tavily search
  - ✅ Wikipedia search
  - ✅ Exit agent loop

## Getting Started
### Clone the repository
```bash
git clone https://github.com/vinayak-barman/chetan-legacy
```
### Create a virtual environment (recommended)
```bash
cd chetan-legacy
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install the required packages
```bash
pip install python-dotenv openai pydantic wikipedia tavily crawl4ai
```
> [!NOTE]
> You would need to setup the `crawl4ai` package.
> You may need `OpenAI`/`AzureOpenAI`/`Groq` API keys for the respective LLMs. 
> For local LLMs, you need to setup `vLLM`. `Ollama` and `LM Studio` enforce strict role alteration and are not supported.

### Run the demo notebook 
Open `demos/demo.ipynb` in Jupyter Notebook or any compatible environment.

## Disclaimer
This is a preliminary version of the Chetan agent system framework. It is **not** intended for *production* use and contains bugs or incomplete features. Use at your own risk.

## Contributing
Thanks for your interest in contribution. These are my recommendations for contributing to this project:

- New actions
- New LLMs
- Bug fixes
- Documentation improvements
- Examples
- Tests

If you are interested, feel free to contribute to [chetan](https://github.com/snayu-ai/chetan), which is actively developed and successor to this repository.

Email for contact: snayu-ai@protonmail.com


## License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.