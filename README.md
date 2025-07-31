# Drug Discovery Knowledge Base Assistant

An AI-powered system for drug discovery insights using experimental data and large language models.

## Features

- Interactive chat interface for querying drug discovery data
- Real-time data visualization and analysis
- Integration with OpenAI GPT-4 for intelligent responses
- Dynamic document retrieval using LangChain
- Data exploration and visualization tools

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment:
- Create a `.env` file in the project root
- Add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. The application will load your experimental data automatically
2. Use the sidebar to explore and visualize your data
3. Ask questions in the chat interface about:
   - Data analysis
   - Drug discovery insights
   - Experimental results
   - Visualization requests

## Data Structure

The application expects a CSV file named `TurboID_ASK1_ML_Final.csv` containing your experimental data.

## Features

- Interactive data exploration
- Natural language querying
- Automated data visualization
- Context-aware responses
- Conversation memory for follow-up questions 