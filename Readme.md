# HR Policy Chatbot

An intelligent chatbot that responds to employee queries about HR policies using LangChain's pandas query engine with Azure OpenAI integration.

##Demo Video(Code working)

https://github.com/user-attachments/assets/af70438b-0176-450d-b792-b0b6463a28fa


## 🚀 Features

- **Smart Query Matching**: Uses TF-IDF similarity matching for direct FAQ responses
- **LangChain Integration**: Leverages pandas dataframe agent for complex queries
- **Azure OpenAI**: Powered by Azure's OpenAI services
- **Modern Web Interface**: Clean, responsive chat interface
- **Dual Response Strategy**: 
  - Direct similarity matching for exact FAQ matches
  - AI-powered pandas agent for complex reasoning
- **Confidence Scoring**: Shows confidence levels and matching methods

## 📋 Prerequisites

- Python 3.8 or higher
- Azure OpenAI account with deployed models
- Git (for cloning the repository)

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/hr-chatbot.git
cd hr-chatbot
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

1. Copy the environment template:
```bash
cp .env.template .env
```

2. Edit `.env` file with your Azure OpenAI credentials:
```bash
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENGINE=your-gpt-deployment-name
AZURE_OPENAI_MODEL=your-model
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_EMBED_DEPLOYMENT_NAME=your-embedding-deployment-name
```

### 5. Prepare FAQ Data

The application includes sample FAQ data in `faqs.csv`. You can:
- Use the provided sample data
- Replace with your own FAQ dataset
- Add more questions and answers to the existing file

### 6. Create Templates Directory

```bash
mkdir templates
# Move index.html to templates/ directory
```

### 7. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`


## 🔧 Configuration

### Azure OpenAI Setup

1. **Create Azure OpenAI Resource**:
   - Go to Azure Portal
   - Create a new Azure OpenAI resource
   - Deploy GPT model (e.g., gpt-35-turbo)
   - Deploy text embedding model (e.g., text-embedding-ada-002)

2. **Get Credentials**:
   - Copy the endpoint URL
   - Copy the API key
   - Note the deployment names

### Customizing FAQ Data

Edit `faqs.csv` to include your organization's specific HR policies:

```csv
Question,Answer
What is the remote work policy?,Employees can work remotely up to 2 days per week with manager approval.
How do I request vacation time?,Submit vacation requests through the HR portal at least 2 weeks in advance.
```

## 🚀 Usage

### Web Interface

1. Open your browser and go to `http://localhost:5000`
2. Type your HR-related question in the chat input
3. The chatbot will provide relevant answers using either:
   - Direct FAQ matching for exact questions
   - AI-powered analysis for complex queries

### API Endpoints

- `POST /chat`: Send a message to the chatbot
- `GET /faqs`: View all available FAQs

Example API usage:
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What benefits does the company offer?"}'
```

## 🎯 How It Works

### 1. Similarity Matching
- Uses TF-IDF vectorization to find the closest matching FAQ
- Returns direct answers for high-confidence matches

### 2. Pandas Agent Integration
- For complex queries, uses LangChain's pandas dataframe agent
- Leverages Azure OpenAI for intelligent reasoning over the FAQ dataset

### 3. Response Strategy
- **High Similarity Match**: Returns direct FAQ answer
- **Low Similarity**: Uses AI agent for complex reasoning
- **No Match**: Provides helpful fallback response

## 🛡️ Security Considerations

- Keep your `.env` file secure and never commit it to version control
- Add `.env` to your `.gitignore` file
- Use environment variables for all sensitive configuration
- Consider implementing rate limiting for production use

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check your virtual environment is activated

2. **Azure OpenAI Connection Issues**:
   - Verify your endpoint URL and API key
   - Check deployment names match your Azure setup
   - Ensure your Azure subscription has quota available

3. **CSV Loading Issues**:
   - Ensure `faqs.csv` is in the root directory
   - Check
