# Customer Service Optimization Project

** Esse projeto contem modulos de tratamento de texto, classification de sentimento do tweet e otimização de fluxo de atendimento

## 📂 estrutura
projeto/
├── agents/ # modelos de agents
│ ├── flux_agents_use.ipynb # exemplo de usar agents de atendimento
│ ├── rate_agents_use.ipynb # exemplo de usar para dar a nota da resposta do agent
│ ├── ds_connect.py # conexão de llm
│ ├── flux_agents.py # agents do atendimento
│ └── rate_agents.py # agents para avaliar a resposta do agent em notas com razção
├── classificador/ # modulo de classificação de sentimento do tweet
│ ├── deploy/ # deploy o modelo de classificação
│ ├── modelos/ # os modelos trainados de classificação
│ ├── utils.py # funções para fazer o tratamento de tweets
│ ├── Q1_agent.ipynb # teste de usando agent para fazer a classificação de sentimento
│ └── Q1_modelos_classicos.ipynb # treinar e avaliar os modelos classicos de classificação de sentimento
├── dados/ # modulo de tratamento de dados raw
│ ├── raw_data/ # raw data
│ ├── clean_raw_data.py # funcao de tratamentode dados
│ └── preprocessed/ # depois do tratamento
├── diagram/ # modulo de desenho de fluxos
│ ├── agent_fluxo.drawio # fluxo de evitar as alucinações e inconsistências na resposta do agent usando agents
│ ├── ajuda_atendimento.drawio # fluxo de como usar os agents no atendimento
│ ├── avaliacao.drawio # fluxo de avaliar a solução
│ ├── fluxo_colecao_dados.drawio # fluxo de coleção do tweets
│ └── rate_agentes.drawio # fluxo de usar agents para avaliar a resposta do agent em notas com razção
├── rag/ # modulo de base de conhecimento do agent
│ ├── exemplo_de_uso.ipynb # exemplo de usar 
│ ├── rag_tweet_hist.ipynb # transferir os texto de tweets historicos pra vetores e save
│ ├── embeddings_tweet_hist.npy # os vetores de embedding de texto
│ └── paragraphs.txt # save texto como paragraphs 
├── requirements.txt # instalar pacotes precisas
└── README.md # readme

## instalação de pacotes
pip install -r requirements.txt



## Detalhes

### clean raw data
python dados/clean_raw_data.py \
  --input_path case_data_science_nlp___analise_de_sentimentos 1.xlsx \
  --output_path analise_de_sentimentos.csv

### classificador de sentimento
#### carregar modelos
vectorizer = joblib.load(open('../modelos/TfidfVectorizer.pkl', 'rb'))
model = joblib.load(open('../modelos/modelNB_TfidfVectorizer.pkl', 'rb'))
#### carregar dados
df_pred = pd.read_csv('tweet_sentiment_airlines_deploy.csv')
df_pred.columns = ['texto_tweet']
#### tratamentos de dados
X = df_pred['texto_tweet']
X = [utils.clean_tweet(x) for x in X]
X = vectorizer.transform(X)
#### predição de sentimento
df_pred['sentimento_tweet'] = model.predict(X)

## otimização de atendimento do agents (flux_agents_use.ipynb)
customer_question = "@airfrance lost luggage in overhead cabin, email no response, phone no one answers. pls help."
output = class_text_agent(client,customer_question)
...
