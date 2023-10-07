import pandas as pd
import maritalk
import dotenv
import os
import time

dotenv.load_dotenv()

key = os.getenv("MARITALK_KEY")

model = maritalk.MariTalk(key=key)

df_name = 'ds_001.parquet'

df = pd.read_parquet(f'data/raw/{df_name}')

# Extrair as mensagens de dentro da conversa
df_messages = df.explode('messages')

df_messages['timeOffset'] = df_messages['messages'].apply(lambda x: x['timeOffset'])
df_messages['text'] = df_messages['messages'].apply(lambda x: x['text'])
df_messages['senderWorkerId'] = df_messages['messages'].apply(lambda x: x['senderWorkerId'])
df_messages['messageID'] = df_messages['messages'].apply(lambda x: x['messageId'])
df_messages = df_messages.drop('messages', axis=1)

print('Mensagens originais antes da tradução:')
print("_"*80)
print(df_messages[['conversationId', 'text']].head())
print("_"*80)
print("\n")

translated_text = []

template = """Tarefa: Traduza o texto abaixo de forma natural para o Português do Brasil: 

{}

Lembrete: Não é necessário explicar a tradução, apenas traduza o texto de forma natural.

Tradução: """

start_time = time.time()
success_count = 0
failure_count = 0

failure_df = pd.DataFrame(columns=['conversationId', 'messageID', 'text'])

for index, row in df_messages.iterrows():
    success = False
    max_attempts = 2 
    attempts = 0
    
    prompt = template.format(row['text'])

    if row['text'].startswith('@'):
        print(f"O texto começa com '@': {row['text']} -> Arquivando mensagem sem tradução e salvando no log de falhas")
        failure_count += 1
        translated_text.append(row['text'])
        new_row = {'conversationId': row['conversationId'], 'messageID': row['messageID'], 'text': row['text']}
        failure_df = pd.concat([failure_df, pd.DataFrame([new_row])], ignore_index=True)
        failure_df.to_csv(f'data/processed/logs/failure_log_{df_name[:-8]}.csv', index=False)
        continue

    while not success and attempts < max_attempts:
        try:
            answer = model.generate(
                prompt,
                chat_mode=False,
                do_sample=False,
                max_tokens=4096
            )

            clean_answer = answer.strip().split('\n')[0]

            translated_text.append(clean_answer)
            success = True
            success_count += 1
            print(f"Tradução bem-sucedida: {row['text']} -> {clean_answer}")
        except Exception as e:
            attempts += 1
            print(f"Erro ao traduzir a mensagem: {row['text']}. Tentativa {attempts}. Erro: {e}")
            time.sleep(5)  # Esperar 5 segundos antes de tentar novamente

    time.sleep(5)

    if not success:
        translated_text.append(row['text'])
        failure_count += 1
        new_row = {'conversationId': row['conversationId'], 'messageID': row['messageID'], 'text': row['text']}
        failure_df = pd.concat([failure_df, pd.DataFrame([new_row])], ignore_index=True)
        failure_df.to_csv(f'data/processed/logs/failure_log_{df_name[:-8]}.csv', index=False)

    df_messages.loc[index, 'text_translated'] = translated_text[-1]
    df_messages.to_parquet(f'data/processed/interim/interim_translated_{df_name[:-8]}.parquet', index=False)

# Adicionando a tradução ao dataframe df_messages
df_messages['text_translated'] = translated_text

def reaggregate_messages_and_translation(group):
    messages = group.apply(lambda x: {
        'timeOffset': x['timeOffset'],
        'text': x['text'],
        'senderWorkerId': x['senderWorkerId'],
        'messageId': x['messageID']
    }, axis=1).tolist()
    
    messages_translated = group.apply(lambda x: {
        'timeOffset': x['timeOffset'],
        'text': x['text_translated'],
        'senderWorkerId': x['senderWorkerId'],
        'messageId': x['messageID']
    }, axis=1).tolist()

    # Colunas constantes dentro de cada grupo
    constant_values = {
        'movieMentions': group['movieMentions'].iloc[0],
        'respondentQuestions': group['respondentQuestions'].iloc[0],
        'respondentWorkerId': group['respondentWorkerId'].iloc[0],
        'initiatorWorkerId': group['initiatorWorkerId'].iloc[0],
        'initiatorQuestions': group['initiatorQuestions'].iloc[0]
    }
    
    return pd.Series({'messages': messages, 'messages_translated': messages_translated, **constant_values})

reconstructed_df = df_messages.groupby('conversationId').apply(reaggregate_messages_and_translation).reset_index()
reconstructed_df.to_parquet(f'data/processed/translated_{df_name[:-8]}.parquet', index=False)

total_time = time.time() - start_time
print(f"Tempo total de execução: {total_time:.2f} segundos")
print(f"Total de traduções bem-sucedidas: {success_count}")
print(f"Total de falhas: {failure_count}")