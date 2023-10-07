import pandas as pd
import maritalk
import dotenv
import os
import re
import time

dotenv.load_dotenv()

key = os.getenv("MARITALK_KEY")

model = maritalk.MariTalk(key=key)

df_name = 'ds_002.parquet'

df = pd.read_parquet(f'data/raw/{df_name}')

# Extrair as mensagens de dentro da conversa
df_messages = df.explode('messages')

df_messages['timeOffset'] = df_messages['messages'].apply(lambda x: x['timeOffset'])
df_messages['text'] = df_messages['messages'].apply(lambda x: x['text'])
df_messages['senderWorkerId'] = df_messages['messages'].apply(lambda x: x['senderWorkerId'])
df_messages['messageID'] = df_messages['messages'].apply(lambda x: x['messageId'])
df_messages = df_messages.drop('messages', axis=1)

total_messages = len(df_messages) 
current_message_count = 0

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

# Verificações para traduções customizadas (não realizaveis com qualidade pelo modelo)

def _translate_remaining(text, row):
    prompt = template.format(text)
    max_attempts = 2
    attempts = 0

    while attempts < max_attempts:
        try:
            answer = model.generate(
                prompt,
                chat_mode=False,
                do_sample=False,
                max_tokens=4096
            )

            clean_answer = answer.strip().split('\n')[0]
            time.sleep(5)

            return clean_answer

        except Exception as e:
            attempts += 1
            print(f"({current_message_count}/{total_messages}) Erro ao traduzir a mensagem: {text}. Tentativa {attempts}. Erro: {e}")
            time.sleep(5)  # Esperar 5 segundos antes de tentar novamente

    new_row = {'conversationId': row['conversationId'], 'messageID': row['messageID'], 'text': text}
    failure_df = pd.concat([failure_df, pd.DataFrame([new_row])], ignore_index=True)
    failure_df.to_csv(f'data/processed/logs/failure_log_{df_name[:-8]}.csv', index=False)

    return text

def _custom_translation(text, row):
    text = text.strip()
    split_text = text.split()

    # Caso 1 -> @123456
    if text.startswith("@") and text[1:].isdigit() and len(split_text) == 1:
        return text 

    # Caso 2 -> Or @123456
    if len(split_text) >= 2:
        if split_text[0].lower() == "or" and split_text[1].startswith('@') and split_text[1][1:].isdigit():
            return "Ou " + split_text[1]

    # Caso 3 -> And @123456
    if len(split_text) >= 2:
        if split_text[0].lower() == "and" and split_text[1].startswith('@') and split_text[1][1:].isdigit():
            return "E " + split_text[1]

    # Caso 4 -> ! ou ? ou . ou " " ou ""
    elif re.match(r"^[!?. \"\"]+$", text):
        return text

    # Caso 5 -> hello ou Hello ou HELLO ou H E L L O...
    elif text.lower().replace(" ", "") == "hello":
        return "Olá"

    # Caso 6 -> @123456 is a great movie.
    elif text.startswith("@") and len(split_text) > 1 and split_text[0][1:].isdigit():
        text_after_number = " ".join(split_text[1:])
        translated_text_after_number = _translate_remaining(text_after_number, row)
        return split_text[0] + " " + translated_text_after_number

    # Não se aplica a nenhuma das condições acima
    else:
        return None

for index, row in df_messages.iterrows():
    success = False
    max_attempts = 2 
    attempts = 0

    # Verificação da necessidade de tradução customizada
    custom_translated = _custom_translation(row['text'], row)

    if custom_translated:
        translated_text.append(custom_translated)
        current_message_count += 1
        success_count += 1
        print(f"({current_message_count}/{total_messages}) Tradução customizada: {row['text']} -> {custom_translated}")
        df_messages.loc[index, 'text_translated'] = translated_text[-1]
        df_messages.to_parquet(f'data/processed/interim/interim_translated_{df_name[:-8]}.parquet', index=False)
        
    else:
        prompt = template.format(row['text'])

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
                current_message_count += 1
                print(f"({current_message_count}/{total_messages}) Tradução bem-sucedida: {row['text']} -> {clean_answer}")
            except Exception as e:
                attempts += 1
                print(f"({current_message_count}/{total_messages}) Erro ao traduzir a mensagem: {row['text']}. Tentativa {attempts}. Erro: {e}")
                time.sleep(5)  # Esperar 5 segundos antes de tentar novamente

        time.sleep(5)

        if not success:
            translated_text.append(row['text'])
            failure_count += 1
            current_message_count += 1
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