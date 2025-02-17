import pandas as pd

# carregar dados raw
def read_raw_data(dateset_name):

    dateset_name = rf"raw_data/{dateset_name}"

    try:
        df_raw = pd.read_excel(
            dateset_name,
            engine="openpyxl",
            dtype={
                "id": str,    
                "texto_tweet": str,   
                "idioma_tweet": str,
                "sentimento_tweet": int 
            }
        )
    except FileNotFoundError:
        print("File not found. Check the file path.")
    except Exception as e:
        print(f"Error: {e}")
    return df_raw

# trata as linhas consecutivas nulls
def merge_previous_rows(df, target_col=1, source_col=0):
    """
    :param df: DataFrame a ser processado
    :param target_col: Indice da coluna a ser verificada para valores ausentes (segunda coluna)
    :param source_col: Indice da coluna de onde os dados serão mesclados (primeira coluna)
    :return: DataFrame processado
    """
    cols = df.columns
    current_block = []
    
    # processar todas as linhas 
    for i in reversed(range(len(df))):
        # verificar a linha se ela é null
        if pd.isna(df.iat[i, target_col]):
            current_block.append(i)
        else:
            if current_block:
                # trata as linhas consecutivas nulls
                start = min(current_block)
                end = max(current_block)
                
                # junta pra linha anterior
                merged = " ".join(df.iloc[start:end+1, source_col].astype(str))
                df.iat[i, target_col] = f"{df.iat[i, target_col]} {merged}".strip()
                
                # limpa 
                df.iloc[start:end+1, source_col] = ""
                df.iloc[start:end+1, target_col] = ""
                
                current_block = []
    
    # trata as linhas consecutivas nulls
    if current_block:
        start = min(current_block)
        end = max(current_block)
        df.iloc[start:end+1, source_col] = ""
        df.iloc[start:end+1, target_col] = ""

    return df

def merge_multi_columns(df_raw):
    
    cols_name = ['id','texto_tweet','idioma_tweet','sentimento_tweet']
    # preencher os nulls primeiro e junta pra uma coluna 
    df = df_raw.fillna("").apply(lambda row: ' '.join(row.values), axis=1)
    
    # usa ," pra separar a coluna em colunas
    df = df.str.split(',"', expand=True).apply(lambda x: x.str.strip('"'))
    df.columns = cols_name
    
    # remove as linhas nao tem dados
    df = df[df.id.str.strip()!=''].reset_index(drop=True)
    
    # aplicar a funcao 
    df = merge_previous_rows(df.copy(), target_col=1, source_col=0)
    
    # remove as linhas foram pra linha anterior
    df = df[df['id']!=''].reset_index(drop=True)
    df = df.fillna('')
    
    return df

def mescla_columns(df):
    
    # junta as colunas mescladas
    for i in range(len(df)-1):
        if (df.loc[i, 'sentimento_tweet'] == '')&(df.loc[i+1, 'sentimento_tweet'] == ''):
            df.loc[i, 'texto_tweet'] += ' ' + df.loc[i+1, 'id']
            df.loc[i, 'idioma_tweet'] = df.loc[i+1, 'texto_tweet']
            df.loc[i, 'sentimento_tweet'] = df.loc[i+1, 'idioma_tweet']
    df = df[df['sentimento_tweet']!=''].replace('"','').reset_index(drop=True)
    df['sentimento_tweet'] = df['sentimento_tweet'].str.replace('"','').str.strip()
    df['texto_tweet'] = df['texto_tweet'].str.replace('  ',' ')
    
    return df

def clean_save_raw_data(dateset_name,output_name):
    
    df_raw = read_raw_data(dateset_name)
    df = merge_multi_columns(df_raw)
    df = mescla_columns(df)

    output_name = f'preprocessed/{output_name}'
    
    return df.to_csv(output_name,index=False) 







