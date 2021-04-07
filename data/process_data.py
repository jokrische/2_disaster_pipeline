import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT: filepath of messages.csv, filepath of categories.csv
    OUTPUT: merged Dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id') 
    return df


def clean_data(df):
    '''
    INPUT: dataframe
    OUTPUT: cleaned dataframe: categories as columns, no duplicates
    '''
    categories = df['categories'].str.split(";",expand=True)
    row = categories.loc[0]
    category_colnames =row.str[:-2]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].str.replace("2", "1")
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    df = df.drop(['categories'], axis =1)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(keep='last', inplace = True)
    
    return df


def save_data(df, database_filename):
    '''
    INPUT: dataframe, name of database
    OUTPUT: dataframe is stored to database
    '''
    CONNECTION_STRING = f"sqlite:///{database_filename}"
    engine = create_engine(CONNECTION_STRING)
    if not engine.dialect.has_table(engine,'disaster_data'):
        df.to_sql('disaster_data', engine, index=False) 


def main():
    '''
    INPUT: messages_filepath, categories_filepath, database_filepath
    OUTPUT: calls functions load_data, clean_data and save_data
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()