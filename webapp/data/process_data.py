import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load data from the message and categories csv files and drop duplicates
    :param messages_filepath: path to message file
    :param categories_filepath: path to categories file
    :return: DataFrame with merged data
    """
    messages = pd.read_csv(messages_filepath).drop_duplicates()
    categories = pd.read_csv(categories_filepath).drop_duplicates()
    return pd.merge(messages, categories, on='id')


def clean_data(df):
    """
    Parse categories column into numerical matrix
    :param df: DataFrame with raw data format
    :return: DataFrame with expanded categories
    """
    categories = df.categories.str.split(pat=';', expand=True)
    category_colnames = categories.iloc[0].str.split(pat='-').apply(lambda arr: arr[0])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = pd.to_numeric(categories[column].str[-1])
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Save DataFrame to sqlite table
    :param df: DataFrame to be saved
    :param database_filename: path to sqlite file
    :return: None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
