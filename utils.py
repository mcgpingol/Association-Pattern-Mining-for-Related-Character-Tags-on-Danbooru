# import libraries
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Image, display, HTML

import fim

from pyspark.sql.functions import col
from pyspark.sql import SparkSession

import requests
from bs4 import BeautifulSoup


# functions

def extract_dataset():
    """Dump pandas DataFrame of file into pickle."""
    spark = (SparkSession
             .builder
             .master('local[*]')
             .getOrCreate())
    fpath = ('/mnt/data/public/tagged-anime-illustrations/danbooru-metadata/'
             '201700.json')
    data = (spark.read.json(fpath)
            .select('id', col('tags.name').alias('tag'),
                    col('tags.category').alias('cat'))
            .orderBy('id')).toPandas()
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
    spark.stop()
    
    
def load_dataset():
    """Load DataFrame from pkl file."""
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def top10_tag_dist(df):
    """Display top 10 tags based on frequency."""
    (df['tag'].explode().value_counts(ascending=True).tail(10)
     .plot.barh(figsize=(10, 6), color='#006DFC'))
    plt.xlabel('Frequency')
    plt.ylabel('Tags')
    plt.title('Top 10 Tags')
    plt.show()
    
    
def category_dist(df):
    """Display distribution of unique tags in each category."""
    idx = {"0":"general", "1":"artist", "2":"???", "3":"copyrights",
           "4":"characters", "5":"meta"}
    (df['cat'].explode().value_counts().rename(index=idx)
     .plot.bar(figsize=(10, 6), color='#006DFC'))
    plt.xlabel('Category')
    plt.xticks(rotation=0)
    plt.ylabel('Frequency')
    plt.title('Tag Category Distribution')
    plt.show()
    
    
def img_category_dist(df):
    """Display distribution of images using a tag category."""
    idx = {"0":"general", "1":"artist", "2":"???", "3":"copyrights",
           "4":"characters", "5":"meta"}
    (df['cat'].apply(lambda x: set(x)).explode().value_counts().rename(index=idx)
     .plot.bar(figsize=(10, 6), color='#006DFC'))
    plt.xlabel('Category')
    plt.xticks(rotation=0)
    plt.ylabel('Frequency')
    plt.title('Image Tag Category Distribution')
    plt.show()
    

def category_dict(df):
    """
    Returns a dictionary with categories as the keys and list of tags
    under that category.
    """
    return (df.explode(['tag', 'cat']).groupby('cat')['tag']
            .apply(lambda x: sorted(set(x))))


def char_dist(df):
    """Returns distribution of characters by number of tags"""
    df = df.explode(['tag', 'cat'])
    return (df[df['cat'] == '4']['tag'].value_counts()
            .reset_index().rename(columns={'index': 'char'}))


def filter_by_char(df, char):
    """Filters a specific character from DataFrame."""
    return (df[df['tag'].apply(lambda x: char in x)]
            .drop('cat', axis=1).explode('tag'))


def dist_tags_per_char(char_list):
    """Return plot of distribution of number of tags per character."""
    plt.figure(figsize=(10, 6))
    plt.hist(char_list['tag'], bins=30, color='#006DFC')
    plt.yscale('log')
    plt.xlabel('Number of Tags')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Tags Per Character')
    plt.show()
    

def filter_char(df, char):
    """Filters a specific character from DataFrame."""
    return (df[df['tag'].apply(lambda x: char in x)]
            .drop('cat', axis=1)['tag'])


def top_related_tags(df, char, supp=20):
    """Return top 25 tags based on confidence."""
    df = pd.DataFrame(fim.arules(df, supp, eval='l', report='l'))
    df = df[df[0] == char]
    return set(df[1].explode().dropna().drop_duplicates().values[:25])


def scrape_tags(char_list):
    """Scrape related tags for a list of characters from Danbooru."""
    dict_tags = {}
    for i, v in enumerate(char_list):
        current_url = 'https://danbooru.donmai.us/related_tag?search%5Bquery%5D=' + v
        page = requests.get(current_url)
        soup = BeautifulSoup(page.content, 'html.parser')
        dict_tags[i] = set([t for t, v in soup.get('tags')])
    with open('scraped.pkl', 'wb') as f:
        pickle.dump(dict_tags, f)
        
        
def load_scraped_tags():
    """Load pickled dictionary of scraped tags."""
    with open('scraped.pkl', 'rb') as f:
        scraped_tags = pickle.load(f)
    return scraped_tags


def related_vs_scraped(data, char_list, scraped_tags):
    """"""
    df = char_list.head(5).append(char_list.iloc[550:555]).reset_index(drop=True)
    df['related_tags'] = df['char'].apply(lambda x: top_related_tags(filter_char(data, x), x))
    df['scraped_tags'] = pd.Series(scraped_tags)
    similarity = [(len(i.intersection(j)) / len(i.union(j))) for i, j in 
                  list(zip(df['related_tags'], df['scraped_tags']))]
    df['similarity'] = similarity
    return df.set_index('char', drop=True)


def plot_similarity(df_test):
    plt.figure(figsize=(12, 6))
    plt.barh(df_test.tail().index, df_test['similarity'].tail(), color=['#006DFC'])
    plt.barh(df_test.head().index, df_test['similarity'].head(), color=['grey'])
    plt.xlabel('Jaccard Similarity')
    plt.ylabel('Characters')
    plt.title('Similarity of Related Tags to Scraped Tags from Website')
    plt.legend(['Low Tag Frequency', 'High Tag Frequency'])
    plt.show()