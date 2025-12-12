import pandas as pd
import os

def load_twitter_data(folder_name):
    """Load tweets and labels from a specific folder"""
    tweets_file = os.path.join(folder_name, 'source_tweets.txt')
    labels_file = os.path.join(folder_name, 'label.txt')
    
    # Load tweets
    tweets_data = {}
    with open(tweets_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    tweet_id = parts[0]
                    tweet_text = parts[1]
                    tweets_data[tweet_id] = tweet_text
    
    # Load labels
    labels_data = {}
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and ':' in line:
                label, tweet_id = line.split(':', 1)
                labels_data[tweet_id] = label
    
    return tweets_data, labels_data

def combine_twitter_datasets():
    """Combine Twitter datasets from both folders into a single DataFrame"""
    
    # Load data from both folders
    tweets_15, labels_15 = load_twitter_data('twitter15')
    tweets_16, labels_16 = load_twitter_data('twitter16')
    
    # Combine the data
    all_tweets = {**tweets_15, **tweets_16}
    all_labels = {**labels_15, **labels_16}
    
    # Create a list to store the combined data
    combined_data = []
    
    # Process all tweets that have labels
    for tweet_id in all_labels:
        if tweet_id in all_tweets:
            combined_data.append({
                'tweet_id': tweet_id,
                'tweet_text': all_tweets[tweet_id],
                'label': all_labels[tweet_id],
                'dataset': 'twitter15' if tweet_id in labels_15 else 'twitter16'
            })
    
    # Create DataFrame
    df = pd.DataFrame(combined_data)
    
    # Sort by tweet_id for consistency
    df = df.sort_values('tweet_id').reset_index(drop=True)
    
    return df

def main():
    """Main function to create the combined Excel dataset"""
    try:
        # Combine the datasets
        combined_df = combine_twitter_datasets()
        
        # Save to Excel
        output_file = 'combined_twitter_dataset.xlsx'
        combined_df.to_excel(output_file, index=False, engine='openpyxl')
        
        print(f"Successfully created combined dataset: {output_file}")
        print(f"Total tweets: {len(combined_df)}")
        print(f"Columns: {list(combined_df.columns)}")
        print("\nLabel distribution:")
        print(combined_df['label'].value_counts())
        print("\nDataset distribution:")
        print(combined_df['dataset'].value_counts())
        
        # Display first few rows
        print("\nFirst 5 rows:")
        print(combined_df.head())
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()