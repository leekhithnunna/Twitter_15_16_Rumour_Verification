import pandas as pd

def add_numeric_labels():
    """Add numeric labels to the existing Excel file"""
    
    # Read the existing Excel file
    df = pd.read_excel('combined_twitter_dataset.xlsx')
    
    # Define the label mapping
    label_mapping = {
        'false': 0,
        'true': 1,
        'unverified': 2,
        'non-rumor': 3
    }
    
    # Create the numeric label column
    df['numeric_label'] = df['label'].map(label_mapping)
    
    # Check if there are any unmapped labels
    unmapped = df[df['numeric_label'].isna()]
    if not unmapped.empty:
        print("Warning: Found unmapped labels:")
        print(unmapped['label'].unique())
    
    # Save the updated DataFrame back to Excel
    df.to_excel('combined_twitter_dataset.xlsx', index=False, engine='openpyxl')
    
    print("Successfully added numeric labels to the Excel file!")
    print(f"Total tweets: {len(df)}")
    print("\nLabel mapping applied:")
    print("false -> 0")
    print("true -> 1") 
    print("unverified -> 2")
    print("non-rumor -> 3")
    
    print("\nNumeric label distribution:")
    print(df['numeric_label'].value_counts().sort_index())
    
    print("\nOriginal vs Numeric label comparison:")
    comparison = df.groupby(['label', 'numeric_label']).size().reset_index(name='count')
    print(comparison)
    
    print("\nFirst 5 rows with new numeric_label column:")
    print(df[['tweet_id', 'label', 'numeric_label', 'dataset']].head())

if __name__ == "__main__":
    add_numeric_labels()