import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

def download_file(url, local_filename):
    """Download a file from a URL to a local file."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def analyze_cdr_data(file_path):
    # Read the CDR data into a pandas DataFrame
    df = pd.read_csv(file_path, sep='\t', header=None)
    
    # Assign column names based on the number of columns
    num_columns = len(df.columns)
    if num_columns == 8:
        df.columns = ['user_id', 'timestamp', 'cell_id', 'sms_out', 'sms_in', 'call_out', 'call_in', 'internet']
    else:
        df.columns = ['col_' + str(i) for i in range(num_columns)]
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Print the number of rows and columns
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    
    # Print the data types of each column
    print("\nData types:")
    print(df.dtypes)
    
    # Print the count, mean, min, and max for each column
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    # Print the unique user IDs and their counts if user_id exists
    if 'user_id' in df.columns:
        print("\nUnique User IDs:")
        print(df['user_id'].value_counts())
    
    # Aggregate statistics if relevant columns exist
    if all(col in df.columns for col in ['sms_out', 'sms_in', 'call_out', 'call_in', 'internet']):
        total_sms_out = df['sms_out'].sum()
        total_sms_in = df['sms_in'].sum()
        total_call_out = df['call_out'].sum()
        total_call_in = df['call_in'].sum()
        total_internet = df['internet'].sum()
        
        avg_sms_out_per_user = total_sms_out / df['user_id'].nunique()
        avg_sms_in_per_user = total_sms_in / df['user_id'].nunique()
        avg_call_out_per_user = total_call_out / df['user_id'].nunique()
        avg_call_in_per_user = total_call_in / df['user_id'].nunique()
        avg_internet_per_user = total_internet / df['user_id'].nunique()
        
        # Print aggregate statistics
        print("\nAggregate Statistics:")
        print(f"Total SMS Out: {total_sms_out:.2f}")
        print(f"Total SMS In: {total_sms_in:.2f}")
        print(f"Total Call Out: {total_call_out:.2f}")
        print(f"Total Call In: {total_call_in:.2f}")
        print(f"Total Internet: {total_internet:.2f}")
        print(f"Average SMS Out per User: {avg_sms_out_per_user:.2f}")
        print(f"Average SMS In per User: {avg_sms_in_per_user:.2f}")
        print(f"Average Call Out per User: {avg_call_out_per_user:.2f}")
        print(f"Average Call In per User: {avg_call_in_per_user:.2f}")
        print(f"Average Internet per User: {avg_internet_per_user:.2f}")
    
    # Visualizations if relevant columns exist
    if all(col in df.columns for col in ['timestamp', 'sms_out', 'sms_in', 'call_out', 'call_in', 'internet']):
        # Temporal patterns
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df, x='timestamp', y='sms_out', label='SMS Out', ax=ax)
        sns.lineplot(data=df, x='timestamp', y='sms_in', label='SMS In', ax=ax)
        sns.lineplot(data=df, x='timestamp', y='call_out', label='Call Out', ax=ax)
        sns.lineplot(data=df, x='timestamp', y='call_in', label='Call In', ax=ax)
        sns.lineplot(data=df, x='timestamp', y='internet', label='Internet', ax=ax)
        ax.set_xlabel('Time of Day')
        ax.set_ylabel('Activity (Count)')
        ax.set_title('Temporal Patterns of User Activity on a Tuesday in early December')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Save the figure
        fig.savefig('temporal_patterns.png')

        # Temporal patterns excluding internet
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df, x='timestamp', y='sms_out', label='SMS Out', ax=ax)
        sns.lineplot(data=df, x='timestamp', y='sms_in', label='SMS In', ax=ax)
        sns.lineplot(data=df, x='timestamp', y='call_out', label='Call Out', ax=ax)
        sns.lineplot(data=df, x='timestamp', y='call_in', label='Call In', ax=ax)
        ax.set_xlabel('Time of Day')
        ax.set_ylabel('Activity (Count)')
        ax.set_title('Temporal Patterns of User Activity (Excluding Internet) on a Tuesday in early December')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Save the figure
        fig.savefig('temporal_patterns_excluding_internet.png')

        # Explanatory note below the plot
        print("\nExplanatory Note:")
        print("The plot shows the temporal patterns of user activity on a Tuesday in early December.")
        print("Activity represents the count of SMS (sent and received), calls (outgoing and incoming), and internet usage.")
        print("The lines depict how the activity levels change throughout the day for each type of activity.")
    
    if 'user_id' in df.columns and all(col in df.columns for col in ['sms_out', 'sms_in', 'call_out', 'call_in', 'internet']):
        # User activity distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        user_activity = df.groupby('user_id')[['sms_out', 'sms_in', 'call_out', 'call_in', 'internet']].sum()
        user_activity.plot(kind='bar', stacked=True, ax=ax)
        ax.set_xlabel('User')
        ax.set_ylabel('Total Activity (Count)')
        ax.set_title('User Activity Distribution')
        ax.legend(title='Activity Type')
        plt.xticks([])  # Remove x-axis tick labels
        plt.tight_layout()
        plt.show()
        
        # Save the figure
        fig.savefig('user_activity_distribution.png')
        
        # Analytical note
        print("\nAnalytical Note:")
        print("The plot depicts the distribution of total activity across different users.")
        print("Each bar represents a unique user, and the height of the bar indicates the total activity count for that user.")
        print("The stacked colors within each bar represent the breakdown of activity types (SMS Out, SMS In, Call Out, Call In, Internet) for each user.")
        print("Users are ordered along the x-axis based on their total activity count.")
        
        # Footnote
        print("\nFootnote:")
        print("In this analysis, 'User' refers to a unique 'user_id' in the dataset.")
        print("It's important to note that 'user_id' may not always correspond one-to-one with actual users.")
        print("Depending on the data collection and anonymization process, a single user may have multiple 'user_id's or vice versa.")
        print("Keep this in mind when interpreting the results and drawing conclusions about individual user behavior.")
    
    # Save descriptive statistics to CSV
    df.describe().to_csv('descriptive_statistics.csv')
    
    if all(col in df.columns for col in ['sms_out', 'sms_in', 'call_out', 'call_in', 'internet']):
        # Save aggregate statistics to CSV
        aggregate_stats = pd.DataFrame({
            'Metric': ['Total SMS Out', 'Total SMS In', 'Total Call Out', 'Total Call In', 'Total Internet',
                       'Average SMS Out per User', 'Average SMS In per User', 'Average Call Out per User',
                       'Average Call In per User', 'Average Internet per User'],
            'Value': [total_sms_out, total_sms_in, total_call_out, total_call_in, total_internet,
                      avg_sms_out_per_user, avg_sms_in_per_user, avg_call_out_per_user,
                      avg_call_in_per_user, avg_internet_per_user]
        })
        aggregate_stats.to_csv('aggregate_statistics.csv', index=False)

# Analyze the CDR data for Milan and Trentino
file_url = "https://dataverse.harvard.edu/api/access/datafile/:persistentId/?persistentId=doi:10.7910/DVN/EGZHFV/1MIYTI"
local_file_name = "sms-call-internet-mi-2013-12-03.txt"
# Download the file
download_file(file_url, local_file_name)

# Now, analyze the downloaded CDR data
analyze_cdr_data(local_file_name)

