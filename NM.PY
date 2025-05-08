import os
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Folder containing speeches (e.g., "speeches/"), filenames like "1963_JFK.txt"
SPEECH_FOLDER = "speeches"

# Load and analyze speeches
data = []

for filename in sorted(os.listdir(SPEECH_FOLDER)):
    if filename.endswith(".txt"):
        path = os.path.join(SPEECH_FOLDER, filename)
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
            scores = analyzer.polarity_scores(text)
            # Extract date from filename (e.g., "1963_JFK.txt")
            date_str = filename.split('_')[0]
            try:
                date = datetime.strptime(date_str, '%Y')
            except ValueError:
                continue
            data.append({
                'date': date,
                'filename': filename,
                'compound': scores['compound'],
                'positive': scores['pos'],
                'neutral': scores['neu'],
                'negative': scores['neg']
            })

# Convert to DataFrame
df = pd.DataFrame(data)
df.sort_values(by='date', inplace=True)

# Plot compound sentiment over time
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['compound'], marker='o', color='blue', label='Compound Sentiment')
plt.title('Sentiment Shift in Political Speeches Over Time')
plt.xlabel('Year')
plt.ylabel('Compound Sentiment Score')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
