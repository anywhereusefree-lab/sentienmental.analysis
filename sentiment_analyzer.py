"""
Sentiment Analysis for E-commerce Product Reviews
This application analyzes customer reviews using TextBlob and VADER sentiment analysis
"""

from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import nltk

# Download required NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentAnalyzer:
    """Class to analyze sentiment of e-commerce product reviews"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.reviews_data = []
    
    def add_reviews(self, reviews_list):
        """Add reviews to the analyzer"""
        self.reviews_data = reviews_list
    
    def analyze_with_textblob(self, text):
        """
        Analyze sentiment using TextBlob
        Returns: polarity (-1 to 1), subjectivity (0 to 1)
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = "POSITIVE"
        elif polarity < -0.1:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        
        return {
            'polarity': round(polarity, 3),
            'subjectivity': round(subjectivity, 3),
            'sentiment': sentiment
        }
    
    def analyze_with_vader(self, text):
        """
        Analyze sentiment using VADER
        Returns: compound score, neg, neu, pos
        """
        scores = self.sia.polarity_scores(text)
        
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = "POSITIVE"
        elif compound <= -0.05:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        
        return {
            'compound': round(compound, 3),
            'negative': round(scores['neg'], 3),
            'neutral': round(scores['neu'], 3),
            'positive': round(scores['pos'], 3),
            'sentiment': sentiment
        }
    
    def analyze_all_reviews(self):
        """Analyze all reviews with both methods"""
        results = []
        
        for review in self.reviews_data:
            textblob_result = self.analyze_with_textblob(review['text'])
            vader_result = self.analyze_with_vader(review['text'])
            
            result = {
                'review_id': review['id'],
                'product': review['product'],
                'rating': review['rating'],
                'text': review['text'],
                'textblob_sentiment': textblob_result['sentiment'],
                'textblob_polarity': textblob_result['polarity'],
                'textblob_subjectivity': textblob_result['subjectivity'],
                'vader_sentiment': vader_result['sentiment'],
                'vader_compound': vader_result['compound'],
                'vader_negative': vader_result['negative'],
                'vader_neutral': vader_result['neutral'],
                'vader_positive': vader_result['positive'],
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_sentiment_summary(self, df):
        """Get summary statistics of sentiments"""
        textblob_counts = df['textblob_sentiment'].value_counts()
        vader_counts = df['vader_sentiment'].value_counts()
        
        print("\n" + "="*80)
        print("SENTIMENT ANALYSIS SUMMARY")
        print("="*80)
        
        print("\nTextBlob Analysis:")
        print("-" * 40)
        for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
            count = textblob_counts.get(sentiment, 0)
            percentage = (count / len(df)) * 100
            print(f"{sentiment:10s}: {count:3d} reviews ({percentage:5.1f}%)")
        
        print("\nVADER Analysis:")
        print("-" * 40)
        for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
            count = vader_counts.get(sentiment, 0)
            percentage = (count / len(df)) * 100
            print(f"{sentiment:10s}: {count:3d} reviews ({percentage:5.1f}%)")
        
        print("\nAverage Ratings by Sentiment (VADER):")
        print("-" * 40)
        for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
            subset = df[df['vader_sentiment'] == sentiment]
            if len(subset) > 0:
                avg_rating = subset['rating'].mean()
                print(f"{sentiment:10s}: {avg_rating:.2f} / 5.0")
        
        return {
            'textblob': textblob_counts.to_dict(),
            'vader': vader_counts.to_dict()
        }
    
    def display_negative_reviews(self, df, limit=5):
        """Display negative sentiment reviews"""
        print("\n" + "="*80)
        print("NEGATIVE SENTIMENT REVIEWS")
        print("="*80)
        
        negative_reviews = df[df['vader_sentiment'] == 'NEGATIVE'].sort_values('vader_compound')
        
        if len(negative_reviews) == 0:
            print("\nNo negative reviews found!")
            return
        
        negative_reviews = negative_reviews.head(limit)
        
        for idx, (_, row) in enumerate(negative_reviews.iterrows(), 1):
            print(f"\n[{idx}] Review ID: {row['review_id']} | Rating: {row['rating']}/5")
            print(f"    Product: {row['product']}")
            print(f"    VADER Score: {row['vader_compound']} (Neg: {row['vader_negative']}, Neu: {row['vader_neutral']}, Pos: {row['vader_positive']})")
            print(f"    Text: {row['text']}")
            print("-" * 80)
    
    def display_neutral_reviews(self, df, limit=5):
        """Display neutral sentiment reviews"""
        print("\n" + "="*80)
        print("NEUTRAL SENTIMENT REVIEWS")
        print("="*80)
        
        neutral_reviews = df[df['vader_sentiment'] == 'NEUTRAL'].sort_values('vader_compound')
        
        if len(neutral_reviews) == 0:
            print("\nNo neutral reviews found!")
            return
        
        neutral_reviews = neutral_reviews.head(limit)
        
        for idx, (_, row) in enumerate(neutral_reviews.iterrows(), 1):
            print(f"\n[{idx}] Review ID: {row['review_id']} | Rating: {row['rating']}/5")
            print(f"    Product: {row['product']}")
            print(f"    VADER Score: {row['vader_compound']} (Neg: {row['vader_negative']}, Neu: {row['vader_neutral']}, Pos: {row['vader_positive']})")
            print(f"    Text: {row['text']}")
            print("-" * 80)
    
    def display_positive_reviews(self, df, limit=5):
        """Display positive sentiment reviews"""
        print("\n" + "="*80)
        print("POSITIVE SENTIMENT REVIEWS")
        print("="*80)
        
        positive_reviews = df[df['vader_sentiment'] == 'POSITIVE'].sort_values('vader_compound', ascending=False)
        
        if len(positive_reviews) == 0:
            print("\nNo positive reviews found!")
            return
        
        positive_reviews = positive_reviews.head(limit)
        
        for idx, (_, row) in enumerate(positive_reviews.iterrows(), 1):
            print(f"\n[{idx}] Review ID: {row['review_id']} | Rating: {row['rating']}/5")
            print(f"    Product: {row['product']}")
            print(f"    VADER Score: {row['vader_compound']} (Neg: {row['vader_negative']}, Neu: {row['vader_neutral']}, Pos: {row['vader_positive']})")
            print(f"    Text: {row['text']}")
            print("-" * 80)
    
    def plot_sentiment_distribution(self, df):
        """Create visualization of sentiment distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # TextBlob Sentiment Distribution
        textblob_counts = df['textblob_sentiment'].value_counts()
        axes[0, 0].bar(textblob_counts.index, textblob_counts.values, color=['green', 'gray', 'red'])
        axes[0, 0].set_title('TextBlob Sentiment Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Reviews')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # VADER Sentiment Distribution
        vader_counts = df['vader_sentiment'].value_counts()
        axes[0, 1].bar(vader_counts.index, vader_counts.values, color=['green', 'gray', 'red'])
        axes[0, 1].set_title('VADER Sentiment Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Number of Reviews')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Polarity Distribution
        axes[1, 0].hist(df['textblob_polarity'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('TextBlob Polarity Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Polarity Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Compound Score Distribution
        axes[1, 1].hist(df['vader_compound'], bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('VADER Compound Score Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Compound Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sentiment_analysis_report.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'sentiment_analysis_report.png'")
        plt.show()
    
    def save_to_csv(self, df, filename='sentiment_analysis_results.csv'):
        """Save results to CSV"""
        df.to_csv(filename, index=False)
        print(f"\n✓ Results saved to '{filename}'")
