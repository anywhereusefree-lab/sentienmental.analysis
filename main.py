#!/usr/bin/env python3
"""
Main application for E-commerce Sentiment Analysis
This script demonstrates sentiment analysis on product reviews
"""

from sentiment_analyzer import SentimentAnalyzer
from dataset import get_dummy_reviews
import sys

def main():
    """Main function to run sentiment analysis"""
    
    print("="*80)
    print("E-COMMERCE PRODUCT REVIEWS - SENTIMENT ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Load dummy reviews
    reviews = get_dummy_reviews()
    print(f"\nLoading {len(reviews)} product reviews...")
    analyzer.add_reviews(reviews)
    
    # Analyze all reviews
    print("Analyzing sentiments...")
    df = analyzer.analyze_all_reviews()
    
    # Display summary statistics
    analyzer.get_sentiment_summary(df)
    
    # Display negative reviews
    print("\n")
    analyzer.display_negative_reviews(df, limit=5)
    
    # Display neutral reviews
    print("\n")
    analyzer.display_neutral_reviews(df, limit=5)
    
    # Display positive reviews
    print("\n")
    analyzer.display_positive_reviews(df, limit=5)
    
    # Save results to CSV
    analyzer.save_to_csv(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_sentiment_distribution(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nFiles generated:")
    print("  - sentiment_analysis_results.csv (detailed results)")
    print("  - sentiment_analysis_report.png (visualization)")

if __name__ == "__main__":
    main()
