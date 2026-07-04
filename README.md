# E-Commerce Product Reviews - Sentiment Analysis Application

A comprehensive Python application for analyzing sentiment in e-commerce product reviews using multiple NLP techniques including TextBlob and VADER sentiment analysis.

## Features

- **Dual Sentiment Analysis**: 
  - TextBlob for polarity and subjectivity analysis
  - VADER (Valence Aware Dictionary and sEntiment Reasoner) for social media-style analysis
  
- **Review Classification**: Automatically categorizes reviews as POSITIVE, NEUTRAL, or NEGATIVE

- **Detailed Reporting**: 
  - Sentiment summary statistics
  - Display of negative, neutral, and positive reviews
  - Average ratings by sentiment
  
- **Data Visualization**: 
  - Sentiment distribution charts
  - Polarity and compound score histograms
  - Comparative analysis between methods

- **Data Export**: Save analysis results to CSV for further processing

- **Dummy Dataset**: Includes 30 pre-labeled product reviews across 6 product categories

## Installation

### Requirements
- Python 3.8+

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download required NLTK data (runs automatically on first execution):
```bash
python -m nltk.downloader vader_lexicon
```

## Usage

### Run the complete analysis:
```bash
python main.py
```

### Example usage in your own script:
```python
from sentiment_analyzer import SentimentAnalyzer
from dataset import get_dummy_reviews

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Load reviews
reviews = get_dummy_reviews()
analyzer.add_reviews(reviews)

# Analyze
df = analyzer.analyze_all_reviews()

# Get summary
analyzer.get_sentiment_summary(df)

# Display specific sentiments
analyzer.display_negative_reviews(df, limit=5)
analyzer.display_neutral_reviews(df, limit=5)
analyzer.display_positive_reviews(df, limit=5)

# Visualize and export
analyzer.plot_sentiment_distribution(df)
analyzer.save_to_csv(df)
```

## Dataset Structure

Each review contains:
- `id`: Unique review identifier
- `product`: Product name
- `rating`: Star rating (1-5)
- `text`: Review text

### Included Products:
1. Dell XPS 13 Laptop (5 reviews)
2. Sony WH-1000XM4 Headphones (5 reviews)
3. Apple Watch Series 7 (5 reviews)
4. Anker USB-C Cable 3-Pack (5 reviews)
5. JBL Flip 6 Speaker (5 reviews)
6. Logitech C920 Pro Webcam (5 reviews)

**Total: 30 reviews**

## Output Files

The application generates:

1. **sentiment_analysis_results.csv** - Detailed results with:
   - Review information
   - TextBlob scores (polarity, subjectivity)
   - VADER scores (compound, negative, neutral, positive)
   - Sentiment classifications from both methods

2. **sentiment_analysis_report.png** - Visualization containing:
   - TextBlob sentiment distribution
   - VADER sentiment distribution
   - Polarity score histogram
   - Compound score histogram

## Sentiment Classification Logic

### TextBlob Polarity:
- **POSITIVE**: Polarity > 0.1
- **NEUTRAL**: Polarity between -0.1 and 0.1
- **NEGATIVE**: Polarity < -0.1

### VADER Compound Score:
- **POSITIVE**: Compound ≥ 0.05
- **NEUTRAL**: Compound between -0.05 and 0.05
- **NEGATIVE**: Compound ≤ -0.05

## API Reference

### SentimentAnalyzer Class

#### Methods:

- `add_reviews(reviews_list)` - Add reviews to analyze
- `analyze_with_textblob(text)` - Analyze using TextBlob
- `analyze_with_vader(text)` - Analyze using VADER
- `analyze_all_reviews()` - Analyze all added reviews (returns DataFrame)
- `get_sentiment_summary(df)` - Print sentiment statistics
- `display_negative_reviews(df, limit=5)` - Display negative reviews
- `display_neutral_reviews(df, limit=5)` - Display neutral reviews
- `display_positive_reviews(df, limit=5)` - Display positive reviews
- `plot_sentiment_distribution(df)` - Generate visualizations
- `save_to_csv(df, filename)` - Export results to CSV

## Example Output

```
================================================================================
SENTIMENT ANALYSIS SUMMARY
================================================================================

TextBlob Analysis:
----------------------------------------
POSITIVE  :  12 reviews ( 40.0%)
NEUTRAL   :   8 reviews ( 26.7%)
NEGATIVE  :  10 reviews ( 33.3%)

VADER Analysis:
----------------------------------------
POSITIVE  :  11 reviews ( 36.7%)
NEUTRAL   :   9 reviews ( 30.0%)
NEGATIVE  :  10 reviews ( 33.3%)

Average Ratings by Sentiment (VADER):
----------------------------------------
POSITIVE  : 4.64 / 5.0
NEUTRAL   : 3.00 / 5.0
NEGATIVE  : 1.30 / 5.0
```

## Use Cases

- E-commerce platforms analyzing customer feedback
- Product quality monitoring and improvement
- Customer sentiment tracking over time
- Competitive analysis
- Review moderation and flagging
- Customer satisfaction metrics
- Marketing and PR analysis

## Dependencies

- **textblob**: Simple API for common NLP tasks
- **nltk**: Natural Language Toolkit for sentiment analysis
- **pandas**: Data manipulation and analysis
- **matplotlib**: Data visualization
- **numpy**: Numerical computing

## License

This project is open source and available for educational and commercial use.

## Author

Created for e-commerce sentiment analysis and NLP education.

## Support

For issues or questions, please refer to the documentation or create an issue in the repository.
