Sample Sentences with Top 3 Emotions, Sarcasm Probability. If Sarcasm Probability is > 50%, the input will be classified as sarcasm.

# Normal Examples

1. "This phone broke after just one week of use."
Sadness: Probability 27.44%
Neutral: Probability 24.43%
Disappointment: Probability 22.81%
Sarcasm Prediction: Sarcastic (Probability: 76.00%)

2. "I can't believe how comfortable these shoes are!"
Surprise: Probability 73.17%
Neutral: Probability 12.31%
Curiosity: Probability 4.85%
Sarcasm Prediction: Not Sarcastic (Probability: 1.02%)

3. "I admire the craftsmanship of this handmade necklace."
Admiration: Probability 87.78%
Excitement: Probability 7.63%
Approval: Probability 1.36%
Sarcasm Prediction: Not Sarcastic (Probability: 1.68%)

4. "I felt so embarrassed when this product didn't work in front of my friends."
Embarrassment: Probability 64.42%
Fear: Probability 9.05%
Disappointment: Probability 8.21%
Sarcasm Prediction: Not Sarcastic (Probability: 33.08%)

Overall, the classification is pretty accurate. However, sentences 1 and 4 have rather high sarcasm ratings, with 4 being falsely classified as sarcastic.

# Negation Examples

1. "This laptop is not only lightweight but also incredibly powerful."
Admiration: Probability 41.0%
Neutral: Probability 33.0%
Approval: Probability 5.37%
Sarcasm Prediction: Not Sarcastic (Probability: 3.54%)

2. "I wasn't sure about the design, but this phone is actually quite stylish."
Confusion: Probability 39.56%
Neutral: Probability 23.58%
Approval: Probability 9.16%
Sarcasm Prediction: Not Sarcastic (Probability: 0.81%)

3. "I didn't expect much from this restaurant, but the food was surprisingly delicious."
Surprise: Probability 72.76%
Disapproval: Probability 8.59%
Neutral: Probability 4.2%
Sarcasm Prediction: Not Sarcastic (Probability: 1.30%)

4. "I had my doubts, but this skincare product has done wonders for my skin."
Caring: Probability 19.64%
Neutral: Probability 15.85%
Sadness: Probability 13.23%
Sarcasm Prediction: Not Sarcastic (Probability: 1.99%)

It can be seen that most classifications are pretty accurate, even though negations such as "didn't", "is not", or "wasn't" were used. Even the implied negation through "I had my doubts, but" was classified rather accurately.

# Sarcasm Examples

1. "Oh, great, another pair of headphones that tangle themselves magically!"
Admiration: Probability 54.62%
Neutral: Probability 16.12%
Excitement: Probability 13.52%
Sarcasm Prediction: Sarcastic (Probability: 80.93%)

2. "Wow, this coffee machine manages to burn every single cup! What a talent!"
Surprise: Probability 25.34%
Admiration: Probability 20.5%
Excitement: Probability 15.44%
Sarcasm Prediction: Sarcastic (Probability: 65.25%)

3. "I absolutely love how this laptop randomly crashes, adds excitement to my day!"
Love: Probability 70.28%
Joy: Probability 11.2%
Amusement: Probability 9.03%
Sarcasm Prediction: Not Sarcastic (Probability: 6.93%)

4. "This camera takes amazing photos... of blurry objects, of course!"
Admiration: Probability 77.99%
Excitement: Probability 10.14%
Neutral: Probability 3.7%
Sarcasm Prediction: Not Sarcastic (Probability: 43.29%)

In most cases, the sarcasm ratings are pretty high, but sentence 3 wasn't correctly classified as sarcasm. While sentence 4 isn't technically classified as sarcastic, the high rating of 43.29% definitely indicates that it might be sarcastic. Since the classification threshold can be set arbitrarily, this could also have been classified as "possibly sarcastic".