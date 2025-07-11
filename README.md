# Burnout Detector - Automatic sentiment analysis and classification

This project takes text data from Reddit to analyze users emotional state and detect signs of **burnout**.
The aim is to identify user posts that could indicate **burnout at work**.

---

## Technologies used throughout the code:

- **Python 3**
- **Jupyter Notebook**
- **Pandas** – data manipulation
- **scikit-learn** – TF-IDF, Random Forest, classification matrix
- **NLTK** – preprocessing text (tokenization, stopwords)
- **TextBlob** – sentiment analysis (polarity and subjectivity)
- **Matplotlib / WordCloud** – data visualization
- **Seaborn** – heatmap for confusion matrix

---

# Explanation of steps 

### 1. Data collection
We researched and searched for subreddits where we found this job burnout very visible. Thus we chose the following subreddits: 
- r/burnout, 
- r/overemployed, 
- r/jobs, 
- r/depression, 
- r/workreform, 
- r/mentalhealth.

For data collection I used a reddit API through PRAW library, extracting relevant posts using .hot, .new, .top methods.

Thus I got **11.261**.

--- 

### 2. Text Cleaning and Preprocessing
At this stage:
- all text has been converted to lowercase,
- links, punctuation and special characters have been removed,
- multiple spaces and newlines have been removed,
- the title and the text of the post were concatenated.

! Also, noticing later when I used **topic modeling**, I realized that we have a topic about a Burnout Paradise Remastered game, which is not relevant for the purpose of the analysis, so I decided that it is noise, and I removed the specific words for this game: 'burnout paradise', 'video game', 'Big Surf Island', 'EA', 'hunter mesquite', 'criterion games', etc.

---

### 3. Word Frequency in Posts - Text Vectorization with TF-IDF

<img width="2140" height="1277" alt="image" src="https://github.com/user-attachments/assets/3fc2f98a-2fca-420c-99af-a341f8f9267c" />

From this analysis, using TF-IDF we clearly observe terms that reflect the theme of job burnout.

- Emotional terms: feel, help, really, need. Here emotional vulnerability and need for support are emphasized.
- Occupational context: job, work, time, years. This indicates exactly what is being talked about in the posts.
- Ambiguous words: just, like. If we look at these terms separately, they can seem ambiguous
- High self-referentiality: im, ive, dont, know. We notice the predominance of the person I, which shows that the authors have personal testimonies.
- Apparently ambiguous words: just, like. If we look at these separate terms, they have a generic appearance, but if we look at the posts, they are used in expressions of frustration, to amplify the mood ("just tired", "like I can't anymore").

Worth mentioning, the word burnout appears with high frequency, but does not dominate the graph. It means that users sometimes discuss the problem by explicitly naming it in posts.

### 4. Part-of-Speech (POS) Analysis

Using POS Tagging we observe a lingivstic pattern, how users express their job burnout.

<img width="531" height="504" alt="image" src="https://github.com/user-attachments/assets/9bbd9da0-5a00-47ee-8c77-d754fe34b59e" />


We can observe that the most dominant categories are:
- Verbs (42%), which emphasizes the way of addressing, an action based language. Users describe what they do, feel or experience.
- Nouns (34%), in this category we use terms from the work environment, emotions and personal references.

Then , followed in small proportions are: 
- Adjectives (4%), this category is crucial for emotional intensity
- Adverbs (4%), amplify users' feelings and experiences.

So we can conclude from this analysis using POS Tagging, that discussions are centered on action (verbs) and experiences (nouns).
Users are actively describing their situations rather than tagging emotions.

---

### 5. Topic Modeling - LDA (Latent Dirichlet Allocation)
We applied the LDA algorithm , and identified 5 main themes that emerge from the data:

Topic 1 : Workplace and daily routine (job, work, oe, jobs, just, company, j2, time, j1, got)

Topic 2 : Informal political discussions (bernie, sanders, bro, lol, unions, war, car, traffic, labor, song)

Topic 3 : Critique of the system and social inequality(burnout, billionaires, crash, america, billionaire, rich, tax, real, car, cars)

Topic 4 : Burnout and labor movements(burnout, workers, revenge, game, wage, american, union, billionaires, pay, healthcare)

Topic 5 : Expression and personal emotions(just, like, im, feel, dont, life, want, know, people, don)

We observe a thematic diversity, which confirms that Burnout is perceived as an individual problem, but also as a symptom of systematic dysfunctions.
So, Topic 1, reflects the work specific termnology, Topic 2 shows the informal and satirical aspect of the discussions, Topic 3 and 4 make a connection between burnout and wider social problems, and Topic 5 highlights the personal emotional side.

---

### 6. Sentiment Analysis - VADER (Valence Aware Dictionary and sentiment Reasoner)

<img width="2238" height="1253" alt="image" src="https://github.com/user-attachments/assets/5c3dcd03-8f23-498d-9e2e-d05a87056108" />

In order to analyze the sentiment of the posts we used VADER, which shows a polarized distribution of emotions in the job burnout posts.

From what we can see, we do not have a normal distribution, meaning that users choose to express themselves in extreme terms, either very negative or very positive.

The highest peak is at neutrality, this shows that most of the discussions are informative, with users not expressing experiences with explicit emotional value.

---

### 7. Wordcloud - Vocabulary comparison

<img width="1589" height="431" alt="image" src="https://github.com/user-attachments/assets/2a0988fb-14a5-4734-9c0e-8292c86e0d1b" />

Visual analysis reveals clear differences in the vocabulary used:

- Words Associated with **BURNOUT**:

Emotional states: "feel", "depression", "anxiety"

Limitations: "can't", "don't", "never"

Persistence of time: "always", "year", "day"


- Words associated with **NON-BURNOUT**:

Solution-oriented: "help", "make", "good", "better"

Professional context: "job", "work", "company", "money"

Positive action: "got", "get", "take", "find"


Burnout posts use an emotional and limitation-centered vocabulary, whereas non-Burnout posts are action-oriented and practical problem-solving.

---

### 8. Classification with Random Forest

<img width="1525" height="1212" alt="image" src="https://github.com/user-attachments/assets/55824559-6532-4f1f-b5a4-1e87553a1c56" />

- Model performance:

Overall accuracy: 67.90%

Parameters: 290 trees, unlimited depth

Features: TF-IDF matrix with text vectorization



Observe that the model best identifies neutral and positive posts, burnout detection is more challenging.
The result is acceptable for a classification into 3 classes per text.

---

### 9. Model Optimization with Grid Search

<img width="742" height="590" alt="image" src="https://github.com/user-attachments/assets/f810f8ff-58fc-4cee-90c6-28695e32964d" />


I tried to improve the model by Grid Search to find optimal hyperparameters for the model by systematically testing all combinations in a range, by using F1-macro (balances performance across all classes), also I used cross-validation (cv = 3) for more robust results, and class balancing by class_weight='balanced'.

Finally we obtained an improvement in real burnout detection, reduced cirtic errors, i.e. fewer missed burnout cases.

The accuracy on the best end is 67.81%.

Even if we do not see an improvement in the accuracy, we observe that our model better recognizes job burnout, reduces the errors and increases the sensitivity for critical cases.
