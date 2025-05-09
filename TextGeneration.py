import openai
import csv
from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns

# OpenRouter API config
openai.api_key = "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"
openai.api_base = "https://openrouter.ai/api/v1"

def generate_response(category, question):
    prompt = f"""
You are an AI sports journalist assistant. Generate a realistic and professional response to a sports interview question based on the given category.

Category: {category}
Question: "{question}"
Response:
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful sports assistant that generates realistic interview responses."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error generating response: {e}"

# Sample input questions
examples = [
    ("Game Strategy", "What adjustments did you make at halftime to improve the team’s attack?"),
    ("Game Strategy", "How does the team’s high press strategy help in regaining possession quickly?"),
    ("Game Strategy", "Can you explain how your team's defensive organization worked against their offensive setup?"),
    ("Game Strategy", "How do you assess the balance between attack and defense in today's game?"),
    ("Game Strategy", "What was your approach to neutralize their star player during the game?"),
    ("Game Strategy", "In what ways do you think the team can improve its counter-attacking play?"),
    ("Game Strategy", "How does the team adjust its game plan when playing against stronger opponents?"),
    ("Game Strategy", "What role does versatility in player positions play in your tactical decisions?"),
    
    ("Player Performance", "What do you think about [player's name] performance in today’s game?"),
    ("Player Performance", "Which player’s contribution stood out the most, and why?"),
    ("Player Performance", "Did the team get the expected performance from [player's name] today?"),
    ("Player Performance", "How would you rate the team's overall performance in comparison to individual performances?"),
    ("Player Performance", "What improvements do you think are necessary for [player's name] to make in the next match?"),
    ("Player Performance", "Which area of [player's name]'s game needs the most work moving forward?"),
    ("Player Performance", "Who do you think played an unsung role in today's game and why?"),
    ("Player Performance", "What did you think of [player's name]'s impact after coming off the bench today?"),
    
    ("Injury Updates", "How is [player's name]'s rehabilitation progressing?"),
    ("Injury Updates", "Can you confirm the expected timeline for [player's name]'s return to full fitness?"),
    ("Injury Updates", "Has [injured player]'s absence affected the team's tactics or formation?"),
    ("Injury Updates", "How have the rest of the team adjusted to the injury of [player's name]?"),
    ("Injury Updates", "What steps are being taken to prevent further injuries in the squad?"),
    ("Injury Updates", "How does the team manage without the presence of key players due to injury?"),
    ("Injury Updates", "What’s the current outlook for [player's name] following their surgery?"),
    ("Injury Updates", "How has the coaching staff adapted training methods in light of current injuries?"),
    
    ("Post-Game Analysis", "Looking back, what do you think were the pivotal moments that influenced the final score?"),
    ("Post-Game Analysis", "How do you think the team handled the pressure in the final moments of the match?"),
    ("Post-Game Analysis", "What areas of the game do you think the team needs to improve upon before the next match?"),
    ("Post-Game Analysis", "Were there any key tactical shifts made during the game that had a significant impact?"),
    ("Post-Game Analysis", "How do you rate the team’s execution of set-pieces in this match?"),
    ("Post-Game Analysis", "What aspects of the game were you most pleased with in terms of performance?"),
    ("Post-Game Analysis", "How would you describe the team’s overall mentality after today’s result?"),
    ("Post-Game Analysis", "Was there a moment in the match where you felt the game was lost or won?"),
    
    ("Team Morale", "How do you think the team is feeling after today's result?"),
    ("Team Morale", "What do you think is the key to maintaining high morale despite challenging performances?"),
    ("Team Morale", "How do you keep the players motivated during a tough season?"),
    ("Team Morale", "What role does positive reinforcement play in keeping the team’s confidence up?"),
    ("Team Morale", "How do you ensure that the team remains focused on the bigger picture, even after a tough loss?"),
    ("Team Morale", "What specific steps are being taken to enhance team bonding and chemistry?"),
    ("Team Morale", "How does the team’s collective spirit affect individual performances?"),
    ("Team Morale", "What are the primary factors that contribute to the team’s sense of unity and motivation?"),
    
    ("Upcoming Matches", "What’s your strategy for facing [team] in the next game?"),
    ("Upcoming Matches", "What do you expect from [team] in the upcoming fixture?"),
    ("Upcoming Matches", "How do you plan to handle the pressure of playing away in the next match?"),
    ("Upcoming Matches", "What adjustments will you make to your game plan to combat [opponent]’s strengths?"),
    ("Upcoming Matches", "How important is the upcoming match for the team’s aspirations this season?"),
    ("Upcoming Matches", "What do you consider the most dangerous aspect of [team]'s play in the upcoming match?"),
    ("Upcoming Matches", "What are your thoughts on the upcoming match, and what’s the priority going into it?"),
    ("Upcoming Matches", "How do you prepare mentally and tactically for facing a high-profile opponent like [team]?"),
    
    ("Off-Game Matters", "How do you manage the off-field distractions and maintain focus on the game?"),
    ("Off-Game Matters", "What hobbies or activities help you unwind after a stressful week?"),
    ("Off-Game Matters", "How do you balance the demands of football with staying connected to your community?"),
    ("Off-Game Matters", "How do you handle the pressure of transfer rumors circulating around your name?"),
    ("Off-Game Matters", "What are you doing outside of football to give back to the fans and your community?"),
    ("Off-Game Matters", "How do you keep yourself fit and focused during the off-season?"),
    ("Off-Game Matters", "Do you find it challenging to disconnect from the sport when not on the pitch?"),
    ("Off-Game Matters", "What steps do you take to maintain a healthy mental state amid a demanding schedule?"),
    
    ("Controversies", "How do you feel about the recent disciplinary actions taken against [player's name]?"),
    ("Controversies", "What’s your take on the refereeing decisions in the game and how they affected the result?"),
    ("Controversies", "Do you believe the media has unfairly portrayed the team after the recent controversy?"),
    ("Controversies", "How do you plan to move forward after the controversy surrounding the club’s decision?"),
    ("Controversies", "What is your opinion on the ongoing debate regarding the league’s controversial rule change?"),
    ("Controversies", "How do you address tensions within the team that may have arisen after the controversy?"),
    ("Controversies", "Do you think the team's public response to the incident has helped or hurt its image?"),
    ("Controversies", "How does the controversy impact the team’s mentality heading into the next few matches?")
]


# Label map
label_map = {
    "Game Strategy": 0,
    "Player Performance": 1,
    "Injury Updates": 2,
    "Post-Game Analysis": 3,
    "Team Morale": 4,
    "Upcoming Matches": 5,
    "Off-Game Matters": 6,
    "Controversies": 7
}

# Extract true labels
true_labels = [label_map[cat] for cat, _ in examples]

data = []
responses = []

# Generate responses
for cat, ques in examples:
    ans = generate_response(cat, ques)
    data.append([cat, ques, ans])
    responses.append(ans)

# Save to CSV
csv_file = "Text_generation_output.csv"
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Category", "Question", "Response"])
    writer.writerows(data)

print(f"Responses saved to {csv_file}")

#Cluster old code

import openai
import csv
from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Clustering Pipeline ----
# Step 1: Convert responses to embeddings
model = SentenceTransformer('all-mpnet-base-v2')  # Better semantic encoding
embeddings = model.encode(responses)

# Step 2: UMAP dimensionality reduction
umap_model = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)
umap_embeddings = umap_model.fit_transform(embeddings)

# Step 3: KMeans clustering
kmeans = KMeans(n_clusters=8, random_state=42)
labels = kmeans.fit_predict(umap_embeddings)

# Step 4: Plot clusters (by KMeans)
plt.figure(figsize=(8, 6))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap='Set1', s=100)
for i, txt in enumerate(range(len(responses))):
    plt.annotate(txt, (umap_embeddings[i, 0], umap_embeddings[i, 1]))
plt.title('UMAP + KMeans Clustering of Interview Responses')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.grid(True)
plt.show()

# Step 5: Evaluate clustering
ari = adjusted_rand_score(true_labels, labels)
nmi = normalized_mutual_info_score(true_labels, labels)

print(f"\nEvaluation Metrics:")
print(f"Adjusted Rand Index (ARI): {ari:.3f}")
print(f"Normalized Mutual Information (NMI): {nmi:.3f}")

# Optional: Visualize true label distribution
plt.figure(figsize=(8, 6))
palette = sns.color_palette("tab10", 8)
sns.scatterplot(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1], hue=true_labels, palette=palette, s=100)
plt.title('UMAP Projection Colored by True Topic Labels')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(title="True Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

