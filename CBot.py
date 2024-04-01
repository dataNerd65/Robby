import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Using Defined dummy data (Step 1)
dummy_data = [
    {"query": "How do I pay my fees?", "intent": "fees_payment"},
    {"query": "Where can I check my exam results?", "intent": "results_check"},
    {"query": "What courses are available this semester?", "intent": "courses_available"},
    {"query": "When is the graduation ceremony?", "intent": "graduation_date"},
    {"query": "How do I access the student portal?", "intent": "portal_access"},
    {"query": "Where can I find information about hostel accommodation?", "intent": "hostel_info"},
    {"query": "How can I join a sports club?", "intent": "sports_club_join"},
    {"query": "What are the library opening hours?", "intent": "library_hours"},
    {"query": "Where can I get ICT support?", "intent": "ict_support"},
    {"query": "How do I enroll in a new course?", "intent": "course_enrollment"}
]

# Preprocess the dummy data (Step 2)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

preprocessed_data = []
for entry in dummy_data:
    query = entry['query'].lower()
    tokens = word_tokenize(query)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    preprocessed_query = ' '.join(tokens)
    preprocessed_data.append({"query": preprocessed_query, "intent": entry['intent']})

# Extract intents for classification
intents = set(entry['intent'] for entry in preprocessed_data)

# Feature Extraction (Step 3)
texts = [entry['query'] for entry in preprocessed_data]

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(texts)
tfidf_vectors = tfidf_vectorizer.transform(texts)

print("Shape of TF-IDF vectors:", tfidf_vectors.shape)

# Define the model architecture
model = Sequential([
    Dense(128, input_shape=(tfidf_vectors.shape[1],), activation='relu'),
    Dropout(0.5),
    Dense(len(intents), activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Converting intents to an array
intents_array = [entry['intent'] for entry in preprocessed_data]

# Encode intents using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(intents_array)

# Split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_vectors.toarray(), y_encoded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Define a function to predict the intent of a query
def predict_intent(query):
    # Preprocess the query
    preprocessed_query = query
    
    # Vectorize the preprocessed query
    query_vector = tfidf_vectorizer.transform([preprocessed_query]).toarray()
    
    # Predict the intent using the trained model
    predictions = model.predict(query_vector)
    
    # Decode the predictions to get the intent labels
    predicted_intent_index = np.argmax(predictions)
    predicted_intent = label_encoder.inverse_transform([predicted_intent_index])[0]
    
    return predicted_intent

# Test the model with some sample queries
sample_queries = [
    "How do I pay my fees?",
]

for query in sample_queries:
    predicted_intent = predict_intent(query)
    print(f"Query: {query} => Predicted Intent: {predicted_intent}")
