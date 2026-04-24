import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

from training_data import training_data

MODEL_PATH = "model.pkl"

CATEGORY_ICONS = {
    "Хоол": "🍽️",
    "Дэлгүүр": "🛒",
    "Тээвэр": "🚕",
    "Коммунал": "💡",
    "Эрүүл мэнд": "🏥",
    "Зугаа цэнгэл": "🎮",
    "Технологи": "📱",
    "Шилжүүлэг": "💸",
    "Ном": "📚",
    "Бусад": "❓",
}


class SMSClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 4),
                max_features=5000,
                sublinear_tf=True,
            )),
            ('clf', MultinomialNB(alpha=0.5)),
        ])
        self.is_trained = False

    def train(self, data=None):
        if data is None:
            data = training_data

        texts = [d[0] for d in data]
        labels = [d[1] for d in data]

        self.pipeline.fit(texts, labels)
        self.is_trained = True

        # Cross-validation нарийвчлал
        scores = cross_val_score(self.pipeline, texts, labels, cv=min(5, len(set(labels))), scoring='accuracy')
        print(f"✅ Model trained! Cross-val accuracy: {np.mean(scores):.2%} (+/- {np.std(scores):.2%})")

        self.save()
        return np.mean(scores)

    def predict(self, sms_text: str) -> dict:
        if not self.is_trained:
            self.load()

        proba = self.pipeline.predict_proba([sms_text])[0]
        classes = self.pipeline.classes_
        predicted_class = classes[np.argmax(proba)]
        confidence = float(np.max(proba))

        # Бүх категорийн магадлал
        all_probabilities = {
            cls: round(float(prob), 4)
            for cls, prob in sorted(zip(classes, proba), key=lambda x: -x[1])
        }

        return {
            "category": predicted_class,
            "icon": CATEGORY_ICONS.get(predicted_class, "❓"),
            "confidence": round(confidence, 4),
            "all_probabilities": all_probabilities,
        }

    def save(self, path=MODEL_PATH):
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"💾 Model saved to {path}")

    def load(self, path=MODEL_PATH):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.pipeline = pickle.load(f)
            self.is_trained = True
            print(f"📂 Model loaded from {path}")
        else:
            print("⚠️  No saved model found, training now...")
            self.train()

    def add_training_sample(self, sms_text: str, correct_category: str):
        """Хэрэглэгч залруулсан категорийг дахин сургах"""
        training_data.append((sms_text, correct_category))
        self.train(training_data)
        print(f"🔄 Model retrained with new sample: '{correct_category}'")


# Singleton instance
classifier = SMSClassifier()

if __name__ == "__main__":
    clf = SMSClassifier()
    accuracy = clf.train()
    print(f"\nTest predictions:")
    tests = [
        "KHAN BANK: ***5678 данснаас 35,000₮ зарцуулагдлаа. Худалдаа: BURGER KING",
        "KHAN BANK: ***5678 данснаас 89,000₮ зарцуулагдлаа. Худалдаа: NOMIN MARKET",
        "KHAN BANK: ***5678 данснаас 45,000₮ зарцуулагдлаа. Худалдаа: ЦАХИЛГААН ХАНГАМЖ",
    ]
    for t in tests:
        result = clf.predict(t)
        print(f"  {result['icon']} {result['category']} ({result['confidence']:.0%}) — {t[:60]}...")