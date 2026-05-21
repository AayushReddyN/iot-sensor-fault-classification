import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile, io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, ConfusionMatrixDisplay,
                              classification_report)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def load_zip(zip_path, source_tag):
    """Extract all .txt sensor files from a zip and return a combined DataFrame."""
    frames = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if not name.endswith('.txt'):
                continue
            with z.open(name) as f:
                raw = f.read().decode('utf-8', errors='ignore')

            rows = []
            for line in raw.splitlines():
                line = line.strip()

                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    reading  = int(parts[0])
                    mote_id  = int(parts[1])
                    humidity = float(parts[2])
                    temp     = float(parts[3])
                    label    = int(parts[4])
                    rows.append([reading, mote_id, humidity, temp, label])
                except ValueError:
                    continue

            if rows:
                df_file = pd.DataFrame(rows, columns=[
                    'Reading', 'MoteID', 'Humidity', 'Temperature', 'Label'])
                df_file['Source']   = source_tag
                df_file['FileName'] = name.split('/')[-1]
                frames.append(df_file)
    return pd.concat(frames, ignore_index=True)


multihop   = load_zip('MultiHopLebelledReadings_1_.zip',  'MultiHop')
singlehop  = load_zip('SingleHopLabelledReadings_1_.zip', 'SingleHop')
df = pd.concat([multihop, singlehop], ignore_index=True)

print(f"Total records loaded: {len(df)}")
print(df.head(10))
print("\nClass distribution:\n", df['Label'].value_counts())

df.dropna(inplace=True)

FEATURES = ['Humidity', 'Temperature']
LABEL    = 'Label'

X = df[FEATURES].values
y = df[LABEL].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, feat in zip(axes, FEATURES):
    for label_val, color, name in zip([0, 1], ['steelblue', 'tomato'],
                                       ['Fault Free (0)', 'Faulty (1)']):
        ax.hist(df[df[LABEL] == label_val][feat], bins=40,
                alpha=0.6, color=color, label=name)
    ax.set_title(f'{feat} Distribution by Class')
    ax.set_xlabel(feat)
    ax.set_ylabel('Count')
    ax.legend()
plt.suptitle('Feature Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_distribution.png', dpi=150)
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(df[FEATURES + [LABEL]].corr(), annot=True, fmt='.3f',
            cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()

results = {}

def run_model(name, model):
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    y_pred    = model.predict(X_test)

    results[name] = {
        'Train Accuracy' : round(train_acc, 4),
        'Accuracy'       : round(accuracy_score(y_test, y_pred), 4),
        'Precision'      : round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        'Recall'         : round(recall_score(y_test, y_pred,    average='weighted', zero_division=0), 4),
        'F1-Score'       : round(f1_score(y_test, y_pred,        average='weighted', zero_division=0), 4),
    }

    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"{'='*45}")
    print(classification_report(y_test, y_pred, target_names=['Fault Free', 'Faulty'],
                                 zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Fault Free', 'Faulty'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix — {name}')
    plt.tight_layout()
    plt.savefig(f'cm_{name.replace(" ", "_")}.png', dpi=150)
    plt.show()



run_model("KNN", KNeighborsClassifier(n_neighbors=5))


run_model("Decision Tree", DecisionTreeClassifier(random_state=42))


run_model("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42))


run_model("SVM", SVC(kernel='rbf', random_state=42))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

nn = Sequential([
    Dense(64,  activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32,  activation='relu'),
    Dropout(0.2),
    Dense(1,   activation='sigmoid')
])

nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = nn.fit(X_train, y_train,
                 epochs=50, batch_size=32,
                 validation_split=0.1,
                 callbacks=[early_stop],
                 verbose=1)

y_pred_nn = (nn.predict(X_test) > 0.5).astype(int).flatten()
train_acc_nn = history.history['accuracy'][-1]

results['Neural Network'] = {
    'Train Accuracy' : round(train_acc_nn, 4),
    'Accuracy'       : round(accuracy_score(y_test, y_pred_nn), 4),
    'Precision'      : round(precision_score(y_test, y_pred_nn, average='weighted', zero_division=0), 4),
    'Recall'         : round(recall_score(y_test, y_pred_nn,    average='weighted', zero_division=0), 4),
    'F1-Score'       : round(f1_score(y_test, y_pred_nn,        average='weighted', zero_division=0), 4),
}

print(f"\n{'='*45}")
print("  Neural Network")
print(f"  Train Accuracy: {train_acc_nn:.4f}")
print(f"{'='*45}")
print(classification_report(y_test, y_pred_nn, target_names=['Fault Free', 'Faulty'], zero_division=0))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_nn),
                       display_labels=['Fault Free', 'Faulty']).plot(cmap='Blues')
plt.title('Confusion Matrix — Neural Network')
plt.tight_layout()
plt.savefig('cm_Neural_Network.png', dpi=150)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Neural Network Training Curve')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(); plt.tight_layout()
plt.savefig('nn_training_curve.png', dpi=150)
plt.show()

results_df = pd.DataFrame(results).T
print("\n========== Model Comparison ==========")
print(results_df.to_string())

colors = ['steelblue', 'seagreen', 'darkorange', 'tomato', 'mediumpurple']
plt.figure(figsize=(10, 5))
bars = plt.bar(results_df.index, results_df['Accuracy'], color=colors, edgecolor='black')
plt.ylim(0, 1.15)
plt.ylabel('Test Accuracy')
plt.title('Model Accuracy Comparison')
for bar, val in zip(bars, results_df['Accuracy']):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.02, str(val), ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('model_accuracy_comparison.png', dpi=150)
plt.show()

results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(
    kind='bar', figsize=(12, 5), colormap='tab10', edgecolor='black'
)
plt.title('All Metrics — Model Comparison')
plt.ylabel('Score')
plt.xticks(rotation=15)
plt.ylim(0, 1.15)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('full_metrics_comparison.png', dpi=150)
plt.show()