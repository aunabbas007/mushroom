# ============================================================
#  model.py — Load Trained Model & Predict New Mushrooms
#
#  ⚠️  Run train.py FIRST to generate mushroom_rf_model.pkl
#
#  This file shows you how to:
#    1. Load the saved .pkl model
#    2. Prepare a new raw mushroom sample
#    3. Encode it the same way training data was encoded
#    4. Predict: Edible or Poisonous
# ============================================================

import pickle
import pandas as pd


# ── STEP 1: Load the Saved Model ─────────────────────────────────────────────
pkl_path = "mushroom_rf_model.pkl"

with open(pkl_path, "rb") as f:            # "rb" = read binary
    model_data = pickle.load(f)

rf_model  = model_data["model"]            # The trained RandomForestClassifier
encoders  = model_data["encoders"]         # LabelEncoders for every column
features  = model_data["features"]         # List of feature column names

print("✅ Model loaded successfully!")
print(f"   Trees in forest : {rf_model.n_estimators}")
print(f"   Features used   : {len(features)}")
print(f"   Feature list    : {features}\n")


# ── STEP 2: Define a New Raw Mushroom Sample ──────────────────────────────────
# These are RAW text values — exactly as they appear in the original CSV
# You can change these values to test different mushrooms
#
# Column reference: https://www.kaggle.com/datasets/uciml/mushroom-classification
#
# Full feature order (22 features, same as training):
#   cap-shape, cap-surface, cap-color, bruises, odor,
#   gill-attachment, gill-spacing, gill-size, gill-color,
#   stalk-shape, stalk-root, stalk-surface-above-ring,
#   stalk-surface-below-ring, stalk-color-above-ring,
#   stalk-color-below-ring, veil-type, veil-color,
#   ring-number, ring-type, spore-print-color,
#   population, habitat

new_mushroom_raw = {
    "cap-shape"               : "x",   # x=convex
    "cap-surface"             : "s",   # s=smooth
    "cap-color"               : "n",   # n=brown
    "bruises"                 : "t",   # t=yes
    "odor"                    : "p",   # p=pungent  ← strong poison signal!
    "gill-attachment"         : "f",   # f=free
    "gill-spacing"            : "c",   # c=close
    "gill-size"               : "n",   # n=narrow
    "gill-color"              : "k",   # k=black
    "stalk-shape"             : "e",   # e=enlarging
    "stalk-root"              : "e",   # e=equal
    "stalk-surface-above-ring": "s",   # s=smooth
    "stalk-surface-below-ring": "s",
    "stalk-color-above-ring"  : "w",   # w=white
    "stalk-color-below-ring"  : "w",
    "veil-type"               : "p",   # p=partial
    "veil-color"              : "w",
    "ring-number"             : "o",   # o=one
    "ring-type"               : "p",   # p=pendant
    "spore-print-color"       : "k",   # k=black
    "population"              : "s",   # s=scattered
    "habitat"                 : "u",   # u=urban
}


# ── STEP 3: Encode the Raw Input ─────────────────────────────────────────────
# We must encode the new sample using the SAME LabelEncoders from training
# If we used different encoding, predictions would be completely wrong

encoded_values = []

for col in features:                                   # iterate in training order
    raw_val = new_mushroom_raw[col]                    # get raw text value
    le      = encoders[col]                            # get the saved encoder

    # Check if this value was seen during training
    if raw_val in le.classes_:
        encoded = le.transform([raw_val])[0]           # encode it
    else:
        # Unseen value — use the most common class (index 0) as fallback
        print(f"⚠️  Unknown value '{raw_val}' for '{col}' → using default")
        encoded = 0

    encoded_values.append(encoded)

# Convert to DataFrame with correct column names
input_df = pd.DataFrame([encoded_values], columns=features)
print("✅ Input encoded successfully!")
print(input_df)


# ── STEP 4: Predict ──────────────────────────────────────────────────────────
prediction      = rf_model.predict(input_df)[0]              # 0 or 1
prediction_proba= rf_model.predict_proba(input_df)[0]        # [prob_edible, prob_poisonous]

label_map = {0: "🟢 EDIBLE", 1: "🔴 POISONOUS"}
result    = label_map[prediction]

print("\n" + "=" * 40)
print(f"  Prediction  : {result}")
print(f"  Confidence  : {max(prediction_proba) * 100:.2f}%")
print(f"  Edible prob : {prediction_proba[0] * 100:.2f}%")
print(f"  Poison prob : {prediction_proba[1] * 100:.2f}%")
print("=" * 40)

if prediction == 1:
    print("\n⚠️  DO NOT EAT THIS MUSHROOM!")
else:
    print("\n✅ This mushroom appears safe to eat.")


# ── STEP 5: Batch Predict Multiple Mushrooms ──────────────────────────────────
# You can also pass a list of mushrooms and predict all at once

print("\n\n── Batch Prediction Example ──────────────────────────")

# Two samples: one edible-looking, one poisonous-looking
batch_raw = [
    # Sample A — likely edible (odor=n → none, good signs)
    {"cap-shape":"x","cap-surface":"s","cap-color":"y","bruises":"t",
     "odor":"n","gill-attachment":"f","gill-spacing":"c","gill-size":"b",
     "gill-color":"n","stalk-shape":"e","stalk-root":"c",
     "stalk-surface-above-ring":"s","stalk-surface-below-ring":"s",
     "stalk-color-above-ring":"w","stalk-color-below-ring":"w",
     "veil-type":"p","veil-color":"w","ring-number":"o","ring-type":"p",
     "spore-print-color":"n","population":"n","habitat":"g"},

    # Sample B — likely poisonous (odor=f → foul)
    {"cap-shape":"k","cap-surface":"f","cap-color":"e","bruises":"f",
     "odor":"f","gill-attachment":"f","gill-spacing":"c","gill-size":"n",
     "gill-color":"b","stalk-shape":"e","stalk-root":"b",
     "stalk-surface-above-ring":"k","stalk-surface-below-ring":"k",
     "stalk-color-above-ring":"n","stalk-color-below-ring":"n",
     "veil-type":"p","veil-color":"w","ring-number":"o","ring-type":"l",
     "spore-print-color":"h","population":"s","habitat":"g"},
]

batch_encoded = []
for sample in batch_raw:
    row = []
    for col in features:
        raw_val = sample[col]
        le      = encoders[col]
        encoded = le.transform([raw_val])[0] if raw_val in le.classes_ else 0
        row.append(encoded)
    batch_encoded.append(row)

batch_df      = pd.DataFrame(batch_encoded, columns=features)
batch_preds   = rf_model.predict(batch_df)
batch_probas  = rf_model.predict_proba(batch_df)

for i, (pred, proba) in enumerate(zip(batch_preds, batch_probas)):
    print(f"\nSample {i+1}: {label_map[pred]}  "
          f"(confidence: {max(proba)*100:.1f}%)")