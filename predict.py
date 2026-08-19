import argparse
import pandas as pd
from model_utils import load_model

def main():
    parser = argparse.ArgumentParser(description="Crop Recommendation Predictor")
    parser.add_argument("--N", type=float, required=True)
    parser.add_argument("--P", type=float, required=True)
    parser.add_argument("--K", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--humidity", type=float, required=True)
    parser.add_argument("--ph", type=float, required=True)
    parser.add_argument("--rainfall", type=float, required=True)
    args = parser.parse_args()

    model = load_model()

    row = pd.DataFrame([{
        "N": args.N,
        "P": args.P,
        "K": args.K,
        "temperature": args.temperature,
        "humidity": args.humidity,
        "ph": args.ph,
        "rainfall": args.rainfall
    }])

    prediction = model.predict(row)[0]
    probs = model.predict_proba(row)[0]
    classes = model.classes_

    top3 = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]

    print("\nRecommended Crop:", prediction)
    print("Top 3 suggestions:")
    for crop, prob in top3:
        print(f"  {crop}: {prob:.2%}")

if __name__ == "__main__":
    main()
