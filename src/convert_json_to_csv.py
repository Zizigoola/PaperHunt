# src/convert_json_to_csv.py
import json
import pandas as pd

def convert_json_to_csv(json_path="data/arxiv-metadata-oai-snapshot.json", 
                        csv_path="data/arxiv_data.csv", 
                        max_rows=50000):
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_rows:  # Limit for testing
                break
            paper = json.loads(line)
            data.append({
                "id": paper.get("id"),
                "title": paper.get("title", "").replace("\n", " "),
                "abstract": paper.get("abstract", "").replace("\n", " "),
                "categories": paper.get("categories"),
                "update_date": paper.get("update_date")
            })
    
    df = pd.DataFrame(data)
    df.dropna(subset=["abstract"], inplace=True)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved {len(df)} rows to {csv_path}")

if __name__ == "__main__":
    convert_json_to_csv()
