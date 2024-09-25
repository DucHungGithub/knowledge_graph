import os
import uuid

import pandas as pd 


def get_documents(input_dir: str) -> pd.DataFrame:
    rows = []
    
    for doc in os.listdir(input_dir):
        row = {
            "id": uuid.uuid4().hex,
            "title": doc,
        }
        rows.append(row)
        
    return pd.DataFrame(rows)


# print(get_documents("inputs"))

        
        