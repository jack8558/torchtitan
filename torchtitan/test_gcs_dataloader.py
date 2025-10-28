import torch
from dataflux_pytorch import dataflux_iterable_dataset
import io
import json

# --- Configuration ---
# 1. Install the connector:
# pip install gcs-connector-for-pytorch

# 2. Set your GCS details
YOUR_PROJECT_ID = "tpu-pytorch"  # Your Google Cloud project ID
YOUR_BUCKET_NAME = "torchprime"  # The name of your GCS bucket
YOUR_DATA_PREFIX = "jackoh-exp/gcs-connector/c4_test" # The "folder" in your bucket

# --- 1. Initialize the Dataset ---
print(f"Connecting to GCS: gs://{YOUR_BUCKET_NAME}/{YOUR_DATA_PREFIX}")

try:
    # This dataset streams a list of objects and loads them lazily
    iterable_dataset = dataflux_iterable_dataset.DataFluxIterableDataset(
        project_name=YOUR_PROJECT_ID,
        bucket_name=YOUR_BUCKET_NAME,
        # Config allows you to specify a prefix (sub-folder)
        config=dataflux_iterable_dataset.Config(
            prefix=YOUR_DATA_PREFIX,
            disable_compose=True
        )

    )

    # --- 2. Wrap with PyTorch DataLoader (Optional, but good practice) ---
    dataloader = torch.utils.data.DataLoader(
        iterable_dataset,
        batch_size=1,
        num_workers=0 
    )

    # --- 3. Confirm it Worked ---
    print("\n--- Successfully loaded dataset. Reading first 5 items... ---")
    
    data_iterator = iter(dataloader)
    
    for i in range(5):
        try:
            batch = next(data_iterator)
            
            # Since batch_size=1, batch[0] is our single item
            item_bytes = batch[0] 
            
            print(f"\n--- Item {i+1}: Loaded {len(item_bytes)} bytes. ---")

            # --- MODIFICATION START ---
            # Decode, parse the JSON lines, and extract the "text" field
            try:
                decoded_string = item_bytes.decode('utf-8')

                text_parts = []
                # The user-provided logic to process the string
                for line in decoded_string.strip().split("\n"):
                    if line:
                        data_dict = json.loads(line)
                        text_parts.append(data_dict["text"])
                
                final_text = "\n".join(text_parts)

                print("--- Extracted and combined 'text' field for dataset: ---")
                print(final_text)

            except UnicodeDecodeError:
                # Fallback if it's not text (e.g., a binary file)
                print("Could not decode item as UTF-8 text. Printing raw bytes instead:")
                print(f"{item_bytes[:150]}...")
            # --- MODIFICATION END ---
            
        except StopIteration:
            print(f"--- Reached end of dataset after {i} items. ---")
            break
        except Exception as e:
            print(f"Error while loading item {i+1}: {e}")
            break
            
    if i == 4:
      print("\n--- Successfully read 5 items. ---")

except Exception as e:
    print(f"\n--- An error occurred ---")
    print(f"Error: {e}")
    print("\nPlease check:")
    print(f"1. You have run 'gcloud auth application-default login'")
    print(f"2. The GCS bucket '{YOUR_BUCKET_NAME}' and project '{YOUR_PROJECT_ID}' are correct.")
    print(f"3. The prefix '{YOUR_DATA_PREFIX}' contains objects.")
    print(f"4. The user/service account has 'Storage Object Viewer' permissions.")