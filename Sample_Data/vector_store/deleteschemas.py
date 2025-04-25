import weaviate
import weaviate.classes as wvc # Or your specific import style
import os
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables (optional, if you use .env for API keys/URL)
load_dotenv(".env.local")
load_dotenv(".env")

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:9090")
# Add API Key header if your Weaviate instance requires it
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
HEADERS = {}
if WEAVIATE_API_KEY:
    HEADERS["X-Api-Key"] = WEAVIATE_API_KEY # Adjust header name if needed

# List all the class names you want to delete
# Add any other classes your application might have created
CLASS_NAMES_TO_DELETE = [
    "UserDocuments"
    # Add any other class names used by your system here
]
# --- End Configuration ---

print(f"--- Attempting to delete Weaviate schemas from {WEAVIATE_URL} ---")
print("WARNING: This will permanently delete the classes and ALL their data.")
confirm = input("Type 'DELETE' to confirm: ")

if confirm != "DELETE":
    print("Deletion cancelled.")
    exit()

try:
    # Connect to Weaviate
    client = weaviate.connect_to_local(
         host="localhost",
         port=9090,
         grpc_port=50051
        # url=WEAVIATE_URL,
        # additional_headers=HEADERS if HEADERS else None # Pass headers if defined
    )
    client.connect() # Ensure connection is active

    print("Connected to Weaviate.")

    # Loop through the class names and attempt deletion
    for class_name in CLASS_NAMES_TO_DELETE:
        print(f"Attempting to delete class: {class_name}...")
        try:
            client.collections.delete(class_name)
            print(f"Successfully deleted class: {class_name}")
        except Exception as e:
            # Handle potential errors, e.g., if the class doesn't exist
            print(f"Could not delete class '{class_name}'. Error: {e}")
            # Check if it's a "not found" error, which is okay in this context
            if "404" in str(e) or "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                 print(f"(Class '{class_name}' likely did not exist anyway)")
            else:
                 print("(There might be another issue, check Weaviate logs if needed)")


    print("\n--- Schema deletion process finished. ---")

except Exception as e:
    print(f"\nAn error occurred during the process: {e}")
    print("Please check your Weaviate connection details and ensure Weaviate is running.")

finally:
    # Close the connection if it was successfully opened
    if 'client' in locals() and client.is_connected():
        client.close()
        print("Weaviate client connection closed.")