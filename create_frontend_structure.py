import os

# Define the folder structure for frontend
frontend_structure = {
    "frontend": [
        "public",
        "src/components",
        "src/pages",
        "src/services",
        "src/styles",
        "src",
    ],
    "frontend/src": [
        "App.js",
        "index.js",
    ],
    "frontend/public": [
        "index.html",
    ],
}

# Function to create the directories and files
def create_frontend_structure():
    for parent, children in frontend_structure.items():
        for child in children:
            path = os.path.join(parent, child)
            if not os.path.exists(path):
                if '.' in child:
                    # Create file
                    open(path, 'w').close()
                else:
                    # Create directory
                    os.makedirs(path)
                print(f"Created: {path}")
                
    print("Frontend folder structure created successfully!")

# Run the function to create the structure
create_frontend_structure()
