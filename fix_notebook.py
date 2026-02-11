import json

# Load the notebook
with open('Wind_mill_model.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Create a new cell to drop Date/Time column
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Drop the Date/Time column as it's not useful for correlation analysis or prediction\n",
        "if 'Date/Time' in df.columns:\n",
        "    df = df.drop(columns=['Date/Time'])\n",
        "    print(\"✓ Dropped 'Date/Time' column\")\n",
        "    print(f\"Updated dataset shape: {df.shape}\")\n",
        "    print(f\"Remaining columns: {df.columns.tolist()}\")"
    ]
}

# Insert the new cell at position 7 (after cell 6, before the correlation cell)
notebook['cells'].insert(7, new_cell)

# Save the updated notebook
with open('Wind_mill_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("✓ Notebook updated successfully!")
print("✓ Added cell to drop Date/Time column before correlation analysis")
print("\nPlease restart the kernel and run all cells again.")
