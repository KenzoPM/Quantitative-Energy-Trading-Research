import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import time

# --- CONFIGURATION ---
# Set this to the location of the downloaded Marktstammdatenregister XML ZIP
ZIP_FILE_PATH = "data/mastr_export.zip"

def get_tag_value(element, keyword):
    for child in element:
        if keyword.lower() in child.tag.lower():
            return child.text
    return None

def process_mastr_data(zip_path, asset_type):
    print(f"\nLaunching Extraction for: {asset_type}")
    records = []

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            target_files = [f for f in z.namelist() if asset_type.lower() in f.lower() and f.endswith('.xml')]
            
            for filename in target_files:
                with z.open(filename) as xml_file:
                    context = ET.iterparse(xml_file, events=('end',))
                    for event, elem in context:
                        tag_name = elem.tag.split('}')[-1]
                        
                        if tag_name.startswith('Einheit') and asset_type.lower() in tag_name.lower():
                            cap_str = get_tag_value(elem, 'nettonennleistung') or get_tag_value(elem, 'bruttoleistung')
                            plz = get_tag_value(elem, 'postleitzahl') or get_tag_value(elem, 'plz')
                            
                            if cap_str:
                                try:
                                    records.append({
                                        'ZIP_Code': plz,
                                        'MW': round(float(cap_str.replace(',', '.')) / 1000, 4)
                                    })
                                except ValueError:
                                    pass
                            elem.clear()
    except FileNotFoundError:
        print("ZIP file not found. Ensure the MaStR data is downloaded and the path is correct.")
        return

    if records:
        df = pd.DataFrame(records)
        df.to_parquet(f"MaStR_V2_{asset_type}_Cleaned.parquet")
        print(f"Extraction complete. {len(df)} assets processed.")

if __name__ == "__main__":
    # Uncomment to run extraction
    # process_mastr_data(ZIP_FILE_PATH, "Stromspeicher")
    # process_mastr_data(ZIP_FILE_PATH, "Solar")
    pass
