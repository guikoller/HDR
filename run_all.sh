IMAGES_DIR="images"
OUTPUT_DIR="output"

for folder in "$IMAGES_DIR"/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")

        output_file="$OUTPUT_DIR/${folder_name}_hdr.png"
        
        python3 main.py -d "$folder" -o "$output_file"
        
        echo "HDR image created for $folder and saved as $output_file"
    fi
done