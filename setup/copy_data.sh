#!/bin/bash
# Script to copy data from Mac to gaming PC

# Color formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==== Data Transfer to Gaming PC ====${NC}"
echo -e "${YELLOW}This script helps transfer data from your Mac to the gaming PC${NC}"
echo ""

# Get SSH connection name
if [ $# -eq 0 ]; then
    read -p "Enter the SSH connection name (from ~/.ssh/config): " CONNECTION_NAME
else
    CONNECTION_NAME=$1
fi

# Get source directory
read -p "Enter the source data directory on Mac [~/code/quant-trader/data]: " SOURCE_DIR
SOURCE_DIR=${SOURCE_DIR:-~/code/quant-trader/data}

# Get remote directory
read -p "Enter the destination directory on Gaming PC [C:/quant-trader/data/raw]: " DEST_DIR
DEST_DIR=${DEST_DIR:-"C:/quant-trader/data/raw"}

# Windows-compatible way to create directory structure
echo -e "${YELLOW}Creating destination directory on gaming PC...${NC}"
# Convert forward slashes to backslashes and use mkdir with proper Windows syntax
WIN_DEST_DIR=$(echo "$DEST_DIR" | tr '/' '\\')
ssh $CONNECTION_NAME "if not exist \"$WIN_DEST_DIR\" mkdir \"$WIN_DEST_DIR\""

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to create destination directory. Check your SSH connection.${NC}"
    exit 1
fi

# Count number of files
FILE_COUNT=$(find "$SOURCE_DIR" -type f \( -name "*.csv" -o -name "*.parquet" \) | wc -l)
echo -e "${GREEN}Found ${FILE_COUNT} data files to transfer.${NC}"

# Confirm transfer
read -p "Do you want to transfer these files? (y/n): " CONFIRM
if [[ $CONFIRM != "y" && $CONFIRM != "Y" ]]; then
    echo -e "${YELLOW}Transfer canceled.${NC}"
    exit 0
fi

# Start transfer
echo -e "${BLUE}Starting file transfer...${NC}"
echo -e "${YELLOW}This may take a while depending on the amount of data.${NC}"

# Create a temp directory for compression
TEMP_DIR=$(mktemp -d)
ARCHIVE_NAME="quant_data_$(date +%Y%m%d_%H%M%S).tar.gz"
ARCHIVE_PATH="$TEMP_DIR/$ARCHIVE_NAME"

# Compress the files
echo -e "${YELLOW}Compressing files...${NC}"
tar -czf "$ARCHIVE_PATH" -C "$(dirname "$SOURCE_DIR")" "$(basename "$SOURCE_DIR")"

# Get file size
FILE_SIZE=$(du -h "$ARCHIVE_PATH" | cut -f1)
echo -e "${GREEN}Compressed archive size: ${FILE_SIZE}${NC}"

# Transfer the archive
# Get the parent directory path
PARENT_DIR=$(echo "$DEST_DIR" | sed 's/\/[^\/]*$//')
echo -e "${YELLOW}Transferring archive to gaming PC...${NC}"
scp "$ARCHIVE_PATH" "$CONNECTION_NAME:\"$PARENT_DIR/$ARCHIVE_NAME\""

if [ $? -ne 0 ]; then
    echo -e "${RED}Transfer failed. Check your SSH connection.${NC}"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Extract the archive on the remote machine - with Windows-compatible paths
echo -e "${YELLOW}Extracting archive on gaming PC...${NC}"
WIN_PARENT_DIR=$(echo "$PARENT_DIR" | tr '/' '\\')
WIN_ARCHIVE_NAME=$(echo "$ARCHIVE_NAME" | tr '/' '\\')

# Use PowerShell for extraction on Windows
ssh $CONNECTION_NAME "powershell -Command \"Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::ExtractToDirectory('$WIN_PARENT_DIR\\$WIN_ARCHIVE_NAME', '$WIN_PARENT_DIR')\""

# Alternative approach using 7zip if the above fails
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}PowerShell extraction failed, trying with 7zip...${NC}"
    ssh $CONNECTION_NAME "cd \"$PARENT_DIR\" && 7z x \"$ARCHIVE_NAME\" -o\"$PARENT_DIR\" && del \"$ARCHIVE_NAME\""
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Extraction failed. Please extract the archive manually.${NC}"
        echo -e "${YELLOW}Archive location: $PARENT_DIR/$ARCHIVE_NAME${NC}"
        rm -rf "$TEMP_DIR"
        exit 1
    fi
fi

# Clean up
rm -rf "$TEMP_DIR"

echo -e "${GREEN}Data transfer complete!${NC}"
echo -e "${YELLOW}Files are now available in $DEST_DIR on the gaming PC.${NC}"

# Suggest next steps
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "1. Connect to the gaming PC: ${YELLOW}~/quant-trader-scripts/open-quant-remote.sh${NC}"
echo -e "2. Process the data using the API or manually run the inventory task"
echo -e "   ${YELLOW}curl -X POST http://localhost:8000/data/inventory${NC}"