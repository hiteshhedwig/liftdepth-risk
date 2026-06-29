#!/usr/bin/env bash

set -e

# ------------------------------------------------------------
# Multi-drive KITTI Raw downloader
#
# Add/remove drives in the DRIVES list below.
#
# Format:
#   "DATE DRIVE_ID"
#
# Example:
#   "2011_09_26 0005"
# ------------------------------------------------------------

DATA_ROOT="data/kitti/raw"
BASE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"

# Start small. Do not download everything at once.
# These are good candidates to test for more traffic/objects.
DRIVES=(
  "2011_09_26 0005"
  "2011_09_26 0011"
  "2011_09_26 0020"
)

mkdir -p "${DATA_ROOT}"

echo "KITTI Raw multi-drive downloader"
echo "Data root: ${DATA_ROOT}"
echo ""

echo "Free disk space:"
df -h .
echo ""

download_file_if_missing() {
    local url="$1"
    local out_file="$2"

    if [ -f "${out_file}" ]; then
        echo "Already exists: ${out_file}"
        return 0
    fi

    echo "Checking URL:"
    echo "  ${url}"

    if wget --spider -q "${url}"; then
        echo "Downloading:"
        echo "  ${out_file}"
        wget -c "${url}" -O "${out_file}"
    else
        echo "WARNING: URL not found or unavailable:"
        echo "  ${url}"
        echo "Skipping."
        return 1
    fi
}

download_calib() {
    local date="$1"
    local calib_zip="${date}_calib.zip"
    local calib_url="${BASE_URL}/${calib_zip}"

    cd "${DATA_ROOT}"

    download_file_if_missing "${calib_url}" "${calib_zip}"

    if [ -f "${calib_zip}" ]; then
        echo "Unzipping calibration: ${calib_zip}"
        unzip -n "${calib_zip}"
    fi

    cd - > /dev/null
}

download_drive() {
    local date="$1"
    local drive="$2"

    local drive_name="${date}_drive_${drive}"
    local sync_zip="${drive_name}_sync.zip"
    local sync_url="${BASE_URL}/${drive_name}/${sync_zip}"

    cd "${DATA_ROOT}"

    echo ""
    echo "------------------------------------------------------------"
    echo "Drive: ${drive_name}"
    echo "------------------------------------------------------------"

    download_file_if_missing "${sync_url}" "${sync_zip}"

    if [ -f "${sync_zip}" ]; then
        echo "Unzipping drive: ${sync_zip}"
        unzip -n "${sync_zip}"
    fi

    cd - > /dev/null
}

# Download calibration once per unique date.
DATES_DONE=""

for item in "${DRIVES[@]}"; do
    read -r DATE DRIVE <<< "${item}"

    if [[ " ${DATES_DONE} " != *" ${DATE} "* ]]; then
        echo ""
        echo "Downloading calibration for date: ${DATE}"
        download_calib "${DATE}"
        DATES_DONE="${DATES_DONE} ${DATE}"
    fi
done

# Download each drive.
for item in "${DRIVES[@]}"; do
    read -r DATE DRIVE <<< "${item}"
    download_drive "${DATE}" "${DRIVE}"
done

echo ""
echo "Done downloading selected KITTI drives."
echo ""
echo "Downloaded folders:"
find "${DATA_ROOT}" -maxdepth 2 -type d | sort
