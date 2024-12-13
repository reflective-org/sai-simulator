# Argument validation check
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <source_endpoint_id> <destination_endpoint_id> <local_destination>"
    exit 1
fi

# Obtain the endpoints from the CLI
source_endpoint_id=$1
destination_endpoint_id=$2
local_destination=$3

# Create the local destination directory if it does not exist
mkdir -p $local_destination

# Define the base directories to download from
base_dirs=()
for i in {1..3}
do
    base_dirs+=("/MA-BASELINE.00$i/atm/proc/tseries/month_1")
    base_dirs+=("/MA-HISTORICAL.00$i/atm/proc/tseries/month_1")
    base_dirs+=("/SSP245-MA-GAUSS-DEFAULT.00$i/atm/proc/tseries/month_1")
    base_dirs+=("/SSP245-MA-GAUSS-LOWER-0.5.00$i/atm/proc/tseries/month_1")
    base_dirs+=("/SSP245-MA-GAUSS-LOWER-1.0.00$i/atm/proc/tseries/month_1")
done

# Loop through each base directory and download the files
for dir in "${base_dirs[@]}"
do
    echo "Listing files in $dir..."
    # List files in the directory and filter for specific patterns
    files=$(globus ls $source_endpoint_id:$dir | grep -E "\.TREFHT\.|\.\PRECT\.|\.TREFHTMN\.|\.TREFHTMX\.|\.QFLX\." | grep -E "\.h0\.")

    # Loop through filtered files and initiate transfer
    for file in $files
    do
        # Check if file already exists at destination
        if globus ls $destination_endpoint_id:$local_destination | grep -q "^$file$"; then
            echo "Skipping $file, already exists at destination."
        else
            echo "Transferring $file from $dir..."
            globus transfer $source_endpoint_id:$dir/$file $destination_endpoint_id:$local_destination$file
        fi
    done
done

# For MA-BASELINE.002 and MA-BASELINE.003, have to download PRECC and PRECL instead of PRECT
for i in {2..3}
do
    echo "Listing files in /MA-BASELINE.00$i/atm/proc/tseries/month_1..."
    files=$(globus ls $source_endpoint_id:/MA-BASELINE.00$i/atm/proc/tseries/month_1 | grep -E "\.PRECL\.|\.\PRECC\." | grep -E "\.h0\.")
    for file in $files
    do
        if globus ls $destination_endpoint_id:$local_destination | grep -q "^$file$"; then
            echo "Skipping $file, already exists at destination."
        else
            echo "Transferring $file from /MA-BASELINE.00$i/atm/proc/tseries/month_1..."
            globus transfer $source_endpoint_id:/MA-BASELINE.00$i/atm/proc/tseries/month_1/$file $destination_endpoint_id:$local_destination$file
        fi
    done
done

# For MA-HISTORICAL, have to download PRECC and PRECL insrtead of PRECT
for i in {1..3}
do
    echo "Listing files in /MA-HISTORICAL.00$i/atm/proc/tseries/month_1..."
    files=$(globus ls $source_endpoint_id:/MA-HISTORICAL.00$i/atm/proc/tseries/month_1 | grep -E "\.PRECL\.|\.\PRECC\." | grep -E "\.h0\.")
    for file in $files
    do
        if globus ls $destination_endpoint_id:$local_destination | grep -q "^$file$"; then
            echo "Skipping $file, already exists at destination."
        else
            echo "Transferring $file from /MA-HISTORICAL.00$i/atm/proc/tseries/month_1..."
            globus transfer $source_endpoint_id:/MA-HISTORICAL.00$i/atm/proc/tseries/month_1/$file $destination_endpoint_id:$local_destination$file
        fi
    done
done

# Download daily TREFHT and PRECT files for base_dirs
for dir in "${base_dirs[@]}"
do
    # Replace month with day
    dir=$(echo $dir | sed 's/month_1/day_1/g')
    echo "Listing files in $dir..."
    files=$(globus ls $source_endpoint_id:$dir | grep -E "\.TREFHT\.|\.PRECT" | grep -E "\.h1\.")
    for file in $files
    do
        if globus ls $destination_endpoint_id:$local_destination | grep -q "^$file$"; then
            echo "Skipping $file, already exists at destination."
        else
            echo "Transferring $file from $dir..."
            globus transfer $source_endpoint_id:$dir/$file $destination_endpoint_id:$local_destination$file
        fi
    done
done

# Download daily PRECT from ARISE-HISTORICAL
echo "Listing files in /ARISE-HISTORICAL"
files=$(globus ls $source_endpoint_id:/ARISE-HISTORICAL | grep -E "\.PRECT\." | grep -E "\.h1\.")
for file in $files
do
    if globus ls $destination_endpoint_id:$local_destination | grep -q "^$file$"; then
        echo "Skipping $file, already exists at destination."
    else
        echo "Transferring $file from /ARISE-HISTORICAL..."
        globus transfer $source_endpoint_id:/ARISE-HISTORICAL/$file $destination_endpoint_id:$local_destination$file
    fi
done
