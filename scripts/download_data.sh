# Argument validation check
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <source_endpoint_id> <destination_endpoint_id> <local_destination>"
    exit 1
fi

# Check if the local destination is a relative path (does not start with / or ~)
if [[ "$3" != /* && "$3" != ~* ]]; then
    echo "ERROR: The local destination must be an absolute path. Do not use relative paths."
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
    # Exclude TREFHTMN and TREFHTMX for MA-BASELINE directories because there is only 1 member (use ARISE data instead - see below)
    if [[ $dir == *"MA-BASELINE"* ]]; then
        files=$(globus ls $source_endpoint_id:$dir | grep -E "\.TREFHT\.|\.\PRECT\.|\.QFLX\.|\.ICEFRAC\." | grep -E "\.h0\.")
    else
        files=$(globus ls $source_endpoint_id:$dir | grep -E "\.TREFHT\.|\.\PRECT\.|\.TREFHTMN\.|\.TREFHTMX\.|\.QFLX\.|\.ICEFRAC\." | grep -E "\.h0\.")
    fi

    # Loop through filtered files and initiate transfer
    for file in $files
    do
        # Check if file already exists at destination
        if globus ls $destination_endpoint_id:$local_destination | grep -q "^$file$"; then
            echo "Skipping $file, already exists at destination."
        else
            echo "Transferring $file from $dir..."
            globus transfer $source_endpoint_id:$dir/$file $destination_endpoint_id:$local_destination/$file
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
            globus transfer $source_endpoint_id:/MA-BASELINE.00$i/atm/proc/tseries/month_1/$file $destination_endpoint_id:$local_destination/$file
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
            globus transfer $source_endpoint_id:/MA-HISTORICAL.00$i/atm/proc/tseries/month_1/$file $destination_endpoint_id:$local_destination/$file
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
            globus transfer $source_endpoint_id:$dir/$file $destination_endpoint_id:$local_destination/$file
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
        globus transfer $source_endpoint_id:/ARISE-HISTORICAL/$file $destination_endpoint_id:$local_destination/$file
    fi
done

######################################################################################################################
# Download tasmin and tasmax from ARISE-SSP245 via AWS S3 bucket
######################################################################################################################
echo "Listing files in /ARISE-SSP245"
BUCKET="s3://ncar-cesm2-arise/CESM2-WACCM-SSP245"
SUB="atm/proc/tseries/month_1"

# Define the members, variables, and time ranges to download
members=(007 008 009)
vars=(TREFHTMN TREFHTMX)
time_ranges=(201501-206412 206501-206912)

# Loop through each member, variable, and time range combination
for m in "${members[@]}"
do
    for v in "${vars[@]}"
    do
        for r in "${time_ranges[@]}"
        do
            FILE="b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.${m}.cam.h0.${v}.${r}.nc"
            S3PATH="${BUCKET}/b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.${m}/${SUB}/${FILE}"
            
            # Check if file already exists at destination
            if [ -f "$local_destination/$FILE" ]; then
                echo "Skipping $FILE, already exists at destination."
            else
                echo "Downloading $FILE from AWS S3..."
                echo "local_destination: ${local_destination}"
                aws s3 cp --no-sign-request --region us-west-2 "$S3PATH" "$local_destination"
            fi
        done
    done
done
