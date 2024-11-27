# Argument validation check
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <local_destination>"
    exit 1
fi

# Obtain the local destination from the CLI
local_destination=$1

# Make the local destination directory if it does not exist
mkdir -p $local_destination

# Download the FaIR data using wget and place it in the destination directory
wget -r -np -nH -nd -P $local_destination TODO
