# Configuration
SCRIPT="./scripts/download_gauss.sh"
SOURCE_ENDPOINT="test_source_endpoint"
DEST_ENDPOINT="test_dest_endpoint"
TEST_DIR="tmp/test"

# Test mutually exclusive flags
test_mutually_exclusive_flags() {
    echo "Testing mutually exclusive flags..."
    
    output=$($SCRIPT $SOURCE_ENDPOINT $DEST_ENDPOINT $TEST_DIR --tas --pr 2>&1)
    if ! echo "$output" | grep -q "cannot select both.*tas.*pr"; then
        echo "✗ Failed to catch mutually exclusive --tas and --pr"
        return 1
    fi
    
    output=$($SCRIPT $SOURCE_ENDPOINT $DEST_ENDPOINT $TEST_DIR --daily --monthly 2>&1)
    if ! echo "$output" | grep -q "cannot select both.*daily.*monthly"; then
        echo "✗ Failed to catch mutually exclusive --daily and --monthly"
        return 1
    fi
    
    echo "✓ Mutually exclusive flags tests passed"
}

test_file_patterns() {
    echo "Testing file pattern generation..."
    
    # Test TAS + MONTHLY 
    output=$($SCRIPT $SOURCE_ENDPOINT $DEST_ENDPOINT $TEST_DIR --tas --monthly 2>&1)
    if echo "$output" | grep -q "TREFHT" && echo "$output" | grep -q "month_1" && ! echo "$output" | grep -q "PRECT\|PRECL\|PRECC\|day_1"; then
        echo "✓ Test passed for --tas --monthly pattern."
    else
        echo "✗ Test failed for --tas --monthly pattern."
        return 1
    fi
    
    # Test PR + MONTHLY
    output=$($SCRIPT $SOURCE_ENDPOINT $DEST_ENDPOINT $TEST_DIR --pr --monthly 2>&1)
    if echo "$output" | grep -q "PRECT" \
       && echo "$output" | grep -q "PRECL\|PRECC" \
       && echo "$output$" | grep -q "month_1" \
       && ! echo "$output" | grep -q "TREFHT\|day_1"; then
        echo "✓ Test passed for --pr --monthly pattern."
    else
        echo "✗ Test failed for --pr --monthly pattern."
        return 1
    fi
        
    # Test TAS + DAILY
    output=$($SCRIPT $SOURCE_ENDPOINT $DEST_ENDPOINT $TEST_DIR --tas --daily 2>&1)
    if echo "$output" | grep -q "TREFHT" \
       && echo "$output" | grep -q "day_1" \
       && ! echo "$output" | grep -q "PRECT\|ARISE\|month_1"; then
        echo "✓ Test passed for --tas --daily pattern."
    else
        echo "✗ Test failed for --tas --daily pattern."
        return 1
    fi

    # Test PR + DAILY
    output=$($SCRIPT $SOURCE_ENDPOINT $DEST_ENDPOINT $TEST_DIR --pr --daily 2>&1)
    if echo "$output" | grep -q "PRECT" \
       && echo "$output" | grep -q "day_1" \
       && echo "$output" | grep -q "ARISE" \
       && ! echo "$output" | grep -q "TREFHT\|month_1"; then
        echo "✓ Test passed for --pr --daily pattern."
    else
        echo "✗ Test failed for --pr --daily pattern."
        return 1
    fi

    # Test TAS only
    output=$($SCRIPT $SOURCE_ENDPOINT $DEST_ENDPOINT $TEST_DIR --tas 2>&1)
    if echo "$output" | grep -q "TREFHT" \
       && echo "$output" | grep -q "day_1" \
       && echo "$output" | grep -q "month_1" \
       && ! echo "$output" | grep -q "PRECT\|ARISE"; then
        echo "✓ Test passed for --tas pattern."
    else
        echo "✗ Test failed for --tas pattern."
        return 1
    fi

    # Test PR only
    output=$($SCRIPT $SOURCE_ENDPOINT $DEST_ENDPOINT $TEST_DIR --pr 2>&1)
    if echo "$output" | grep -q "PRECT" \
       && echo "$output" | grep -q "day_1" \
       && echo "$output" | grep -q "month_1" \
       && echo "$output" | grep -q "ARISE" \
       && echo "$output" | grep -q "PRECL\|PRECC" \
       && ! echo "$output" | grep -q "TREFHT"; then
        echo "✓ Test passed for --pr pattern."
    else
        echo "✗ Test failed for --pr pattern."
        return 1
    fi

    # Test MONTHLY only
    output=$($SCRIPT $SOURCE_ENDPOINT $DEST_ENDPOINT $TEST_DIR --monthly 2>&1)
    if echo "$output" | grep -q "PRECT" \
       && echo "$output" | grep -q "PRECL\|PRECC" \
       && echo "$output" | grep -q "TREFHT" \
       && echo "$output$" | grep -q "month_1" \
       && ! echo "$output" | grep -q "day_1\|ARISE"; then
        echo "✓ Test passed for --monthly pattern."
    else
        echo "✗ Test failed for --monthly pattern."
        return 1
    fi

    # Test DAILY only
    output=$($SCRIPT $SOURCE_ENDPOINT $DEST_ENDPOINT $TEST_DIR --daily 2>&1)
    if echo "$output" | grep -q "PRECT" \
       && echo "$output" | grep -q "TREFHT" \
       && echo "$output$" | grep -q "ARISE" \
       && echo "$output$" | grep -q "day_1" \
       && ! echo "$output" | grep -q "PRECL\|PRECC\|month_1"; then
        echo "✓ Test passed for --daily pattern."
    else
        echo "✗ Test failed for --daily pattern."
        return 1
    fi

    # Test with no patterns
    output=$($SCRIPT $SOURCE_ENDPOINT $DEST_ENDPOINT $TEST_DIR 2>&1)
    if echo "$output" | grep -q "PRECT" \
       && echo "$output" | grep -q "TREFHT" \
       && echo "$output" | grep -q "PRECL\|PRECC" \
       && echo "$output$" | grep -q "ARISE" \
       && echo "$output$" | grep -q "day_1" \
       && echo "$output$" | grep -q "month_1"; then
        echo "✓ Test passed for no pattern (original function)."
    else
        echo "✗ Test failed for no pattern (original function)."
        return 1
    fi
}


# Main test runner
main() {

    # Run all tests
    tests=(
        test_mutually_exclusive_flags
        test_file_patterns
    )
    
    failed=0
    for test in "${tests[@]}"; do
        if ! $test; then
            failed=$((failed + 1))
        fi
    done

    # Report results
    total=${#tests[@]}
    passed=$((total - failed))
    echo "Test Results: $passed/$total tests passed"
    
    return $failed
}

# Run the test suite
main "$@"