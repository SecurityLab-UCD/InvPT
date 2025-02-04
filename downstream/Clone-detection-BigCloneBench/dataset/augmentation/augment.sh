#!/bin/bash
# augment.sh [-l LIBPATH] TEST_ID [TEST_ID...]
# Runs SPAT

while getopts "l:" opt; do
    case $opt in
    l)
        lib_path="${OPTARG}"
        shift 2
        ;;
	\?)
	    echo "Invalid option: -$OPTARG" >&2
	    ;;
    esac
done

lib_path=${lib_path:-"/usr/lib/jvm/java-18-openjdk-amd64/lib"}

# Assume default directory structure
# Benchmarks
# └── "$benchmark_name"
#     ├── Original
#     └── transformed
original_path="artifacts/original"
augmented_path="artifacts/augmented"

# 0, 1, 2, 3, 6, 7 for PIA
# 0. LocalVarRenaming:
# 1. For2While
# 2. While2For
# 3. ReverseIfElse
# 4. SingleIF2ConditionalExp
# 5. ConditionalExp2SingleIF
# 6. PP2AddAssignment
# 7. AddAssignemnt2EqualAssignment
# 8. InfixExpressionDividing
# 9. IfDividing
# 10. StatementsOrderRearrangement
# 11. LoopIfContinue2Else
# 12. VarDeclarationMerging
# 13. VarDeclarationDividing
# 14. SwitchEqualSides
# 15. SwitchStringEqual
# 16. PrePostFixExpressionDividing
# 17. Case2IfElse
for test_id in "$@"; do
  echo "Running test ID: $test_id"
  java -jar SPAT-linux.jar \
      "$test_id" \
      "$original_path" \
      "$augmented_path/_$test_id" \
      "$lib_path" \
      >/dev/null
  echo "Number of files successfully augmented:"
  ls "$augmented_path/_$test_id" | wc -l
done
