# Java Transform with SPAT

```bash
python transform.py -h
```

This directory provides utility functions to call SPAT on Java code.

### Run on CodeSearchNet

```bash

# apply single transformation rule
python transform.py -t LocalVarRenaming -i csn_java.jsonl -o csv_java_LocVarRenaming.jsonl

# do it all
python augment_pretrain.py -i csn_java.jsonl -o csv_java_aug.jsonl
```

---

# `augment_test.py`

## Quick start

```
python3 augment_test.py ./datasets/example.jsonl ./datasets/aug_example.jsonl
```

## Input

The input is a jsonl Java dataset preprocessed into POJ104 format with the
following columns:

```
   label  index                                               code
0      2    600  import java.util.*;\n\nclass Main {\n  public ...
1      2    601  import java.util.*;\n\npublic class Main {\n  ...
2      2    602  import java.io.BufferedReader;\nimport java.io...
```

Optionally, `augment.py` also takes in the following SPAT arguments

- `--spat_jar`: the path to the SPAT jar file
- `--spat_lib`: the library used by SPAT to augment the dataset
- `--rules`: the set of augmentation rules to perform

The rules are the following. By default, it is set to 0,1,2,3,6,7. 0. LocalVarRenaming\*

1. For2While\*
2. While2For\*
3. ReverseIfElse\*
4. SingleIF2ConditionalExp
5. ConditionalExp2SingleIF
6. PP2AddAssignment\*
7. AddAssignemnt2EqualAssignment\*
8. InfixExpressionDividing
9. IfDividing
10. StatementsOrderRearrangement
11. LoopIfContinue2Else
12. VarDeclarationMerging
13. VarDeclarationDividing
14. SwitchEqualSides
15. SwitchStringEqual
16. PrePostFixExpressionDividing
17. Case2IfElse

For more information on these arguments, visit
[](https://github.com/SecurityLab-UCD/SPAT).

## Output

The augmented dataset contains all of the original columns

- `index`: Augmented entries start after the highest existing index
- `label`: Augmented entries have the same label as their original
- `code`: Augmented entries have the augmented code in this column

In addition, augmented entries will have the following properties

- `aug_type_<iter>`: the readable string label for the augmentation rule (such as
  "LocalVarRenaming") or "None" if the entry is an original
- `aug_from_<iter>`: the `index` value of the original
- `aug_success_<iter>`: True if the augmentation is successful, False otherwise.
  - If False, `aug_entry.code == original_entry.code`

## Chained augmentations

`augment.py` supports augmentation over multiple iterations. Each iteration is
marked by the `<iter>` key on the three augmentation columns shown above.

Iteration 0 refers to the most recent augmentation. When a dataset that has
already been augmented is provided as input, all existing augmentation columns
have their iteration incremented and the new augmentation becomes
`aug_<statistic>_0`.

Use the `--accumulate` flag as a shortcut for chaining augmentations.

# `augment_pretrain.py`

`augment_pretrain.py` has nearly identical flags with `augment_test.py`.
However, it takes in a different dataset format and supports different features.

The format of the input dataset should be CodeSearchNet, with at least the
`code: str` column. These entries are then mapped (not accumulated) to the
augmentation rules to create augmented entries.

If an augmentation fails on an entry, it is ignored. The original input dataset
is a subset of the augmented output dataset.

The output dataset has all the columns of the original with two extra:

- `transformed: str`: the augmented `code`. If the entry is not modified,
  `transformed = code`
- `aug_type: str`: the augmentation rule applied in human-readable ID. If the
  entry is not modified, `aug_type = "None"`

Notably, `code` is the original unaugmented code in augmented entries.
