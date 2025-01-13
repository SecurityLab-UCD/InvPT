# PIA
Program-Invariant-Aware Training for Large Language Models in Code Understanding

## Instruction
Run the following command:
```
python3 [RuleId] [RootDir] [OutputDir]

python3 Main.py 1 source output
```

RuleId stands for one specific transformation method:
- 0 Local Varible Renaming
- 1 Function Definition Reordering
- 2 Reverse If Else Statement
- 3 Statements Order Rearrangement
- 4 Operation Assignment to EqualAssignment
- 5 While to For
- 6 For to While

## TODO
[ ] Tests
[ ] For2While: implementation of iterable-for
[ ] For2While: what if `step` is a reference
[ ] While2For: what if the while loop has one else statement
[ ] FunctionDefinitionReorder: methods investigation 