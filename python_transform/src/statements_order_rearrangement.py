import ast

# import astor
import random
import copy


class StatementOrderRearrangement:

    def analyze_dependencies(self):
        """
        Create a comprehensive dependency graph of the code
        Tracks:
        - Variable read/write dependencies
        - Function call side effects
        - Control flow dependencies
        """
        dependency_graph = {}

        class DependencyVisitor(ast.NodeVisitor):
            def __init__(self):
                self.variable_reads = set()
                self.variable_writes = set()
                self.function_calls = set()
                self.control_flow_nodes = set()

            def visit_Name(self, node):
                # Track variable reads and writes
                if isinstance(node.ctx, ast.Load):
                    self.variable_reads.add(node.id)
                elif isinstance(node.ctx, ast.Store):
                    self.variable_writes.add(node.id)
                self.generic_visit(node)

            def visit_Call(self, node):
                # Track function calls
                if isinstance(node.func, ast.Name):
                    self.function_calls.add(node.func.id)
                self.generic_visit(node)

            def visit_Return(self, node):
                self.control_flow_nodes.add("return")
                self.generic_visit(node)

            def visit_Break(self, node):
                self.control_flow_nodes.add("break")
                self.generic_visit(node)

            def visit_Continue(self, node):
                self.control_flow_nodes.add("continue")
                self.generic_visit(node)

        # Analyze dependencies for each statement
        for node in ast.walk(self.original_ast):
            if isinstance(node, (ast.stmt, ast.expr)):
                visitor = DependencyVisitor()
                visitor.visit(node)
                dependency_graph[node] = {
                    "reads": visitor.variable_reads,
                    "writes": visitor.variable_writes,
                    "calls": visitor.function_calls,
                    "control_flow": visitor.control_flow_nodes,
                }

        return dependency_graph

    def can_safely_reorder(self, node1, node2, dependency_graph):
        """
        Determine if two nodes can be safely reordered
        Checks for:
        - Variable dependency conflicts
        - Function call side effects
        - Control flow interactions
        """
        deps1 = dependency_graph[node1]
        deps2 = dependency_graph[node2]

        # Check for write-read dependencies
        write_conflicts = deps1["writes"].intersection(deps2["reads"]) or deps2[
            "writes"
        ].intersection(deps1["reads"])

        # Check for write-write dependencies
        write_write_conflicts = deps1["writes"].intersection(deps2["writes"])

        # Check for control flow constraints
        control_flow_conflicts = deps1["control_flow"] or deps2["control_flow"]

        # Check for function call side effects
        function_call_conflicts = bool(deps1["calls"] or deps2["calls"])

        return not (
            write_conflicts
            or write_write_conflicts
            or control_flow_conflicts
            or function_call_conflicts
        )

    def write(self, source_filename, target_filename):
        with open(source_filename) as file:
            lines = file.readlines()
            source = "".join(lines)
            module = ast.parse(source)

            transformed_code = self.transform_code(module)

            with open(target_filename, "w") as file:
                file.write(transformed_code)

    def transform_code(self, original_ast, max_transformations=5):
        """
        Perform safe code transformations
        """
        # Deep copy to avoid modifying original AST
        transformed_ast = copy.deepcopy(original_ast)

        # Analyze dependencies
        dependency_graph = self.analyze_dependencies()

        # Collect all statement nodes
        statements = [
            node for node in ast.walk(transformed_ast) if isinstance(node, ast.stmt)
        ]

        # Perform transformations
        transformations_count = 0
        for _ in range(len(statements) * 2):  # Multiple pass attempts
            if transformations_count >= max_transformations:
                break

            # Randomly select two different statements
            if len(statements) < 2:
                break

            node1, node2 = random.sample(statements, 2)

            # Check if reordering is safe
            if self.can_safely_reorder(node1, node2, dependency_graph):
                # Swap the statements
                node1_index = statements.index(node1)
                node2_index = statements.index(node2)
                statements[node1_index], statements[node2_index] = (
                    statements[node2_index],
                    statements[node1_index],
                )

                transformations_count += 1

        # Reconstruct the transformed AST
        transformed_module = ast.Module(body=statements, type_ignores=[])
        ast.fix_missing_locations(transformed_module)

        return astor.to_source(transformed_module)
