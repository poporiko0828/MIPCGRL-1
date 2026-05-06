from typing import Optional, List
from queue import Queue


class NodeInfo:
    """Represents a node information."""
    def __init__(self, node_id: int, parent_id: int, iteration: int, refer_ids: list = list()) -> None:
        self.node_id: int = node_id
        self.parent_id: int = parent_id
        self.iteration: int = iteration
        self.refer_ids: List[int] = refer_ids

    def __str__(self) -> str:
        return (
            f"NodeInfo(node_id={self.node_id}, parent_id={self.parent_id}, iteration={self.iteration}, refer_ids={self.refer_ids})"
        )


    def to_dict(self) -> dict:
        # save all info
        return {
            'node_id': self.node_id,
            'parent_id': self.parent_id,
            'iteration': self.iteration,
            'refer_ids': self.refer_ids
        }

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def from_dict(data: dict) -> 'NodeInfo':
        return NodeInfo(**data)


class GraphManager:
    def __init__(self, storage: 'Storage', max_breadth: int = 3):
        self.storage = storage
        self.max_breadth = max_breadth  # Maximum number of nodes to expand at each level

    def get_children(self, node: NodeInfo) -> List[NodeInfo]:
        """Returns all child nodes of the given parent node."""
        iterations = self.storage.get_iterations()

        children = list()

        for iteration in iterations:
            if iteration.node.parent_id == node.iteration:
                children.append(iteration.node)

        return children

    def expand_node(self, iteration_num: int) -> Optional[NodeInfo]:
        """Finds an expandable parent node with space for more children, and adds a new child node with a dynamic ID."""

        # Check if there are any iterations in Storage
        if len(self.storage.get_iterations()) == 0:
            node = NodeInfo(node_id=1, parent_id=0, iteration=iteration_num)
            node.update()
            return node

        expandable_nodes = []

        # Iterate over all iterations in Storage
        for iteration in self.storage.get_iterations():

            current_node = iteration.node

            # Check if the current node can expand
            if len(self.get_children(current_node)) < self.max_breadth:
                expandable_nodes.append(current_node)

        # Sort expandable nodes by fitness and select the best one
        if expandable_nodes:
            parent_node = max(expandable_nodes, key=lambda x: self.storage.get_iteration(x.iteration).get_evaluation().total)

            # Generate a new child ID by appending the child number to the parent's ID
            new_child_id = f"{parent_node.node_id}-{len(self.get_children(parent_node)) + 1}"
            new_child = NodeInfo(node_id=new_child_id, parent_id=parent_node.iteration, iteration=iteration_num)

            new_child.update()

            # print(f"Added new child node: {new_child} to parent node: {parent_node}")
            return new_child

        # print("No expandable parent found.")
        return None

    def print_tree(self, iteration_marker: int, best_marker: bool) -> str:
        """Prints the tree structure starting from the first node in Storage as root."""

        iteration = self.storage.get_iteration(iteration_num=1)
        if not iteration:
            print("No iterations found in storage.")
            return ""

        root_node = iteration.node
        if best_marker:
            best_node = max(self.storage.get_iterations(), key=lambda x: x.get_evaluation().total).node
        else:
            best_node = None

        tree_str = self._build_tree_string(root_node, iteration_marker=iteration_marker, best_node=best_node, depth=0)
        return tree_str

    def _build_tree_string(self, node: NodeInfo, iteration_marker: int, best_node: Optional[NodeInfo],
                           depth: int = 0) -> str:
        """Helper function to build the tree structure as a single string."""

        iteration = self.storage.get_iteration(node.iteration)
        fitness = iteration.get_evaluation().total if iteration else "N/A"

        # Add markers for best and iteration
        marker = ""
        if node.iteration == iteration_marker:
            marker += " [*Iteration*]"
        if best_node and node.node_id == best_node.node_id:
            marker += " [*Best*]"

        result = "    " * depth + f"{node} (fitness: {fitness}){marker}\n"

        children = self.get_children(node)
        if not children:
            return result

        for child in children:
            result += self._build_tree_string(child, iteration_marker, best_node, depth + 1)

        return result

    def update(self, node: NodeInfo, **kwargs) -> None:
        """Updates the node with the given key-value pairs."""
        node.update(**kwargs)

        # find the iteration
        iteration = self.storage.get_iteration(node.iteration)
        iteration.set_node(node)

    def __str__(self) -> str:
        return f"GraphManager(max_breadth={self.max_breadth})"

    def get_node(self, interation_num: int) -> Optional[NodeInfo]:
        iteration = self.storage.get_iteration(interation_num)
        if iteration:
            return iteration.node
        return None

    from typing import List

    def get_best_iteration_nums(self, max_n: int, excludes: list = list()) -> List[int]:
        """Returns the best n iterations excluding the ones in the excludes list."""
        iterations = self.storage.get_iterations()
        iterations = [iteration for iteration in iterations if iteration.iteration_num not in excludes]

        # Sort iterations based on fitness in descending order
        best_iterations = sorted(iterations, key=lambda x: x.get_evaluation().total, reverse=True)

        # Get the top n iterations, or all available iterations if there are fewer than n
        best_iteration_nums = [iteration.iteration_num for iteration in best_iterations[:min(max_n, len(best_iterations))]]

        return best_iteration_nums
