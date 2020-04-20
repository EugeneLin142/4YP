from typing import Tuple

class TrieNode():

    def __init__(self, char: str, index: int, parent):
        self.char = char
        self.children = []
        self.word_finished = False # is this the last char of the word (end of branch)
        self.counter = 1 # How many times has character appeared in addition process
        self.index = index # Vector feature value selected for this node
        if self.char == '*':
            self.parent = None
        else:
            self.parent = parent



def add(root, word: str):
    """
    Adding a word (could be multiple nodes) to the trie structure
    """
    node = root
    for char in word:
        found_in_child = False
        #Search for the character in the children of the present node
        for child in node.children:
            if child.char == char:
                # Successfully found char, increase the counter by 1 to keep track of how many words has that char
                child.counter += 1
                node = child
                found_in_child = True
                break
        # We didn't find it, so add a new child...
        if not found_in_child:
            try:
                new_index = new_node_calc(node.index, node.parent.index)
            except:
                new_index = new_node_calc(node.index, None)
            new_node = TrieNode(char, new_index, node)
            node.children.append(new_node)
            # point node to new child
            node = new_node
    # Everything finished. Mark as end of a word now.
    node.word_finished = True

def new_node_calc(current_index, parent_index):
    try:
        new_index = abs((current_index - parent_index) // 2)
        # print("parent index is", parent_index)
        # print("node index is", current_index)
        # print("new index is", new_index)
    except:
        new_index = abs(current_index // 2)
        # print("initial", new_index)
    return new_index

def find_word(root, prefix: str) -> Tuple[bool, int]:
    """
    Check and return
      1. If the prefix exsists in any of the words we added so far
      2. If yes then how may words actually have the prefix
    """
    node = root
    # If the root node has no children, then return False.
    # Because it means we are trying to search in an empty trie
    if not root.children:
        return False, 0
    for char in prefix:
        char_not_found = True
        # Search through all the children of the present `node`
        for child in node.children:
            if child.char == char:
                # We found the char existing in the child.
                char_not_found = False
                # Assign node as the child containing the char and break
                node = child
                break
        # Return False anyway when we did not find a char.
        if char_not_found:
            return False, 0
    # Well, we are here means we have found the prefix. Return true to indicate that
    # And also the counter of the last node. This indicates how many words have this
    # prefix
    return True, node.char, node.counter, node.index



if __name__ == "__main__":
    range = 1000000000000
    filepath = "users/bill/appdata/local"

    root = TrieNode('*', range//2, None)

    filepath = filepath.split("/")
    for folder in filepath:
        add(root, folder)

    print(find_word(root, 'appdata'))
    print(find_word(root, 'local'))
