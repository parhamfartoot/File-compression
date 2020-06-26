"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode

# ====================

# Helper functions for uncompress

def reverse_code(text,dict,size):
    encode = ''
    current = ''
    decode = []
    for byte in text:
        encode += byte_to_bits(byte)
    encode = str(encode)
    i = 0
    for bit in encode:
        current += bit
        if i < size:
            if current in dict:
                char = dict[current]
                decode.append(char)
                i += 1
                current = ''
    return decode


def reverse_dict(tree):
    dict = get_codes(tree)
    rev_dict = {v: k for k, v in dict.items()}
    return rev_dict


#  Helper Functions
def improve(tree,cop):

    if tree == None:
        return
    if tree.is_leaf() :
        tree.symbol = max(cop, key=cop.get)
        del cop[tree.symbol]
    improve(tree.left,cop)
    improve(tree.right,cop)
    return tree


def post_order(tree,byte):
    if tree.left:
        post_order(tree.left,byte)
    if tree.right:
        post_order(tree.right,byte)
    if tree.is_leaf() == False:
        if tree.left.is_leaf():
            byte.append(0)
            byte.append(tree.left.symbol)
        elif not tree.left.is_leaf():
            byte.append(1)
            byte.append(tree.left.number)

        if tree.right.is_leaf():
            byte.append(0)
            byte.append(tree.right.symbol)
        elif not tree.right.is_leaf():
            byte.append(1)
            byte.append(tree.right.number)
    return byte
def postorder_number(tree,internal) -> None:

    # Assigns number to the internal nodes of the tree in Postorder starting at 0

    if tree.left:
        postorder_number(tree.left,internal)
    if tree.right :
        postorder_number(tree.right,internal)
    if tree.is_leaf() == False:
        internal.append(tree)
    for i in range (0,len(internal)):
        internal[i].number = i




def get_codes_helper(tree,current_code,dict) -> dict:
    # Assigns Code to each symbol in the huffman tree
    if tree == None:
        return
    if tree.symbol != None:
        dict[tree.symbol] = current_code
    get_codes_helper(tree.left, current_code + "0",dict)
    get_codes_helper(tree.right, current_code + "1",dict)
    return dict

def find_min(L):
    min = L[0][1]
    res = (0, 0)
    for item in L:
        if item[1] < min:
            min = item[1]
    for item in L:
        if min == item[1]:
            res = (item[0], item[1])
    return res

def make(freq_dict):
    List = []
    cop_freq2=dict(freq_dict)
    for key in cop_freq2.keys():
        List.append((HuffmanNode(key),cop_freq2[key]))
    return List


# ===================================
# Helper functions for manipulating bytes

def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict(int,int)

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    frequency = {}
    for character in text:
        if not character in frequency:
            frequency[character] = 0
        frequency[character] += 1
    return frequency




def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    copfreq = dict(freq_dict)
    List = make(copfreq)
    while len(List) > 1:
        min1 = find_min(List)
        key1 = min1[0]
        List.remove(min1)
        min2 = find_min(List)
        key2 = min2[0]
        List.remove(min2)
        merged = HuffmanNode(None, key1, key2)
        value = min1[1]+min2[1]
        List.append((merged,value))
    return List[0][0]


def get_codes(tree):
    """ Return a dict mapping symbols from Huffman tree to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    dict ={}
    current_code = ''
    return get_codes_helper(tree,current_code,dict)


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2

    """
    # See postorder helper function
    postorder_number(tree,[])



def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    counter = 0
    sum = 0
    code_dict = get_codes(tree)
    for key in freq_dict:
        sum += len(code_dict[key]) * freq_dict[key]
        counter +=freq_dict[key]
    if counter != 0:
        res = sum / counter
        #res = float(format(res, '.1f'))
        return res
    else:
        return 0.0


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mapping from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    comp = ''
    comper = []
    res = []
    for byte in text:
        if byte in codes:
            comp = comp + str(codes[byte])
            if len(comp) > 8:
                comper.append(comp[0:8])
                comp = comp[8:]
    comper.append(comp)
    for byte in comper:
        res.append(bits_to_byte(byte))
    res = bytes(res)
    return res




def tree_to_bytes(tree):
    """ Return a bytes representation of the Huffman tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    return bytes(post_order(tree,[]))




def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file to compress
    @param str out_file: output file to store compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in node_lst.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in 'node_lst'
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    root = node_lst[root_index]
    if root.l_type == 0:
        left = HuffmanNode(root.l_data)
    else:
        left = generate_tree_general(node_lst,root.l_data)

    if root.r_type == 0:
        right = HuffmanNode(root.r_data)
    else:
        right = generate_tree_general(node_lst,root.r_data)
    return HuffmanNode(None,left,right)


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that node_lst represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in 'node_lst'
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """

    root = node_lst[root_index]

    if root.l_type == 0:
        left = HuffmanNode(root.l_data)
    else:
        left = generate_tree_general(node_lst, root.l_data)
    del node_lst[0]

    if root.r_type == 0:
        right = HuffmanNode(root.r_data)
    else:
        right = generate_tree_general(node_lst, root.r_data)
    return HuffmanNode(None, left, right)




def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: number of bytes to decompress from text.
    @rtype: bytes
    """
    reverse = reverse_dict(tree)
    code = reverse_code(text,reverse,size)
    return bytes(code)



def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        print(size)
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    cop = dict(freq_dict)
    improve(tree,cop)



#'''
if __name__ == "__main__":
    import doctest
    doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
#'''
