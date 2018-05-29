"""
Unit tests for the tree class
"""
# noinspection PyShadowingNames
# pylint: disable=redefined-outer-name,missing-docstring,unused-variable,unused-argument,
# pylint: disable=protected-access,unidiomatic-typecheck
import io
import pickle
import threading
import time
from collections import namedtuple
from copy import deepcopy, copy
from unittest import mock
import pytest
import blinker
from pympler.asizeof import asizeof

from bushn import (DELETE, DuplicateChildError, IndexingNode, Node,
                   NotifyingDict)

__author__ = 'CrossCloud GmbH'


@pytest.fixture(params=['standard', 'indexed'])
def root(request):
    """
    Creates a default Root node
    :return: Node
    """
    if request.param == 'standard':
        return Node(name=None)
    return IndexingNode(name=None, indexes=['_id'])


@pytest.fixture
def one_child(default_root):
    """ a root with one child """
    return (default_root, default_root.add_child('one_child', {'one': 'props'}))


@pytest.fixture
def two_child(one_child):
    """ a root with one child """
    default_root, one_child = one_child
    return (default_root, one_child.add_child('two_child', {'two': 'props'}))


@pytest.fixture(params=['standard', 'factory'])
def default_root(root, request):
    """ Creates a default Root node """
    if request.param == 'standard':
        return root

    new_root = root.create_instance()
    root.parent = new_root
    root.name = 'old_root'
    return new_root

@pytest.fixture
def default_root_two_children(default_root):
    """ Creates a default Root node """
    default_root.add_child('a').add_child('a_a')
    default_root.add_child('b').add_child('b_a')
    return default_root

def test_add_and_check(default_root):
    """  Add a element to the tree and check if it is added  """
    original_len = len(default_root)
    childs = create_children(default_root)

    assert childs.node.path <= childs.path
    assert childs.node.depth == 100
    assert len(default_root) == 100 + original_len

    node = default_root.get_node(childs.path)
    assert childs.node is node


def test_child_iterator(default_root):
    """
    Iterate over all children
    :return: None
    """
    children_before = set(default_root)

    new_children = create_children(default_root)

    assert set(default_root) == children_before | new_children.all_childs


@pytest.fixture
def default_root_with_events(default_root):
    event_mock = mock.MagicMock()

    # weak is set to false to be able to use mock objects with blinker
    default_root.on_create.connect(event_mock.on_create, weak=False)
    default_root.on_delete.connect(event_mock.on_delete, weak=False)
    default_root.on_moved.connect(event_mock.on_moved, weak=False)
    default_root.on_renamed.connect(event_mock.on_renamed, weak=False)
    default_root.on_update.connect(event_mock.on_update, weak=False)

    new_child = default_root.add_child('Hello Childs', props={'Aha': 'not oho!'})
    return default_root, event_mock, new_child


def test_pass_notifying_dict_constructor(default_root_with_events):
    """When we pass a notifying as props to the node constructor, on a different tree,
    we expect that the events are not triggered on the old tree.
    """
    default_root, event_mock, _ = default_root_with_events
    props = {'a': 'b'}
    node1 = default_root.add_child('mynode', props)

    root2 = default_root.create_instance(name=None)
    node2 = root2.add_child('secondnode', node1.props)
    node2.props['asdf'] = 17
    event_mock.on_update.assert_not_called()


def test_pass_notifying_dict_setter(default_root_with_events):
    """When we pass a notifying as props to the props setter, on a different tree,
    we expect that the events are not triggered on the old tree.
    """
    default_root, event_mock, _ = default_root_with_events
    props = {'a': 'b'}
    node1 = default_root.add_child('mynode', props)

    root2 = default_root.create_instance(name=None)
    node2 = root2.add_child('secondnode', {})
    node2.props = node1.props
    node2.props['asdf'] = 17
    event_mock.on_update.assert_not_called()


def test_signals(default_root_with_events):
    """ tests the signalling in the trmee """
    _, event_mock, new_child = default_root_with_events
    event_mock.on_create.assert_called_once_with(new_child)
    event_mock.on_moved.assert_not_called()


def test_signal_move(default_root_with_events):
    default_root, event_mock, new_child = default_root_with_events

    the_other = default_root.add_child("the other")
    new_child.parent = the_other
    event_mock.on_moved.assert_called_once_with(new_child, old_parent=default_root)


def test_monkey_move(default_root):
    """When a node is assigned a new parent, it must be removed from the old parent.

    At first monkey and banana are in root. But what will happen when the banana is placed
    in the monkey. Come find out.
    """
    monkey = default_root.add_child('monkey')
    banana = default_root.add_child('banana.txt')

    assert default_root.get_node(['monkey']) == monkey
    assert default_root.get_node(['banana.txt']) == banana

    prev_len = len(default_root.children)

    # move
    banana.parent = monkey

    assert default_root.get_node(['monkey']) == monkey
    assert default_root.get_node(['monkey', 'banana.txt']) == banana
    assert len(default_root.children) == prev_len - 1
    assert len(monkey.children) == 1
    assert default_root.path == []

    with pytest.raises(KeyError):
        assert default_root.get_node(['banana.txt']) == banana


def test_signal_rename(default_root_with_events):
    default_root, event_mock, new_child = default_root_with_events
    new_child.name = 'Hello Funny Guy'
    event_mock.on_renamed.assert_called_once_with(new_child, old_name='Hello Childs')


@pytest.mark.parametrize('value', ['oho!', None, 0])
def test_signal_update_prop(default_root_with_events, value):
    default_root, event_mock, new_child = default_root_with_events

    new_child.props['Aha'] = value
    event_mock.on_update.assert_called_once_with(new_child, other={'Aha': value})


@pytest.mark.parametrize('value', ['oho!', None, 0])
def test_signal_assign_new_item(default_root_with_events, value):
    default_root, event_mock, new_child = default_root_with_events

    new_child.props['asdfasdf'] = value
    event_mock.on_update.assert_called_once_with(new_child, other={'asdfasdf': value})


def test_signal_update_prop_no_change(default_root_with_events):
    default_root, event_mock, new_child = default_root_with_events

    # # A more complicated use case: change, delete, update
    new_child.props['Aha'] = 'not oho!'
    event_mock.on_update.assert_not_called()


def test_signal_update_propV2(default_root_with_events):
    default_root, event_mock, new_child = default_root_with_events
    # check the update function
    new_child.props.update({'123': 'Gadi'})
    event_mock.on_update.assert_called_once_with(new_child, other={'123': 'Gadi'})


def test_signal_delete_prop(default_root_with_events):
    default_root, event_mock, new_child = default_root_with_events

    del new_child.props['Aha']
    event_mock.on_update.assert_called_once_with(new_child, other={'Aha': DELETE})


def test_signal_delete_prop_by_pop(default_root_with_events):
    default_root, event_mock, new_child = default_root_with_events

    new_child.props.pop('Aha', None)
    event_mock.on_update.assert_called_once_with(new_child, other={'Aha': DELETE})


def test_signal_delete_prop_is_hiding(default_root_with_events):
    default_root, event_mock, new_child = default_root_with_events
    # # deletion of a non existing node should not raise an event
    with pytest.raises(KeyError):
        del new_child.props['in hiding']
    assert not event_mock.on_update.called


def test_signal_update_prop_to_empty(default_root_with_events):
    default_root, event_mock, new_child = default_root_with_events
    # check the update function
    new_child.props.update({})
    event_mock.on_update.assert_not_called()


def test_signal_node_delete(default_root_with_events):
    default_root, event_mock, new_child = default_root_with_events
    new_child.delete()
    event_mock.on_delete.assert_called_once_with(new_child)


def test_signal_assign_new_props(default_root_with_events):
    default_root, event_mock, new_child = default_root_with_events
    default_root.props = {'Aha': 123}

    event_mock.on_update.assert_called_once_with(default_root, other={'Aha': 123})


def test_signal_assign_update_prop(default_root_with_events):
    default_root, event_mock, new_child = default_root_with_events
    new_child.props = {'Aha': 123}

    event_mock.on_update.assert_called_once_with(new_child, other={'Aha': 123})


def test_signal_assign_delete_props(default_root_with_events):
    default_root, event_mock, new_child = default_root_with_events
    new_child.props = {}

    event_mock.on_update.assert_called_once_with(new_child, other={'Aha': DELETE})


def test_signal_assign_update(default_root_with_events):
    default_root, event_mock, new_child = default_root_with_events
    new_child.props = {'nom': 93}

    event_mock.on_update.assert_called_once_with(new_child, other={'Aha': DELETE, 'nom': 93})


def test_signal_copy_of_props(default_root_with_events):
    default_root, event_mock, new_child = default_root_with_events

    the_copy = copy(new_child.props)
    the_copy['nom'] = 93
    event_mock.on_update.assert_not_called()


def test_signal_deepcopy_of_props(default_root_with_events):
    default_root, event_mock, new_child = default_root_with_events

    the_copy = copy(new_child.props)
    the_copy['nom'] = 93
    event_mock.on_update.assert_not_called()


def test_signal_deepcopy_of_node(default_root_with_events):
    """Tests if signaling in a copy of a tree still works"""
    default_root, event_mock, new_child = default_root_with_events

    new_event_mock = mock.Mock()
    the_copy = deepcopy(new_child)
    the_copy_root = the_copy.parent
    the_copy.on_update.connect(new_event_mock, weak=False)
    the_copy.props['nom'] = 93

    for node in the_copy_root:
        assert node.lock is the_copy_root.lock
        for sig in Node.SIGNALS:
            assert getattr(node, sig) is getattr(the_copy_root, sig)

    assert the_copy_root.lock is not default_root.lock
    for sig in Node.SIGNALS:
        assert getattr(default_root, sig) is not getattr(the_copy_root, sig)

    event_mock.on_update.assert_not_called()
    new_event_mock.assert_called_with(the_copy, other={'nom': 93})


def test_signal_dict_copy_of_props(default_root_with_events):
    default_root, event_mock, new_child = default_root_with_events

    the_copy = new_child.props.copy()
    the_copy['nom'] = 93
    event_mock.on_update.assert_not_called()


def test_assign_props_is_notifying(default_root):
    """by assigning a prop, a notifying dict is created.
    """
    default_root.props = {'Hello': 123}

    assert isinstance(default_root.props, NotifyingDict)


@pytest.mark.parametrize("operation,slot_name",
                         [(lambda x, y: x.add_child('Hello'), 'on_create'),
                          (lambda x, y: setattr(x, 'name', 'other name'), 'on_renamed'),
                          (lambda x, y: x.props.update({'x': 'y'}), 'on_update'),
                          (lambda x, y: x.delete(), 'on_delete'),
                          (lambda x, y: setattr(x, 'parent', y), 'on_moved')])
def test_signal_deleted_child(default_root, operation, slot_name):
    """ a deleted child never should emit signals to the tree it had before """

    child1 = default_root.add_child('child 1')
    child3 = child1.add_child('child 3')
    child2 = child1.add_child('child 2')

    child1.delete()

    func_mock = mock.Mock()
    getattr(default_root, slot_name).connect(func_mock, weak=False)

    operation(child2, child3)

    func_mock.assert_not_called()


def test_delete_and_check(default_root):
    """
    Add a element to the tree and check if it is added
    :return: None
    """
    len_before = len(default_root)

    childs = create_children(default_root)

    nodetodelete = childs.node
    for _ in range(49):
        nodetodelete = nodetodelete.parent
    lastnode = nodetodelete.parent
    nodetodelete.delete()

    assert lastnode.depth == 50
    assert len(default_root) == 50 + len_before


def test_node_props_cls():
    """Check that Node correctly uses a custom props type passed in."""
    class MyFancyProps:
        def __init__(self, *args, **kwargs):
            self.on_update = blinker.Signal()

    test_node = Node(None, props_cls=MyFancyProps)

    assert type(test_node.props) == MyFancyProps

    test_node.props = None

    assert type(test_node.props) == MyFancyProps


def test_root_node(default_root):
    """
    Check the root node of an empty tree
    :return: None
    """
    assert isinstance(default_root, Node)
    assert default_root.parent is None
    assert default_root.path == []
    assert default_root.name is None
    assert default_root.props == {}
    assert default_root.depth == 0


def test_node_equals_simple():
    """
    Checks the node equals method
    :return:
    """
    node1 = Node(name="aaaa", props={"a": "b", "c": "d"})
    node2 = Node(name="aaaa", props={"a": "b", "c": "d"})
    node3 = Node(name="bbbb", props={"a": "b", "c": "d"})
    node4 = Node(name="aaaa", props={"a": "b", "c": "e"})

    assert node1 == node2
    assert node1 != node3
    assert node2 == node4


def test_node_equals_recursive():
    """
    Compares two nodes and their children
    :return:
    """

    node1 = Node(name="aaaa", props={"a": "b", "c": "d"})
    node2 = Node(name="aaaa", props={"a": "b", "c": "d"})

    node1.add_child("bbbb", props={"a": "b", "c": "d"})
    node2.add_child("aaaa", props={"a": "b", "c": "e"})

    assert node1.subtree_equals(node2) is False

    node3 = Node(name="aaaa", props={"a": "b", "c": "d"})
    node4 = Node(name="aaaa", props={"a": "b", "c": "d"})

    node3.add_child("bbbb", props={"a": "b", "c": "d"})
    node41 = node4.add_child("bbbb", props={"a": "b", "c": "d"})

    assert node3.subtree_equals(node4)
    assert not node3.subtree_equals(node41)


def test_parent_rename_node():
    """
    Renames a node and changes it's parent to test the path functionality
    :return:
    """
    node1 = Node(name="aaaa", props={"a": "b", "c": "d"})
    node2 = Node(name="bbbb", props={"a": "b", "c": "d"})
    node3 = Node(name="cccc", props={"a": "b", "c": "d"})

    assert node1.path == ['aaaa']
    assert node2.path == ['bbbb']
    assert node3.path == ['cccc']

    node2.parent = node1
    node3.parent = node2

    assert node2.path == ['aaaa', 'bbbb']
    assert node3.path == ['aaaa', 'bbbb', 'cccc']

    node2.name = 'dddd'

    assert node2.path == ['aaaa', 'dddd']
    assert node3.path == ['aaaa', 'dddd', 'cccc']

    assert node1.get_node(['dddd']) == node2


def test_big_balanced_tree(default_root):
    """
    Create a huge tree to check performance and test a small intersection
    """
    create_equal_distributed_tree(default_root, 50)

    other_root = Node(None)

    create_equal_distributed_tree(other_root, 4)
    all_items = set(default_root)

    all_other_items = set(other_root)

    # the root plus 4 items
    assert len(all_items.intersection(all_other_items)) == 5

    print(len(all_items))


def test_lock_refs(default_root):
    """
    Tests if children are created, if they keep the right lock reference
    """

    the_lock = default_root.lock

    assert default_root.lock is the_lock

    node = default_root.add_child(name="Test")
    assert node.lock is the_lock


def test_copy(default_root):
    """
    Copies a tree and checks if the copy is equal to the original one
    :param default_root: fixture
    """
    create_equal_distributed_tree(default_root, 10)
    for node in default_root:
        default_root.props['Hi'] = 'you'

    copied_root = deepcopy(default_root)

    assert default_root == copied_root
    assert default_root.lock is not copied_root.lock
    assert default_root is not copied_root

    for node in default_root:
        new_node = copied_root.get_node(node.path)
        assert new_node.props == node.props
        print(type(new_node.props))
        print(type(node.props))
        assert isinstance(new_node.props, NotifyingDict)

    # this fails with `AttributeError: on_update` if the properties have not been
    # copied properly
    copied_root.props['Hello'] = 'World'

    assert copied_root.props['Hi'] == 'you'


def test_pickle(default_root_two_children):
    """
    pickles a tree and checks if the pickled is equal to the original one
    :param default_root: fixture
    """
    create_equal_distributed_tree(default_root_two_children, 10)
    fileobj = io.BytesIO()

    default_root_two_children.lock.acquire()
    default_root_two_children.props['Hi'] = 123

    assert default_root_two_children.lock
    pickle.dump(default_root_two_children, fileobj)
    fileobj.seek(0)

    copied_root = pickle.load(fileobj)

    assert copied_root._props_cls != dict
    assert type(copied_root.props) != dict

    def check_lock():
        """
        Checks the deserialized object
        """
        assert copied_root.lock.acquire(False)
        assert default_root_two_children == copied_root
        assert default_root_two_children.lock is not copied_root.lock
        assert default_root_two_children is not copied_root

    thread = threading.Thread(target=check_lock)
    thread.start()
    thread.join()

    assert copied_root.props['Hi'] == 123

    # this fails with `AttributeError: on_update` if the properties have not been
    # copied properly
    copied_root.props['Hello'] = 'World'


def test_deepcopy(default_root):
    """
    pickles a tree and checks if the pickled is equal to the original one
    :param default_root: fixture
    """
    create_equal_distributed_tree(default_root, 10)
    default_root.props['Hi'] = 123

    copied_root = deepcopy(default_root)

    assert copied_root._props_cls != dict
    assert type(copied_root.props) != dict

    # this fails with `AttributeError: on_update` if the properties have not been
    # copied properly
    copied_root.props['Hello'] = 'World'


def test_has_child():
    '''
    Tests has child function.
    '''
    root_node = Node(name=None)
    assert not root_node.has_child('test')

    root_node.add_child('test')
    assert root_node.has_child('test')


def test_delete_delete():
    """
    Tests if 2 consequent delete on the same child will not raise exceptions
    """
    root_node = Node(name=None)
    child = root_node.add_child('test')

    child.delete()
    child.delete()


def test_hash_function():
    """
    check if the hash function is only effected by the path
    :return: None
    """
    root_node1 = Node(name=None)
    root_node2 = Node(name=None)
    assert hash(root_node1) == hash(root_node2)

    child1 = root_node1.add_child("that child")
    assert hash(child1) != hash(root_node1)

    child2 = root_node2.add_child("that child")
    assert hash(child1) == hash(child2)

    child3 = child2.add_child("Hulla")
    assert hash(child3) != hash(child2)


def test_set_operations():
    """
    check if the set operations are only effected by the path
    """
    root_node1 = Node(name=None)
    root_node2 = Node(name=None)
    root_node1.add_child("that child")
    child2 = root_node2.add_child("that child")
    child3 = child2.add_child("Hulla")

    assert {child3} == set(root_node2) - set(root_node1)

    child2.props = {'hola': 'drio'}
    assert {child3} == set(root_node2) - set(root_node1)


def test_get_node_safe():
    """ Tests get_node_safe
    """

    root = Node(None)
    path = ['a', 'b', 'c', 'd']
    node = root.get_node_safe(path)
    assert path == node.path

    path = ['a', 'b', 'f']
    node = root.get_node_safe(path)
    assert path == node.path

    path = ['bb']
    node = root.get_node_safe(path)

    assert root.get_node(['bb']) is node

    assert path == node.path


def test_setdefault_existing(one_child):
    """ tests setdefault with existing child"""
    # todo, assign case
    root, one_child = one_child

    # child = root.setdefault(['one_child'], props={'hi': 'there'})
    child = root.setdefault(['one_child'], props={'one': 'there'})

    assert child is one_child
    assert child.props == {'one': 'there'}
    # assert child.props == {'hi': 'there'}


def test_setdefault_new(one_child):
    """ tests setdefault with existing child"""
    root, one_child = one_child

    child = root.setdefault(['other_child'], props={'hi': 'there'})

    assert child.props == {'hi': 'there'}

    # Create a second child without props
    child = root.setdefault(['other_child_two'])

    assert child.props == {}

def create_equal_distributed_tree(parent, number_childs):
    """
    :param parent:
    :param number_childs:
    """
    for child_number in range(number_childs // 2):
        child = parent.add_child(name="Child {}".format(child_number),
                                 props={'small': 'prop'})
        create_equal_distributed_tree(parent=child,
                                      number_childs=number_childs // 2)


def create_children(parent, depth=100):
    """
    Creates a subtree of the parent node with a given depth
    :param parent: the root node
    :param depth: the depth
    """
    path = []
    all_children = set()
    for cnt in range(depth):
        name = 'child_{}'.format(cnt)
        node = parent.add_child(name=name)
        path.append(name)
        all_children.add(node)
        parent = node

    children = namedtuple('Childs', ['node', 'path', 'all_childs'])
    return children(node=node, path=path, all_childs=all_children)


def test_iter_up():
    """
    Tests the
    """
    root = Node(name=None)
    assert list(root.iter_up) == [root]

    child1 = root.add_child('test')
    assert list(child1.iter_up) == [child1, root]

    child2 = child1.add_child('test')
    assert list(child2.iter_up) == [child2, child1, root]


def test_iter_up_existing():
    """
    Tests the iter_up_existing generator
    """
    root = Node(name=None)
    assert list(root.iter_up_existing(['node'])) == [root]
    assert list(root.iter_up_existing(['node', 'bla', 'foo'])) == [root]

    child1 = root.add_child('test')
    assert list(root.iter_up_existing(['test', 'node'])) == [child1, root]
    assert list(child1.iter_up_existing(['node'])) == [child1]

    child2 = child1.add_child('test')
    assert list(root.iter_up_existing(['test', 'test', 'node'])) == [child2, child1, root]
    assert list(child1.iter_up_existing(['test', 'node'])) == [child2, child1]
    assert list(child2.iter_up_existing(['node'])) == [child2]


@pytest.mark.skip('external properties not supported ATM')
def test_external_props():
    """ tests if an external properties object is taken even if it evaluates to false"""

    class CustomProps:
        # pylint: disable=missing-docstring

        def __init__(self, node, *args, **kwargs):
            self.node = node

        def update(self, *args, **kwargs):
            pass

    myfanzyprops = CustomProps(node=None)

    node = Node(None, props=myfanzyprops, props_cls=CustomProps)

    assert node.props.node is node
    assert isinstance(node.props, CustomProps)


def test_child_same_name(default_root):
    """
    test tests if a key error is thrown a child with the same name is added
    """
    child = default_root.add_child(name='Hello')
    print(list(default_root))
    with pytest.raises(KeyError):
        default_root.add_child(name='Hello')

    print(list(default_root))
    assert child is default_root.get_node(['Hello'])


def test_cyclic_tree_detection(default_root):
    """ check if a cylic tree inser throws a RecuresionError """
    child_a = default_root.add_child(name='Hello')
    child_b = child_a.add_child(name='Hello')
    child_c = child_b.add_child(name='Hello')
    with pytest.raises(RuntimeError):
        child_a.parent = child_c


@pytest.fixture(params=['standard', 'factory', 'copied', 'pickled'])
def indexing_node(request):
    """ fixture to test the indexing node """
    if request.param == 'standard':
        return IndexingNode(None, indexes=['_id'])

    elif request.param == 'factory':
        old_root = IndexingNode(None, indexes=['_id'])
        new_root = old_root.create_instance()
        old_root.name = 'hello'
        old_root.parent = new_root
        return new_root
    elif request.param == 'copied':
        old_root = IndexingNode(None, indexes=['_id'])
        return deepcopy(old_root)
    elif request.param == 'pickled':
        old_root = IndexingNode(None, indexes=['_id'])
        fobj = io.BytesIO()
        pickle.dump(old_root, fobj)
        fobj.seek(0)
        return pickle.load(fobj)

    else:
        assert False


def test_root(indexing_node):
    """Test if the index works on the root"""

    indexing_node.props['_id'] = 123
    assert indexing_node.index['_id'][123] is indexing_node


def test_indexing_node(indexing_node):
    """ general tests for indexing node """

    new_node = indexing_node.get_node_safe(['a', 'b', 'c'])
    new_node.props['_id'] = 123

    print(indexing_node.index)

    assert indexing_node.index['_id'][123] is new_node

    # try to set a not index props
    new_node.props['other_prop'] = 312

    # that should also not be in here
    assert 'other_prop' not in indexing_node.index


def test_indexing_node_no_parent_or_index():
    with pytest.raises(ValueError):
        node = IndexingNode(None)


def test_indexing_update(indexing_node):
    """ test if the indexes are refreshed with the update method the props """
    node = indexing_node.add_child(name="Hello")
    node.props.update({'_id': 'Holla'})
    assert 'Holla' in indexing_node.index['_id']


def test_indexing_props_add_child_with_props(indexing_node):
    """ test if the indexes are refreshed when adding a choild with the props kwarg """
    indexing_node.add_child(name="Hello", props={'_id': 'Holla'})
    assert 'Holla' in indexing_node.index['_id']


def test_indexing_props_delete_node(indexing_node):
    """ test if the indexes are deleted, if the node is removed """
    node = indexing_node.add_child(name="Hello", props={'_id': 'Holla'})
    node.delete()
    assert 'Holla' not in indexing_node.index['_id']
    assert node not in indexing_node.index['_id'].values()


def test_indexing_unique_violation(indexing_node):
    """ tests if a node is created with an already existing index value, a key error
    should be thrown to keep unique items in the index, and not overwrite any """
    old_len = len(indexing_node)

    indexing_node.add_child(name="Hello", props={'_id': 'Holla'})
    with pytest.raises(KeyError):
        indexing_node.add_child(name="ASD", props={'_id': 'Holla'})

    # check if the node is not created
    assert len(indexing_node) == 1 + old_len


def test_indexing_not_unique_violation(indexing_node):
    """ tests if a node is created with an already existing index value, a key error
    should be thrown to keep unique items in the index, and not overwrite any """
    node = indexing_node.add_child(name="Hello", props={'_id': 'Holla'})

    # id stays the same, that is ok and no exception should be thrown
    node.props.update({'_id': 'Holla'})


def test_indexing_props_set_new_value(indexing_node):
    """ test if the indexes are deleted, if the node is removed """
    node = indexing_node.add_child(name="Hello", props={'_id': 'Holla'})
    node.props['_id'] = 'New WOW'
    assert 'Holla' not in indexing_node.index['_id']
    assert node is indexing_node.index['_id']['New WOW']


def test_indexing_props_pop_value(indexing_node):
    """ test if the indexes are deleted, if the node is removed """
    node = indexing_node.add_child(name="Hello", props={'_id': 'Holla'})
    value = node.props.pop('_id')
    assert value == 'Holla'
    assert 'Holla' not in indexing_node.index['_id']

    assert node not in indexing_node.index['_id'].values()


def test_merge_set_with_id(indexing_node):
    """ tests the merge functionality """
    indexing_node.props['_id'] = 0

    indexing_node.add_child('Hello', {'_id': 1})

    mergeset = [('UPSERT', 2, 'Other Name', {'_id': 11}),
                ('UPSERT', 0, 'New Node', {'_id': 2}),
                ('UPSERT', 0, 'Other Name', {'_id': 1})]

    assert indexing_node.add_merge_set_with_id(mergeset, key='_id') == []

    assert indexing_node.index['_id'][1].name == 'Other Name'

    mergeset = [('DELETE', 2, 'New Node', {'_id': 11})]
    assert indexing_node.add_merge_set_with_id(mergeset, key='_id') == []

    assert 11 not in indexing_node.index['_id']

    # Removing a node that is not in the index should do nothing
    mergeset = [('DELETE', 0, 'Invalid Node', {'_id': 9999})]
    assert indexing_node.add_merge_set_with_id(mergeset, key='_id') == []

    assert 0 in indexing_node.index['_id']
    assert 1 in indexing_node.index['_id']
    assert 2 in indexing_node.index['_id']


def test_merge_set_with_id_using_update(indexing_node):
    """ tests the merge functionality """
    indexing_node.props['_id'] = 0

    mergeset = [('UPSERT', 0, 'New Node', {'_id': 2}),
                ('UPSERT', 0, 'Other Name', {'_id': 1})]

    assert indexing_node.add_merge_set_with_id(mergeset, key='_id') == []

    # Update node with a new property using .update()
    mergeset = [('UPSERT', 0, 'New Node', {'_id': 2, '_test_prop': 'TestProp'})]
    assert indexing_node.add_merge_set_with_id(mergeset, key='_id', using_update=True) == []

    assert indexing_node.index['_id'][2].props['_test_prop'] == 'TestProp'

    # Update node, skipping the previously added property using .update()
    # -> it should still be there
    mergeset = [('UPSERT', 0, 'New Node', {'_id': 2})]
    assert indexing_node.add_merge_set_with_id(mergeset, key='_id', using_update=True) == []

    assert indexing_node.index['_id'][2].props['_test_prop'] == 'TestProp'


def test_merge_set_with_id_dulicate_mergeset(indexing_node):
    """ tests the merge with duplicate entries functionality """
    indexing_node.props['_id'] = 0

    mergeset = [('UPSERT', 0, 'New Node', {'_id': 1, 'xyz': 'qwert'}),
                ('UPSERT', 0, 'New Node', {'_id': 1, 'xyz': 'qwertz'})]

    assert indexing_node.add_merge_set_with_id(mergeset, key='_id') == []
    assert indexing_node.index['_id'][1].props == {'xyz': 'qwertz', '_id': 1}


def test_merge_set_with_id_double_child(indexing_node):
    """ tests the merge with duplicate entries functionality """
    indexing_node.props['_id'] = 0

    mergeset = [('UPSERT', 0, 'New Node', {'_id': 1, 'xyz': 'qwert'}),
                ('UPSERT', 0, 'New Node', {'_id': 2, 'xyz': 'qwertz'})]

    assert indexing_node.add_merge_set_with_id(mergeset, key='_id') == []
    assert indexing_node.index['_id'][1].props == {'xyz': 'qwert', '_id': 1}


def test_merge_set_with_id_multiple_parent_event(indexing_node):
    """ tests the merge with multiple parents entries functionality
        this should result in the element mounted only in one (the first)
        parent in the list having it as child only
    """
    indexing_node.props['_id'] = 0

    mergeset = [('UPSERT', 0, 'second parent', {'_id': 1, 'xyz': 'asdf'}),
                ('UPSERT', 0, 'child', {'_id': 2, 'xyz': 'qwert'}),
                ('UPSERT', 1, 'child', {'_id': 2, 'xyz': 'qwert'})]

    assert indexing_node.add_merge_set_with_id(mergeset, key='_id') == []
    assert indexing_node.index['_id'][2].props == {'xyz': 'qwert', '_id': 2}
    assert indexing_node.index['_id'][2].parent == indexing_node.index['_id'][1]


def test_index_update_recursive_delete(indexing_node):
    """ tests the merge functionality """
    indexing_node.props['_id'] = 0

    node1 = indexing_node.add_child('Hello', {'_id': 1})
    node1.add_child('Hello 2', {'_id': 2})

    node1.delete()

    # the node itself of course should not be in the index
    assert 1 not in indexing_node.index['_id']

    # its child neither
    assert 2 not in indexing_node.index['_id']


def test_indexing_add_items_iterator(indexing_node):
    """
    Resulting Tree:
              R
             /
           10
          / \
        12  13
       /
     11
    """
    unsorted_stuff = [
        ('UPSERT', 10, '12', {'_id': 12}),
        ('UPSERT', 10, '13', {'_id': 13}),
        ('UPSERT', 12, '11', {'_id': 11}),
        ('UPSERT', 0, '10', {'_id': 10}),
        # this is an abandoned child, it can't be inserted
        ('UPSERT', 100, '110', {'_id': 110}),
        ('UPSERT', 100, '111', {'_id': 111})
    ]

    indexing_node.props['_id'] = 0
    abandoned_nodes = indexing_node.add_merge_set_with_id(unsorted_stuff, key='_id')

    assert indexing_node.index['_id'][11].parent.parent.parent is indexing_node
    assert indexing_node.index['_id'][12].parent.parent is indexing_node
    assert indexing_node.index['_id'][13].parent.parent is indexing_node
    assert indexing_node.index['_id'][10].parent is indexing_node

    assert unsorted_stuff[-2:] == abandoned_nodes


def test_duplicate_child(indexing_node):
    """ test the insertion of multiple nodes with the same name but different ids """
    indexing_node.add_child(name="Hello", props={'_id': '123'})
    with pytest.raises(DuplicateChildError):
        indexing_node.add_child(name="Hello", props={'_id': '456'})
    assert '123' in indexing_node.index['_id']
    assert '456' not in indexing_node.index['_id']


class MyFancyProperties(object):
    """ Costum properties object """

    __slots__ = ['_id', 'node']

    def __init__(self, node):
        self.node = node
        self._id = None

    @property
    def id(self):  # pylint: disable=invalid-name
        """ id attribute """
        return self._id

    @id.setter
    def id(self, value):  # pylint: disable=invalid-name
        self.node.on_update.send(self.node, other={'_id': value})
        self._id = value

    def get(self, key):
        """ get like the get from a dict """
        return getattr(self, key, None)

    def items(self):
        """ required for the indexing tree, like a dict """
        return [('_id', getattr(self, '_id', None))]

    def update(self, other, send_signal=True, destructive=False):
        """ update like the update from a dict """
        if send_signal:
            self.node.on_update.send(self.node, other=other)

        if 'id' in other:
            self._id = other['id']

    def __iter__(self):
        yield 'id'

    def __getitem__(self, item):
        return self.get(item)


@pytest.fixture
def indexing_node_cust_props():
    """ fixture to test the indexing node """
    return IndexingNode(None, indexes=['_id'], props_cls=MyFancyProperties)


@pytest.mark.skip('external properties not supported ATM')
def test_index_cust(indexing_node_cust_props):
    """ tests if costum properties are working with indexingnodes """
    child = indexing_node_cust_props.add_child(name='Der seppl')
    assert isinstance(child.props, MyFancyProperties)

    child.props.id = 123
    assert indexing_node_cust_props.index['_id'][123] == child

    with pytest.raises(KeyError):
        indexing_node_cust_props.add_child(name='Not seppl', props={'id': 123})

    child.delete()
    assert 123 not in indexing_node_cust_props.index['_id']


def test_pickle_indexed(indexing_node):
    """Check if a unpickled indexed node still has everything in place.

    This thes requires the environment variable ``PYTHONHASHSEED=17`` to be set to fail"""

    create_equal_distributed_tree(indexing_node, 10)
    # add a index attribute to all the nodes
    for num, node in zip(range(len(indexing_node)), indexing_node):
        node.props['_id'] = num

    fileobj = io.BytesIO()
    pickle.dump(indexing_node, fileobj)
    fileobj.seek(0)

    copied_root = pickle.load(fileobj)

    # check if all references are fine
    for node in copied_root:
        assert node.lock is copied_root.lock
        assert node.index is copied_root.index
        assert node.on_update is copied_root.on_update
        # assert node._on_dict_update is list(node.props.on_update.receivers.values())[0]()

        for signal in node.SIGNALS:
            assert getattr(node, signal) is getattr(copied_root, signal)

    # check if the index of the copied root is ok
    assert list(copied_root.index['_id'].keys()) == list(indexing_node.index['_id'].keys())
    copied_root.props['_id'] = 123

    print(copied_root.index['_id'])
    assert copied_root.index['_id'][123] is copied_root


@pytest.mark.parametrize('flat, test_size', [(True, 15000), (False, 100)])
def test_tree_performance(flat, test_size):
    """
    Tests the trees add performance.

    :param test_size How many node to create
    :param flat is set to False it is limited to the maximum recursion depth of python.
    """

    # create a deep storage model
    storage_model = Node(name=None)
    parent = storage_model

    start = time.time()
    for cnt in range(test_size):
        name = 'child_{}'.format(cnt)
        node = parent.add_child(name=name)
        # node.props['version_id'] = 0
        # node.props['is_dir'] = False
        if not flat:
            parent = node

    dur = time.time() - start
    size = asizeof(storage_model, limit=test_size + 1)
    print('creating {} nodes took {}, thats {}/per 100 nodes'.format(
        test_size, dur, dur / (test_size / 100)))
    print('creating {} nodes took {} Mbytes, thats {} bytes/per node'.format(
        test_size, size / 1024 / 1024, size / test_size))

    start = time.time()
    set(storage_model)
    print(time.time() - start)


def test_notifyingdict_deepcopy():
    """Check deepcopy of a NotifyingDict does not copy the whole tree"""
    node = mock.Mock()
    ndict = NotifyingDict()
    ndict['asd'] = {}
    ndict['asd']['asd'] = {}

    ndict_copy = deepcopy(ndict)

    assert isinstance(ndict_copy, NotifyingDict)
    assert ndict['asd'] == ndict_copy['asd']
    assert ndict['asd'] is not ndict_copy['asd']
    assert ndict['asd']['asd'] == ndict_copy['asd']['asd']
    assert ndict['asd']['asd'] is not ndict_copy['asd']['asd']


def test_notifyingdict_pop_default():
    """Check if defaultvalue is working."""
    node = mock.Mock()
    ndict = NotifyingDict()
    assert ndict.pop('bla', 'hey default') == 'hey default'
    assert ndict.pop('bla', None) is None


def test_notifyingdict_pop_keyerror():
    """Check popping to non existing key is throwing"""
    node = mock.Mock()
    ndict = NotifyingDict()
    with pytest.raises(KeyError):
        ndict.pop('bla')


def test_notifyingdict_update_other_none():
    """Check updating a NotifyingDict with other being None"""
    node = mock.Mock()
    ndict = NotifyingDict({'bla': 123})
    ndict_copy = deepcopy(ndict)
    ndict.update()

    assert set(ndict.items()) == set(ndict_copy.items())
